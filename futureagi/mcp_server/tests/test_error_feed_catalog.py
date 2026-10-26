"""Contract coverage for the error feed MCP tools (TH-8798).

The feed is read first and debugged second: list the issues, pull the data
behind one, then run and read its root-cause analysis. These tests pin the
operations those steps map onto, the safety hints a client shows before
calling them, and the page-size ceilings that keep a cluster's evidence
inside the MCP response limit.
"""

from uuid import uuid4

import pytest
from django.urls import resolve
from jsonschema import Draft7Validator

from mcp_server.generated_registry import GeneratedToolRegistry
from mcp_server.tests.test_tool_generation import CATALOG_PATH, CONTRACT_PATH
from mcp_server.tool_generation import generate_tool_manifest

OPERATIONS = [
    ("list_error_clusters", "GET", "/tracer/feed/issues/"),
    ("get_error_cluster", "GET", "/tracer/feed/issues/{cluster_id}/"),
    (
        "get_error_cluster_overview",
        "GET",
        "/tracer/feed/issues/{cluster_id}/overview/",
    ),
    ("list_error_cluster_traces", "GET", "/tracer/feed/issues/{cluster_id}/traces/"),
    (
        "get_error_cluster_root_cause",
        "GET",
        "/tracer/feed/issues/{cluster_id}/root-cause/",
    ),
    (
        "analyze_error_cluster_trace",
        "POST",
        "/tracer/feed/issues/{cluster_id}/deep-analysis/",
    ),
]


@pytest.fixture(scope="module")
def registry():
    return GeneratedToolRegistry.from_manifest()


@pytest.fixture(scope="module")
def generated():
    return {
        tool["name"]: tool
        for tool in generate_tool_manifest(CONTRACT_PATH, CATALOG_PATH)["tools"]
    }


def test_error_feed_group_is_exactly_the_read_and_debug_surface(registry):
    """Triage writes stay out: the group reads the feed and debugs a cluster.

    Changing an issue's status, severity or assignee, and filing it to
    Linear, are deliberately not exposed — an agent that could resolve an
    issue could hide a live regression from the humans watching the feed.
    """
    assert {tool.name for tool in registry.list_by_groups(["error_feed"])} == {
        name for name, _, _ in OPERATIONS
    }


@pytest.mark.parametrize("name,method,path", OPERATIONS)
def test_tools_reuse_mounted_feed_operations(registry, name, method, path):
    tool = registry.get(name)

    assert tool.request["method"] == method
    assert tool.request["path"] == path
    assert tool.is_available()

    concrete = path
    for field in tool.request["parameters"]["path"]:
        concrete = concrete.replace("{" + field + "}", str(uuid4()))
    view = resolve(concrete).func
    cls = getattr(view, "cls", None) or getattr(view, "view_class", None)
    assert hasattr(cls, method.lower())


@pytest.mark.parametrize("name,method,_path", OPERATIONS)
def test_reading_the_feed_is_annotated_read_only(registry, name, method, _path):
    annotations = registry.get(name).annotations
    assert annotations["readOnlyHint"] is (method == "GET")
    if method == "GET":
        assert annotations["destructiveHint"] is False
        assert annotations["openWorldHint"] is False


def test_dispatching_an_analysis_declares_its_cost_and_its_overwrite(generated):
    """The one write in the group must not look like a cached read.

    POST /deep-analysis/ dispatches model work the organization pays for, and
    `force=true` deletes the stored analysis before re-running it. A client
    that saw a plain write hint would run it unprompted on every cluster.
    """
    tool = generated["analyze_error_cluster_trace"]

    assert tool["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    }
    schema = tool["inputSchema"]
    assert sorted(schema["required"]) == ["cluster_id", "trace_id"]
    # Re-running is opt-in: omitting force must reuse the stored analysis.
    assert schema["properties"]["force"]["default"] is False


def test_root_cause_reads_are_scoped_to_one_trace(generated):
    """A cluster groups many traces and the analysis is stored per trace.

    Letting `trace_id` be optional would make the tool answer about whichever
    trace the view happened to pick, which reads as the cluster's root cause.
    """
    schema = generated["get_error_cluster_root_cause"]["inputSchema"]

    assert sorted(schema["required"]) == ["cluster_id", "trace_id"]
    assert not Draft7Validator(schema).is_valid({"cluster_id": str(uuid4())})


@pytest.mark.parametrize(
    "name,field,maximum,rest_maximum",
    [
        ("get_error_cluster_overview", "rep_limit", 20, 200),
        ("list_error_cluster_traces", "limit", 100, 500),
    ],
)
def test_cluster_evidence_pages_are_capped_below_the_rest_bound(
    generated, name, field, maximum, rest_maximum
):
    """Cluster evidence carries whole trace inputs and outputs.

    The REST bounds let a caller ask for a page far larger than the 64KB MCP
    response limit, which only surfaces as a size error after the query runs.
    """
    schema = generated[name]["inputSchema"]["properties"][field]
    validator = Draft7Validator(schema)

    assert maximum < rest_maximum
    assert schema["maximum"] == maximum
    assert validator.is_valid(maximum)
    assert not validator.is_valid(maximum + 1)
    # The ceiling must not reject the default the view applies on its own.
    assert validator.is_valid(schema["default"])
