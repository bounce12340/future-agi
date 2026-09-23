"""The root conftest's live ClickHouse test-target guard, with no socket.

Live test modules take their ClickHouse clients from the conftest helpers and
then issue DDL. These tests drive the port resolution with plain mappings, so
the process environment is never touched, and the client lifecycles with a
recording ``Client``, so no statement reaches any port. A source scan pins each
live module to those helpers, so no module resolves a port of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import clickhouse_driver
import pytest
from clickhouse_driver.errors import ServerException

from conftest import (
    _FORWARDED_CH_PORTS,
    UnsafeClickHouseTestTarget,
    _ch_test_native_client,
    _ch_test_native_port,
    _ch_test_owned_database,
    _open_ch_test_native_client,
)

pytestmark = pytest.mark.unit

_IDENTITY_STATEMENT = "SELECT getMacro('replica')"
_OPT_IN = "FI_CH_TEST_SIDECAR_NATIVE_PORT"
_GENERAL = ("CH25_NATIVE_PORT", "CH25_TCP_PORT", "CH_NATIVE_PORT")
_SIDECAR_ANSWER = [("test-01",)]
_FOREIGN_ANSWERS = pytest.mark.parametrize(
    "replica_answer",
    [
        [("chi-prod-0-0",)],
        [],
        ServerException("No macro replica in config", code=139),
    ],
    ids=["other_replica", "no_rows", "no_macro"],
)

# Every module that opens a live ClickHouse client to issue DDL or INSERTs.
_FUTUREAGI = Path(__file__).resolve().parents[1]
_LIVE_MODULES = (
    "tracer/tests/test_hourly_aggregate_state_exactness_ch25.py",
    "tracer/tests/test_trace_conjunction_seed_gate_ch25.py",
    "tracer/tests/test_users_matching_walk_ch25.py",
    "tracer/tests/test_users_matching_walk_differential_ch25.py",
    "tracer/tests/test_users_seeded_page_read_settings_ch25.py",
)


class _RecordingClient:
    def __init__(self, log, replica_answer, kwargs):
        self._log = log
        self._replica_answer = replica_answer
        self.kwargs = kwargs
        self.statements: list[str] = []
        self.disconnects = 0

    def execute(self, statement, *args, **kwargs):
        self.statements.append(statement)
        self._log.append(statement)
        if statement == _IDENTITY_STATEMENT:
            if isinstance(self._replica_answer, Exception):
                raise self._replica_answer
            return self._replica_answer
        return []

    def disconnect(self):
        self.disconnects += 1


class _RecordingClients:
    """Stands in for ``clickhouse_driver.Client``; records every statement."""

    def __init__(self, replica_answer):
        self.replica_answer = replica_answer
        self.clients: list[_RecordingClient] = []
        self.statements: list[str] = []

    def __call__(self, **kwargs):
        client = _RecordingClient(self.statements, self.replica_answer, kwargs)
        self.clients.append(client)
        return client

    @property
    def disconnects(self) -> int:
        return sum(client.disconnects for client in self.clients)


def _is_ddl(statement: str) -> bool:
    return statement.lstrip().upper().startswith(("CREATE", "DROP", "INSERT"))


@pytest.fixture()
def recording_clients(monkeypatch):
    monkeypatch.delenv("FI_ALLOW_NONLOCAL_CH25_TEST_MUTATIONS", raising=False)

    def install(replica_answer=None):
        clients = _RecordingClients(replica_answer)
        monkeypatch.setattr(clickhouse_driver, "Client", clients)
        return clients

    return install


def _open_owned_database(environ):
    with _ch_test_owned_database("test_guard_", environ=environ):
        pass


def _open_client(environ):
    _open_ch_test_native_client(environ=environ).disconnect()


def _open_client_context(environ):
    with _ch_test_native_client(database="test_guard_0", environ=environ):
        pass


_ENTRY_POINTS = pytest.mark.parametrize(
    "open_target",
    [_open_owned_database, _open_client, _open_client_context],
    ids=["owned_database", "open_client", "client_context"],
)


@pytest.mark.parametrize("variable", _GENERAL)
@pytest.mark.parametrize("port", sorted(_FORWARDED_CH_PORTS))
def test_forwarded_port_from_a_general_variable_is_refused(variable, port):
    with pytest.raises(UnsafeClickHouseTestTarget, match=f"forwarded port {port}\\b"):
        _ch_test_native_port({variable: str(port)})


@pytest.mark.parametrize("port", sorted(_FORWARDED_CH_PORTS))
def test_github_actions_does_not_relax_the_refusal(port):
    with pytest.raises(UnsafeClickHouseTestTarget):
        _ch_test_native_port({"GITHUB_ACTIONS": "true", "CH25_NATIVE_PORT": str(port)})


def test_an_empty_variable_does_not_hide_a_forwarded_one_behind_it():
    with pytest.raises(UnsafeClickHouseTestTarget, match="forwarded port 19000"):
        _ch_test_native_port({"CH25_NATIVE_PORT": "", "CH25_TCP_PORT": "19000"})


@pytest.mark.parametrize(
    "environ",
    [{}, {"GITHUB_ACTIONS": "true"}, {"CI": "true"}, {"CH25_NATIVE_PORT": " "}],
    ids=["unset", "github_actions", "ci", "blank"],
)
def test_no_port_skips_and_never_defaults(environ):
    with pytest.raises(pytest.skip.Exception, match=_OPT_IN):
        _ch_test_native_port(environ)


@pytest.mark.parametrize("variable", _GENERAL)
def test_a_local_port_needs_no_sidecar_proof(variable):
    assert _ch_test_native_port({variable: "39000"}) == (39000, False)


def test_the_opt_in_names_the_port_and_always_requires_the_sidecar_proof():
    assert _ch_test_native_port({_OPT_IN: "19000"}) == (19000, True)
    assert _ch_test_native_port({_OPT_IN: "39000", "CH25_NATIVE_PORT": "19010"}) == (
        39000,
        True,
    )


@_ENTRY_POINTS
@pytest.mark.parametrize("port", ["19010", "19000"])
def test_refused_port_never_constructs_a_client(recording_clients, open_target, port):
    clients = recording_clients()

    with pytest.raises(UnsafeClickHouseTestTarget):
        open_target({"CH25_NATIVE_PORT": port})

    assert clients.clients == []


@_ENTRY_POINTS
def test_unset_port_skips_before_constructing_a_client(recording_clients, open_target):
    clients = recording_clients()

    with pytest.raises(pytest.skip.Exception):
        open_target({})

    assert clients.clients == []


@_ENTRY_POINTS
def test_non_loopback_host_is_refused_before_constructing_a_client(
    recording_clients, open_target
):
    clients = recording_clients()

    with pytest.raises(UnsafeClickHouseTestTarget, match="non-loopback"):
        open_target({"CH25_HOST": "clickhouse", "CH25_NATIVE_PORT": "39000"})

    assert clients.clients == []


@_FOREIGN_ANSWERS
@_ENTRY_POINTS
def test_opt_in_without_the_sidecar_replica_is_refused_before_any_ddl(
    recording_clients, open_target, replica_answer
):
    clients = recording_clients(replica_answer)

    with pytest.raises(UnsafeClickHouseTestTarget, match="test sidecar"):
        open_target({_OPT_IN: "19000"})

    assert [client.kwargs["port"] for client in clients.clients] == [19000]
    assert clients.statements == ["SELECT 1", _IDENTITY_STATEMENT]
    assert clients.disconnects == 1


def test_opt_in_with_the_sidecar_replica_proves_it_before_ddl(recording_clients):
    """Positive control: every client proves the sidecar before its first DDL."""
    clients = recording_clients(_SIDECAR_ANSWER)
    environ = {_OPT_IN: "19000", "CH25_NATIVE_PORT": "19010"}

    with _ch_test_owned_database("test_guard_", environ=environ) as database:
        with _ch_test_native_client(database=database, environ=environ) as client:
            client.execute("CREATE TABLE spans (id String) ENGINE = Memory")

    assert clients.statements == [
        "SELECT 1",
        _IDENTITY_STATEMENT,
        f"CREATE DATABASE {database}",
        "SELECT 1",
        _IDENTITY_STATEMENT,
        "CREATE TABLE spans (id String) ENGINE = Memory",
        f"DROP DATABASE IF EXISTS {database} SYNC",
    ]
    assert [client.kwargs["port"] for client in clients.clients] == [19000, 19000]
    for client in clients.clients:
        first_ddl = next(i for i, s in enumerate(client.statements) if _is_ddl(s))
        assert _IDENTITY_STATEMENT in client.statements[:first_ddl]
        assert client.disconnects == 1


def test_a_local_port_issues_no_identity_statement(recording_clients):
    clients = recording_clients()

    with _ch_test_owned_database("test_guard_", environ={"CH25_NATIVE_PORT": "39000"}):
        pass

    assert [client.kwargs["port"] for client in clients.clients] == [39000]
    assert [s.split(" test_guard_")[0] for s in clients.statements] == [
        "SELECT 1",
        "CREATE DATABASE",
        "DROP DATABASE IF EXISTS",
    ]


def test_client_options_pass_through(recording_clients):
    clients = recording_clients()
    environ = {"CH_NATIVE_PORT": "39000", "CH_USERNAME": "lane", "CH_PASSWORD": "pw"}

    _open_ch_test_native_client(
        environ=environ, connect_timeout=2, send_receive_timeout=10
    ).disconnect()

    (client,) = clients.clients
    assert client.kwargs == {
        "host": "127.0.0.1",
        "port": 39000,
        "database": "default",
        "user": "lane",
        "password": "pw",
        "connect_timeout": 2,
        "send_receive_timeout": 10,
    }


@pytest.mark.parametrize("module", _LIVE_MODULES)
def test_live_module_takes_its_clients_from_the_guard(module):
    source = (_FUTUREAGI / module).read_text()

    assert not re.search(r"\b(19010|19000|19001|19002|1823[0-2])\b", source)
    assert not re.search(
        r"\b(CH25_NATIVE_PORT|CH25_TCP_PORT|CH_NATIVE_PORT|CH25_HTTP_PORT"
        r"|CH_HTTP_PORT|CH_PORT)\b",
        source,
    )
    assert not re.search(r"(?<!\w)Client\(|get_client\(", source)
    assert re.search(
        r"\b(_ch_test_owned_database|_ch_test_native_client"
        r"|_open_ch_test_native_client)\(",
        source,
    )
