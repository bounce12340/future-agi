"""The aggregate-state parity module's ClickHouse target guard, with no socket.

``test_hourly_aggregate_state_exactness_ch25`` issues CREATE and DROP
DATABASE. These tests drive its target resolution with plain mappings, so the
process environment is never touched, and its database and table lifecycles
with a recording ``Client``, so no statement reaches any port.
"""

from __future__ import annotations

import pytest
from clickhouse_driver.errors import ServerException

from tracer.tests import test_hourly_aggregate_state_exactness_ch25 as parity

pytestmark = pytest.mark.unit

_IDENTITY_STATEMENT = "SELECT getMacro('replica')"


class _RecordingClients:
    """Stands in for ``clickhouse_driver.Client``; records every statement."""

    def __init__(self, replica_answer):
        self.replica_answer = replica_answer
        self.constructed: list[dict] = []
        self.statements: list[str] = []
        self.disconnects = 0

    def __call__(self, **kwargs):
        self.constructed.append(kwargs)
        return self

    def execute(self, statement, *args, **kwargs):
        self.statements.append(statement)
        if statement == _IDENTITY_STATEMENT:
            if isinstance(self.replica_answer, Exception):
                raise self.replica_answer
            return self.replica_answer
        return []

    def disconnect(self):
        self.disconnects += 1


def _ddl(statements: list[str]) -> list[str]:
    return [s for s in statements if s.lstrip().upper().startswith(("CREATE", "DROP"))]


@pytest.fixture()
def recording_clients(monkeypatch):
    def install(replica_answer=None):
        clients = _RecordingClients(replica_answer)
        monkeypatch.setattr(parity, "Client", clients)
        return clients

    return install


@pytest.mark.parametrize("variable", ["CH25_NATIVE_PORT", "CH_NATIVE_PORT"])
@pytest.mark.parametrize("port", sorted(parity._FORWARDED_PORTS))
def test_forwarded_port_from_a_general_variable_is_refused(variable, port):
    with pytest.raises(
        parity.UnsafeClickHouseTestTarget, match=f"forwarded port {port}\\b"
    ):
        parity._native_target({variable: str(port)})


@pytest.mark.parametrize("port", sorted(parity._FORWARDED_PORTS))
def test_github_actions_does_not_relax_the_refusal(port):
    with pytest.raises(parity.UnsafeClickHouseTestTarget):
        parity._native_target({"GITHUB_ACTIONS": "true", "CH25_NATIVE_PORT": str(port)})


@pytest.mark.parametrize(
    "environ",
    [{}, {"GITHUB_ACTIONS": "true"}, {"CI": "true"}, {"CH25_NATIVE_PORT": ""}],
    ids=["unset", "github_actions", "ci", "empty"],
)
def test_no_port_skips_and_never_defaults(environ):
    with pytest.raises(pytest.skip.Exception):
        parity._native_target(environ)


def test_a_local_port_needs_no_sidecar_proof():
    assert parity._native_target({"CH25_NATIVE_PORT": "39000"}) == (39000, False)
    assert parity._native_target({"CH_NATIVE_PORT": "39000"}) == (39000, False)


def test_the_opt_in_names_the_port_and_always_requires_the_sidecar_proof():
    assert parity._native_target({"FI_CH_PARITY_NATIVE_PORT": "19000"}) == (
        19000,
        True,
    )
    assert parity._native_target(
        {"FI_CH_PARITY_NATIVE_PORT": "39000", "CH25_NATIVE_PORT": "19010"}
    ) == (39000, True)


@pytest.mark.parametrize("port", ["19010", "19000"])
def test_refused_port_never_constructs_a_client(recording_clients, port):
    clients = recording_clients()

    with pytest.raises(parity.UnsafeClickHouseTestTarget):
        with parity._test_owned_database({"CH25_NATIVE_PORT": port}):
            pass

    assert clients.constructed == []
    assert clients.statements == []


def test_unset_port_skips_before_constructing_a_client(recording_clients):
    clients = recording_clients()

    with pytest.raises(pytest.skip.Exception):
        with parity._test_owned_database({}):
            pass

    assert clients.constructed == []


@pytest.mark.parametrize(
    "replica_answer",
    [
        [("chi-prod-0-0",)],
        [],
        ServerException("No macro replica in config", code=139),
    ],
    ids=["other_replica", "no_rows", "no_macro"],
)
def test_opt_in_without_the_sidecar_replica_is_refused_before_ddl(
    recording_clients, replica_answer
):
    clients = recording_clients(replica_answer)

    with pytest.raises(parity.UnsafeClickHouseTestTarget, match="test sidecar"):
        with parity._test_owned_database({"FI_CH_PARITY_NATIVE_PORT": "19000"}):
            pytest.fail("the database must not be created")

    assert [kwargs["port"] for kwargs in clients.constructed] == [19000]
    assert clients.statements == ["SELECT 1", _IDENTITY_STATEMENT]
    assert _ddl(clients.statements) == []
    assert clients.disconnects == 1


def test_opt_in_table_client_is_refused_before_ddl(recording_clients):
    clients = recording_clients([("chi-prod-0-0",)])

    with pytest.raises(parity.UnsafeClickHouseTestTarget, match="test sidecar"):
        with parity._test_spans_table(
            "test_agg_exact_0", {"FI_CH_PARITY_NATIVE_PORT": "19000"}
        ):
            pytest.fail("the table must not be created")

    assert clients.statements == [_IDENTITY_STATEMENT]
    assert clients.disconnects == 1


def test_opt_in_with_the_sidecar_replica_proves_it_before_ddl(recording_clients):
    """Positive control: the proof passes on the sidecar and precedes DDL."""
    clients = recording_clients([(parity._TEST_SIDECAR_REPLICA,)])

    with parity._test_owned_database({"FI_CH_PARITY_NATIVE_PORT": "19000"}) as database:
        with parity._test_spans_table(database, {"FI_CH_PARITY_NATIVE_PORT": "19000"}):
            pass

    assert clients.statements == [
        "SELECT 1",
        _IDENTITY_STATEMENT,
        f"CREATE DATABASE {database}",
        _IDENTITY_STATEMENT,
        parity._SPANS_DDL,
        "DROP TABLE IF EXISTS spans SYNC",
        f"DROP DATABASE IF EXISTS {database} SYNC",
    ]


def test_a_local_port_issues_no_identity_statement(recording_clients):
    clients = recording_clients()

    with parity._test_owned_database({"CH25_NATIVE_PORT": "39000"}) as database:
        pass

    assert [kwargs["port"] for kwargs in clients.constructed] == [39000]
    assert clients.statements == [
        "SELECT 1",
        f"CREATE DATABASE {database}",
        f"DROP DATABASE IF EXISTS {database} SYNC",
    ]
