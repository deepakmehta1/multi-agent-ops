"""The seeded systems layer. These tests also guard the fixture data itself:
the scenario only works if the pull request, its failing CI run, its coverage
drop and the open incident all line up.
"""

from __future__ import annotations

import pytest

from agentops.core.exceptions import ToolExecutionError
from agentops.systems.store import get_store


@pytest.fixture
def store():
    """The shared seeded store."""
    return get_store()


def test_the_organisation_and_its_services(store):
    """The org and its three services exist, and an unknown service names the real ones."""
    assert store.organisation()["name"] == "Northwind Payments"
    names = {service["name"] for service in store.services()}
    assert names == {"checkout-api", "ledger-service", "notification-worker"}
    assert store.service("checkout-api")["tier"] == 1
    with pytest.raises(ToolExecutionError, match="Unknown service"):
        store.service("nope")


def test_pull_requests_can_be_filtered(store):
    """Pull requests filter by state and by repo, using the short or the full repo name."""
    assert len(store.pull_requests()) == 3
    assert len(store.pull_requests(state="open")) == 3
    # The repo filter accepts the bare name as well as owner/name.
    assert len(store.pull_requests(repo="checkout-api")) == 1
    assert len(store.pull_requests(repo="northwind/checkout-api")) == 1
    with pytest.raises(ToolExecutionError, match="No pull request"):
        store.pull_request(9999)


def test_the_diff_is_only_returned_when_asked_for(store):
    """A pull request omits each file's patch unless `include_diff` is set."""
    without = store.pull_request(482)
    assert all("patch" not in f for f in without["changed_files"])
    assert any("patch" in f for f in store.pull_request(482, include_diff=True)["changed_files"])


def test_pr_482_is_the_scenario_the_agents_have_to_solve(store):
    """The bug, the failing test and the coverage drop must agree."""
    pr = store.pull_request(482, include_diff=True)
    idempotency = next(f for f in pr["changed_files"] if "idempotency" in f["path"])
    assert "no unique constraint handling" in idempotency["patch"]

    run = store.ci_runs(pull_request=482)[0]
    assert run["status"] == "failure"
    failing = run["jobs"][-1]["failed_tests"][0]
    assert "concurrent_retries" in failing["test"]
    assert "duplicate key value" in failing["message"]

    coverage = store.coverage("checkout-api")
    assert coverage["total_percent"] < coverage["baseline_percent"]
    worst = min(coverage["files"], key=lambda f: f["percent"])
    assert "idempotency" in worst["path"]


def test_the_incident_lines_up_with_the_deploy_and_the_logs(store):
    """INC-204 opens after the deploy, with matching pool saturation and log lines."""
    incident = store.incidents(status="open")[0]
    assert incident["id"] == "INC-204"
    assert incident["service"] == "checkout-api"

    deploy = store.deployments(service="checkout-api")[0]
    # The deploy lands before the alert fires -- that correlation is the whole point.
    assert deploy["deployed_at"] < incident["opened_at"]

    saturation = store.metrics("checkout-api", metric="db_connection_pool_saturation_percent")
    assert saturation["db_connection_pool_saturation_percent"][-1]["value"] > 90
    assert store.logs(service="checkout-api", contains="pool exhausted")
    assert store.logs(level="error")


def test_unknown_metrics_report_what_is_available(store):
    """An unknown service or metric raises an error that lists the valid ones."""
    with pytest.raises(ToolExecutionError, match=r"known_services|No metrics"):
        store.metrics("nope")
    with pytest.raises(ToolExecutionError, match="No metric"):
        store.metrics("checkout-api", metric="nope")


def test_tickets_test_suites_and_coverage_are_queryable(store):
    """Tickets, test suites and coverage can be queried, and each miss has a clear message."""
    assert store.ticket("del-1123")["priority"] == "P1"
    assert store.tickets(team="payments-core")
    assert store.tickets(status="in review")
    assert "pytest" in store.test_suite("checkout-api")["conventions"]
    for missing, call in (
        ("No ticket", lambda: store.ticket("DEL-0000")),
        ("No coverage report", lambda: store.coverage("nope")),
        ("No test suite", lambda: store.test_suite("nope")),
    ):
        with pytest.raises(ToolExecutionError, match=missing):
            call()


def test_the_release_is_blocked_by_things_that_are_really_there(store):
    """The two blocked checklist items point at a real ticket and a real incident."""
    checklist = store.release_checklist()
    assert checklist["version"] == "2.14.0"
    blocked = [item for item in checklist["items"] if item["status"] == "blocked"]
    assert len(blocked) == 2
    # Each blocker must be verifiable against another system, not just asserted.
    assert any("DEL-1123" in (item["note"] or "") for item in blocked)
    assert any("INC-204" in (item["note"] or "") for item in blocked)
    assert store.merged_since_last_release(repo="checkout-api")


def test_oncall_covers_every_team(store):
    """Every delivery team has an on-call rota."""
    teams = {rota["team"] for rota in store.oncall()}
    assert teams == set(store.organisation()["delivery_teams"])
    assert store.oncall(team="payments-core")[0]["primary"] == "rkapoor"


def test_reads_cannot_mutate_the_seed(store):
    """Mutating a returned value does not change the seed (reads return deep copies)."""
    first = store.pull_request(482)
    first["title"] = "tampered"
    assert store.pull_request(482)["title"] != "tampered"
