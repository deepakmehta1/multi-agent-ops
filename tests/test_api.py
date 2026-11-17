from __future__ import annotations

import json

from fastapi.testclient import TestClient

from agentops.core.config import Settings
from agentops.main import create_app


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"


def test_readyz_reports_where_credentials_come_from(client):
    body = client.get("/readyz").json()
    assert body == {
        "status": "ready",
        "llm_credentials": "caller-supplied",
        "agents": 4,
        "tools": 17,
    }


def test_agents_endpoint_describes_the_roster(client):
    body = client.get("/v1/agents").json()
    names = [agent["name"] for agent in body["agents"]]
    assert names == ["code_reviewer", "test_engineer", "incident_responder", "release_manager"]
    assert body["orchestrator"]["role"] == "Delivery Lead"
    assert all(agent["mandate"] and agent["tools"] for agent in body["agents"])


def test_tools_endpoint_links_tools_to_their_agents(client):
    tools = {tool["name"]: tool for tool in client.get("/v1/tools").json()}
    assert "code_reviewer" in tools["get_pull_request"]["used_by"]
    assert tools["search_logs"]["used_by"] == ["incident_responder"]
    assert tools["get_coverage_report"]["parameters"]["required"] == ["repo"]


def test_a_request_is_planned_delegated_and_answered(client):
    response = client.post("/v1/requests", json={"request": "Review PR 482 on checkout-api."})
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "succeeded"
    assert body["answer"]
    assert body["agents_used"] == ["code_reviewer"]
    assert body["plan"]["steps"][0]["agent"] == "code_reviewer"
    assert body["steps"][0]["status"] == "succeeded"
    assert body["trace"] is None


def test_the_trace_is_returned_on_request(client):
    body = client.post(
        "/v1/requests",
        json={"request": "Review PR 482 on checkout-api.", "include_trace": True},
    ).json()
    kinds = [event["type"] for event in body["trace"]]
    assert kinds[0] == "run.started"
    assert kinds[-1] == "run.completed"
    assert [event["seq"] for event in body["trace"]] == list(range(1, len(kinds) + 1))


def test_a_finished_run_can_be_fetched_again(client):
    run_id = client.post("/v1/requests", json={"request": "Review PR 482."}).json()["run_id"]

    assert run_id in client.get("/v1/runs").json()
    fetched = client.get(f"/v1/runs/{run_id}").json()
    assert fetched["run_id"] == run_id
    assert fetched["trace"], "a stored run keeps its trace"


def test_an_unknown_run_is_a_404(client):
    response = client.get("/v1/runs/run_does_not_exist")
    assert response.status_code == 404
    assert response.json()["code"] == "run_not_found"


def test_a_too_short_request_is_rejected(client):
    assert client.post("/v1/requests", json={"request": "hi"}).status_code == 422


def test_the_stream_emits_the_trace_then_the_result(client):
    with client.stream(
        "POST", "/v1/requests/stream", json={"request": "Review PR 482 on checkout-api."}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    events = _parse_sse(raw)
    kinds = [name for name, _ in events]
    assert kinds[0] == "run"
    assert kinds[-1] == "result"
    assert "trace" in kinds

    traced = [payload["type"] for name, payload in events if name == "trace"]
    assert "plan.created" in traced
    assert "step.started" in traced

    _, result = events[-1]
    assert result["status"] == "succeeded"
    assert result["answer"]


def test_without_any_key_the_service_says_so_rather_than_guessing(settings):
    """No server key and no header: a clear 401, not a confusing provider error."""
    app = create_app(settings.model_copy(update={"openai_api_key": None}))
    with TestClient(app) as bare:
        response = bare.post("/v1/requests", json={"request": "Review PR 482."})
    assert response.status_code == 401
    assert response.json()["code"] == "missing_credentials"


def test_the_openapi_schema_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) >= {
        "/healthz",
        "/readyz",
        "/v1/agents",
        "/v1/tools",
        "/v1/requests",
        "/v1/requests/stream",
        "/v1/runs",
        "/v1/runs/{run_id}",
    }


def test_responses_carry_a_correlation_id(client):
    response = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert client.get("/healthz").headers["X-Request-ID"]


def test_settings_defaults_are_sane():
    settings = Settings(openai_api_key=None)
    assert settings.max_plan_steps >= 1
    assert settings.max_parallel_steps >= 1
    assert not settings.has_server_key


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        name = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
        if name and payload is not None:
            events.append((name, payload))
    return events
