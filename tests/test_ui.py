"""The console is a static page; what can break is serving it and what it depends on."""

from __future__ import annotations

import re

from agentops.llm.fake import Reply
from helpers import agent_turn, is_first_plan, is_replan, plan_reply, step


def test_the_root_redirects_to_the_console(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_the_console_and_its_assets_are_served(client):
    page = client.get("/ui/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "multi-agent-ops" in page.text

    script = client.get("/ui/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]

    sheet = client.get("/ui/app.css")
    assert sheet.status_code == 200
    assert sheet.headers["content-type"].startswith("text/css")


def test_the_console_needs_nothing_from_the_internet(client):
    """It ships in a container that may have no egress: no CDN scripts, fonts or styles."""
    page = client.get("/ui/").text
    external = re.findall(r"""(?:src|href)\s*=\s*["'](https?://[^"']+)""", page)
    assert external == []

    for asset in ("/ui/app.js", "/ui/app.css"):
        body = client.get(asset).text
        assert not re.search(r"""(?:import\s|@import|url\()\s*["']?https?://""", body), asset


def test_the_console_is_not_part_of_the_api_schema(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
    assert not any(path.startswith("/ui") for path in paths)


def test_a_run_reports_the_request_it_answered(client, scripted):
    """The console labels history and replays with it."""
    scripted.when(is_first_plan, plan_reply(step("s1", "code_reviewer", "Review PR 482.")))
    scripted.when(agent_turn("code_reviewer"), Reply(content="Looks risky."))
    scripted.when(is_replan, plan_reply(), sticky=True)

    request = "Review PR 482 on checkout-api."
    created = client.post("/v1/requests", json={"request": request}).json()
    fetched = client.get(f"/v1/runs/{created['run_id']}").json()

    assert created["request"] == request
    assert fetched["request"] == request
