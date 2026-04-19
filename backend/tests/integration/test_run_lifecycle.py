from __future__ import annotations

import json


def _parse_sse(response_text: str) -> list[dict]:
    events: list[dict] = []
    current_event = "message"
    for line in response_text.splitlines():
        if line.startswith("event: "):
            current_event = line.replace("event: ", "", 1).strip()
        elif line.startswith("data: "):
            payload = json.loads(line.replace("data: ", "", 1))
            events.append({"event": current_event, "data": payload})
    return events


def _patch_sufficient_retrieval(monkeypatch) -> None:
    import app.agents.nodes as nodes

    long_content = (
        "This source provides detailed, validated context for production rollout decisions, "
        "including scope, risks, mitigations, sequencing, monitoring, and rollback planning."
    )

    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query: {
            "query": query,
            "web_search_used": True,
            "source": "tavily",
            "results": [
                {"title": "Source A", "url": "https://example.com/a", "content": f"{long_content} A"},
                {"title": "Source B", "url": "https://example.com/b", "content": f"{long_content} B"},
                {"title": "Source C", "url": "https://example.com/c", "content": f"{long_content} C"},
            ],
        },
    )
    monkeypatch.setattr(
        nodes,
        "rerank",
        lambda query, documents: [
            {**documents[0], "cross_encoder_score": 9.8},
            {**documents[1], "cross_encoder_score": 9.2},
            {**documents[2], "cross_encoder_score": 8.7},
        ] if len(documents) >= 3 else [],
    )


def test_high_impact_run_pauses_for_approval(client, monkeypatch):
    _patch_sufficient_retrieval(monkeypatch)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Build a cloud migration strategy", "high_impact": True, "token_budget": 9000},
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0]["event"] == "run_started"
    assert any(event["event"] == "awaiting_approval" for event in events)


def test_approval_resume_completes_run(client, monkeypatch):
    _patch_sufficient_retrieval(monkeypatch)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Evaluate production rollout", "high_impact": True, "token_budget": 9000},
    )
    create_events = _parse_sse(create_response.text)
    run_id = create_events[0]["data"]["run_id"]

    resume_response = client.post(
        f"/api/runs/{run_id}/resume/stream",
        json={"decision": "approve", "reviewer": "ops_lead", "notes": "Ship with safeguards"},
    )
    assert resume_response.status_code == 200

    resume_events = _parse_sse(resume_response.text)
    assert resume_events[0]["event"] == "run_resumed"
    assert any(e["event"] == "run_finished" for e in resume_events)

    status = client.get(f"/api/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    timeline = client.get(f"/api/runs/{run_id}/timeline")
    assert timeline.status_code == 200
    assert len(timeline.json()["events"]) > 0


def test_stream_returns_immediate_cached_run_finished(client, monkeypatch):
    import app.api.routes as routes

    monkeypatch.setattr(
        routes,
        "get_cached_run_result",
        lambda session, objective, high_impact=False: {
            "run_id": "cached-run-1",
            "status": "completed",
            "current_node": "finalize",
            "output": "cached response",
            "cached": True,
        },
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("orchestrator.start_stream should not be called on cache hit")

    monkeypatch.setattr(routes.orchestrator, "start_stream", _should_not_run)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Duplicate objective", "high_impact": False, "token_budget": 9000},
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert len(events) == 1
    assert events[0]["event"] == "run_finished"
    assert events[0]["data"]["cached"] is True
    assert events[0]["data"]["output"] == "cached response"
