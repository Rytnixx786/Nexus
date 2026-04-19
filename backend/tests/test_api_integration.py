from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import app.api.routes as api_routes
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from app.core.settings import settings
from app.db import repository
from conftest import TestingSessionLocal


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


def _install_fake_start_stream(monkeypatch, include_two_timeline: bool = False):
    def fake_start_stream(session, request):
        now = datetime.now(timezone.utc)
        run_id = uuid.uuid4().hex
        state = {
            "run_id": run_id,
            "objective": request.objective,
            "high_impact": request.high_impact,
            "status": "created",
            "current_node": "planner",
            "plan": "",
            "research_notes": [],
            "analysis": "",
            "draft": "",
            "critique": "",
            "final_output": "",
            "iteration_count": 0,
            "max_iterations": settings.max_iterations,
            "token_budget_remaining": request.token_budget,
            "run_deadline_epoch": 9_999_999_999.0,
            "require_human_approval": request.high_impact,
            "human_decision": "",
            "human_reviewer": "",
            "human_notes": "",
            "started_at": now,
            "updated_at": now,
            "trace": [],
        }

        repository.create_run(session, state)

        yield {
            "event": "run_started",
            "data": {
                "run_id": run_id,
                "status": "created",
                "current_node": "planner",
                "token_budget_remaining": request.token_budget,
            },
        }

        timeline_payloads = [
            {
                "seq": 1,
                "event_type": "plan_created",
                "node": "planner",
                "message": "Plan generated",
                "token_budget_remaining": request.token_budget - 100,
            }
        ]
        if include_two_timeline:
            timeline_payloads = [
                {
                    "seq": 2,
                    "event_type": "analysis_done",
                    "node": "analyst",
                    "message": "Analysis completed",
                    "token_budget_remaining": request.token_budget - 240,
                },
                {
                    "seq": 1,
                    "event_type": "plan_created",
                    "node": "planner",
                    "message": "Plan generated",
                    "token_budget_remaining": request.token_budget - 100,
                },
            ]

        for event in timeline_payloads:
            state["status"] = "running"
            state["current_node"] = event["node"]
            state["token_budget_remaining"] = event["token_budget_remaining"]
            state["updated_at"] = datetime.now(timezone.utc)
            repository.persist_step(
                session,
                state=state,
                seq=event["seq"],
                event_type=event["event_type"],
                node=event["node"],
                message=event["message"],
                data={},
            )

            yield {
                "event": "timeline",
                "data": {
                    "run_id": run_id,
                    "seq": event["seq"],
                    "status": "running",
                    "current_node": event["node"],
                    "token_budget_remaining": event["token_budget_remaining"],
                    "message": event["message"],
                    "event_type": event["event_type"],
                    "node": event["node"],
                    "data": {},
                },
            }

        state["status"] = "completed"
        state["current_node"] = "finalize"
        state["final_output"] = "Synthetic final output"
        state["updated_at"] = datetime.now(timezone.utc)
        repository.update_run(session, state)

        yield {
            "event": "run_finished",
            "data": {
                "run_id": run_id,
                "status": "completed",
                "current_node": "finalize",
                "output": state["final_output"],
                "token_budget_remaining": state["token_budget_remaining"],
            },
        }

    monkeypatch.setattr(api_routes.orchestrator, "start_stream", fake_start_stream)


def test_health_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_preflight_allows_streaming_headers(client):
    response = client.options(
        "/api/runs/stream",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,idempotency-key,last-event-id,x-request-id",
        },
    )

    assert response.status_code in {200, 204}
    allowed = response.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allowed
    assert "content-type" in allowed
    assert "idempotency-key" in allowed
    assert "last-event-id" in allowed
    assert "x-request-id" in allowed


def test_create_run_requires_auth(client, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    monkeypatch.setattr(settings, "api_key", "test-api-key")

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
    )
    assert response.status_code == 401


def test_create_run_stream_returns_sse(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_run_events_include_run_started(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )

    events = _parse_sse(response.text)
    assert events[0]["event"] == "run_started"


def test_run_events_include_run_finished(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )

    events = _parse_sse(response.text)
    assert events[-1]["event"] == "run_finished"


def test_get_run_returns_correct_status(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    events = _parse_sse(create_response.text)
    run_id = events[0]["data"]["run_id"]

    status_response = client.get(f"/api/runs/{run_id}", headers=auth_headers)
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert "estimated_cost_usd" in status_response.json()


def test_get_runs_returns_total_count(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    objectives = [
        "Build enterprise rollout plan for logistics migration",
        "Evaluate coral reef restoration policy outcomes and funding models",
    ]

    for objective in objectives:
        response = client.post(
            "/api/runs/stream",
            json={"objective": objective, "high_impact": False, "token_budget": 9000},
            headers=auth_headers,
        )
        assert response.status_code == 200

    paged = client.get("/api/runs?limit=1&offset=0", headers=auth_headers)
    assert paged.status_code == 200
    payload = paged.json()
    assert len(payload["runs"]) == 1
    assert payload["total"] == 2
    assert "estimated_cost_usd" in payload["runs"][0]


def test_get_runs_supports_cost_filters(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Cost filter coverage", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    assert create_response.status_code == 200

    only_free = client.get("/api/runs?min_cost_usd=0&max_cost_usd=0", headers=auth_headers)
    assert only_free.status_code == 200
    payload = only_free.json()
    assert payload["total"] >= 1
    assert all(float(run.get("estimated_cost_usd", 0.0)) <= 0.0 for run in payload["runs"])


def test_get_runs_supports_min_cost_for_nonzero_run(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Non-zero cost filter coverage", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    assert create_response.status_code == 200
    run_id = _parse_sse(create_response.text)[0]["data"]["run_id"]

    with TestingSessionLocal() as session:
        repository.append_token_usage(
            session,
            run_id=run_id,
            seq=99,
            node="planner",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=2000,
            completion_tokens=1000,
            total_tokens=3000,
            metering_mode="actual",
        )

    non_zero = client.get("/api/runs?min_cost_usd=0.0001", headers=auth_headers)
    assert non_zero.status_code == 200
    payload = non_zero.json()
    target = next((run for run in payload["runs"] if run.get("run_id") == run_id), None)
    assert target is not None
    assert float(target.get("estimated_cost_usd", 0.0)) > 0.0


def test_get_timeline_returns_events(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch, include_two_timeline=True)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    events = _parse_sse(create_response.text)
    run_id = events[0]["data"]["run_id"]

    timeline_response = client.get(f"/api/runs/{run_id}/timeline", headers=auth_headers)
    assert timeline_response.status_code == 200
    timeline_payload = timeline_response.json()
    timeline = timeline_payload["events"]
    assert [event["seq"] for event in timeline] == [1, 2]
    assert "estimated_cost_usd" in timeline_payload


def test_get_metrics_returns_expected_shape(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    assert response.status_code == 200

    metrics = client.get("/api/metrics", headers=auth_headers)
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["total_runs"] >= 1
    assert "runs_by_status" in payload
    assert "avg_token_usage_per_run" in payload
    assert "avg_steps_per_run" in payload
    assert "runs_last_24h" in payload
    assert "total_cost_usd" in payload
    assert "avg_cost_per_run_usd" in payload
    assert "cost_by_provider" in payload


def test_get_run_404_for_unknown(client, auth_headers):
    response = client.get("/api/runs/nonexistent", headers=auth_headers)
    assert response.status_code == 404


def test_resume_run_requires_awaiting_state(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    run_id = _parse_sse(create_response.text)[0]["data"]["run_id"]

    def eager_streaming_response(iterator, media_type=None, headers=None):
        try:
            for _ in iterator:
                pass
            return JSONResponse(status_code=200, content={"detail": "ok"})
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    monkeypatch.setattr(api_routes, "StreamingResponse", eager_streaming_response)

    response = client.post(
        f"/api/runs/{run_id}/resume/stream",
        json={"decision": "approve", "reviewer": "ops_lead", "notes": "Ship"},
        headers=auth_headers,
    )
    assert response.status_code == 409


def _install_fake_budget_exhausted_start_stream(monkeypatch):
    def fake_start_stream(session, request):
        now = datetime.now(timezone.utc)
        run_id = uuid.uuid4().hex
        state = {
            "run_id": run_id,
            "objective": request.objective,
            "high_impact": request.high_impact,
            "status": "budget_exhausted",
            "current_node": "finalize",
            "plan": "",
            "research_notes": [],
            "analysis": "",
            "draft": "",
            "critique": "",
            "final_output": "",
            "iteration_count": 1,
            "max_iterations": settings.max_iterations,
            "initial_token_budget": request.token_budget,
            "token_budget_remaining": 0,
            "run_deadline_epoch": 9_999_999_999.0,
            "require_human_approval": request.high_impact,
            "human_decision": "",
            "human_reviewer": "",
            "human_notes": "",
            "started_at": now,
            "updated_at": now,
            "trace": [
                {
                    "seq": 1,
                    "ts": now.isoformat(),
                    "event_type": "node_guard",
                    "node": "analyst",
                    "message": "Token budget exhausted",
                    "data": {"reason": "budget_exhausted"},
                }
            ],
        }
        repository.create_run(session, state)
        yield {
            "event": "run_started",
            "data": {"run_id": run_id, "status": "created", "current_node": "planner", "token_budget_remaining": request.token_budget},
        }
        yield {
            "event": "run_finished",
            "data": {
                "run_id": run_id,
                "status": "budget_exhausted",
                "current_node": "finalize",
                "output": "",
                "token_budget_remaining": 0,
            },
        }

    monkeypatch.setattr(api_routes.orchestrator, "start_stream", fake_start_stream)


def test_resume_budget_stream_requires_budget_exhausted_status(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
        headers=auth_headers,
    )
    run_id = _parse_sse(create_response.text)[0]["data"]["run_id"]

    response = client.post(
        f"/api/runs/{run_id}/resume-budget/stream",
        json={"additional_budget": 3000},
        headers=auth_headers,
    )
    assert response.status_code == 409


def test_resume_budget_stream_applies_additional_budget(client, auth_headers, monkeypatch):
    _install_fake_budget_exhausted_start_stream(monkeypatch)

    create_response = client.post(
        "/api/runs/stream",
        json={"objective": "Recover exhausted run", "high_impact": False, "token_budget": 2000},
        headers=auth_headers,
    )
    run_id = _parse_sse(create_response.text)[0]["data"]["run_id"]

    response = client.post(
        f"/api/runs/{run_id}/resume-budget/stream",
        json={"additional_budget": 4000},
        headers=auth_headers,
    )
    assert response.status_code == 200

    events = _parse_sse(response.text)
    assert events[0]["event"] == "run_resumed_budget"
    assert int(events[0]["data"]["additional_budget"]) == 4000
    assert int(events[0]["data"]["token_budget_remaining"]) >= 4000


def test_create_run_validates_objective_min_length(client, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", False)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "abcd", "high_impact": False, "token_budget": 9000},
    )
    assert response.status_code == 422


def test_create_run_validates_token_budget_min(client, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", False)

    response = client.post(
        "/api/runs/stream",
        json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 500},
    )
    assert response.status_code == 422


def test_rate_limit_triggers_429(client, auth_headers, monkeypatch):
    _install_fake_start_stream(monkeypatch)

    class _FixedWindowLimiter:
        def __init__(self, allowed: int) -> None:
            self.allowed = allowed
            self.calls = 0

        def check(self, _key: str) -> bool:
            self.calls += 1
            return self.calls <= self.allowed

        def is_available(self) -> bool:
            return True

    monkeypatch.setattr(api_routes, "limiter", _FixedWindowLimiter(60))

    response = None
    for _ in range(61):
        response = client.post(
            "/api/runs/stream",
            json={"objective": "Build rollout plan", "high_impact": False, "token_budget": 9000},
            headers=auth_headers,
        )

    assert response is not None
    assert response.status_code == 429


def test_upload_sources_returns_combined_context(client, auth_headers):
    files = [
        (
            "files",
            ("brief.txt", b"Launch strategy evidence from uploaded file", "text/plain"),
        )
    ]

    response = client.post("/api/uploads", files=files, headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["combined_chars"] > 0
    assert "Launch strategy evidence" in payload["combined_context"]
    assert payload["files"][0]["filename"] == "brief.txt"


def test_upload_sources_rejects_executable_masquerading_as_pdf(client, auth_headers):
    """Test that executable files with .pdf extension are rejected based on magic bytes."""
    # ZIP file magic bytes (will be detected as application/zip, not in allowlist)
    zip_bytes = b"PK\x03\x04" + b"\x00" * 100
    
    files = [
        (
            "files",
            ("malicious.pdf", zip_bytes, "application/pdf"),
        )
    ]

    response = client.post("/api/uploads", files=files, headers=auth_headers)

    assert response.status_code == 415
    payload = response.json()
    assert "Unsupported file type detected" in payload["detail"]


def test_upload_sources_accepts_valid_pdf(client, auth_headers):
    """Test that valid PDF files are accepted."""
    # Minimal valid PDF structure
    pdf_bytes = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000229 00000 n 
0000000310 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
400
%%EOF"""
    
    files = [
        (
            "files",
            ("valid.pdf", pdf_bytes, "application/pdf"),
        )
    ]

    response = client.post("/api/uploads", files=files, headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["files"]) == 1
    assert payload["files"][0]["filename"] == "valid.pdf"
