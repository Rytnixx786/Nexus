from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone
from hashlib import sha256

import magic
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi import Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.logging import current_request_id, get_logger
from app.core.auth import AuthContext, enforce_role, require_auth_context
from app.core.rate_limiter import create_rate_limiter
from app.core.models import (
    ApprovalDecisionRequest,
    BudgetResumeRequest,
    RunCreateRequest,
    RunListResponse,
    StopRunRequest,
    RunStatusResponse,
    RunTimelineResponse,
    TimelineEvent,
)
from app.core.orchestrator import Orchestrator, sse_pack
from app.core.cache import get_cached_run_result, save_run_result_to_cache
from app.core.settings import settings
from app.db import repository
from app.db.session import get_session

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import docx2txt
except ImportError:  # pragma: no cover
    docx2txt = None


router = APIRouter(prefix="/api", tags=["nexus"])
orchestrator = Orchestrator()
logger = get_logger(__name__)
limiter = create_rate_limiter(settings)
TERMINAL_STATUSES = {"completed", "failed", "stopped", "rejected", "timeout", "budget_exhausted"}


def _validate_run_id(run_id: str) -> None:
    """Validate run_id format at API boundary.
    
    run_id must be:
    - non-empty and max 64 characters
    - alphanumeric (letters, digits, hyphens, underscores)
    """
    if not run_id or not isinstance(run_id, str):
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    
    if len(run_id) > 64:
        raise HTTPException(status_code=400, detail="run_id exceeds maximum length of 64")
    
    # Check for safe characters: alphanumeric, hyphen, underscore
    if not all(c.isalnum() or c in '-_' for c in run_id):
        raise HTTPException(status_code=400, detail="run_id contains invalid characters")


def _as_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _extract_upload_text(filename: str, payload: bytes) -> str:
    """Extract text from uploaded file with guaranteed cleanup."""
    suffix = Path(filename or "").suffix.lower()

    if suffix == ".pdf" and PdfReader is not None:
        reader = PdfReader(BytesIO(payload))
        chunks = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(chunks).strip()

    if suffix == ".docx" and docx2txt is not None:
        temp_path = None
        try:
            with NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
                tmp_file.write(payload)
                temp_path = tmp_file.name
            result = str(docx2txt.process(temp_path) or "").strip()
            return result
        except Exception as exc:
            logger.error("Failed to extract docx text", extra={"filename": filename, "error": str(exc)})
            raise HTTPException(status_code=422, detail="Failed to extract text from document") from exc
        finally:
            # Guarantee cleanup even if extraction fails
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Failed to remove temporary upload file", extra={"path": temp_path, "error": str(exc)})

    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        logger.warning("UTF-8 decoding failed, falling back to latin-1", extra={"error": str(exc)})
        try:
            return payload.decode("latin-1").strip()
        except (UnicodeDecodeError, AttributeError) as fallback_exc:
            logger.error("Latin-1 decoding also failed", extra={"error": str(fallback_exc)})
            return payload.decode("latin-1", errors="replace").strip()  # Replace with U+FFFD


@router.post("/uploads")
async def upload_sources(
    files: list[UploadFile] = File(...),
    auth: AuthContext = Depends(require_auth_context),
) -> dict[str, object]:
    enforce_role(auth, {"admin", "operator", "reviewer"})
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > settings.upload_max_files:
        raise HTTPException(status_code=400, detail=f"Too many files. Max allowed: {settings.upload_max_files}")

    # Allowed MIME types
    ALLOWED_MIME_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }

    extracted_parts: list[str] = []
    uploaded: list[dict[str, object]] = []

    for file in files:
        payload = await file.read()
        
        # Validate MIME type using magic bytes
        detected_mime = magic.from_buffer(payload, mime=True)
        if detected_mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type detected"
            )
        
        text = _extract_upload_text(file.filename or "uploaded_file", payload)
        if not text:
            text = f"[No extractable text found in {file.filename or 'uploaded_file'}]"
        uploaded.append({
            "filename": file.filename or "uploaded_file",
            "size_bytes": len(payload),
            "chars_extracted": len(text),
        })
        extracted_parts.append(f"## {file.filename or 'uploaded_file'}\n{text}")

    combined = "\n\n".join(extracted_parts)
    max_chars = max(1000, int(settings.upload_context_max_chars))
    context = combined[:max_chars].strip()

    return {
        "files": uploaded,
        "combined_context": context,
        "combined_chars": len(context),
        "truncated": len(combined) > len(context),
    }


def require_auth_or_context(
    auth: AuthContext = Depends(require_auth_context),
) -> AuthContext:
    return auth


def optional_auth(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuthContext:
    """Optional authentication - returns a default context if auth fails.
    
    Useful for metrics and other endpoints that don't strictly require authentication
    but respect auth if provided.
    """
    try:
        auth = require_auth_context(x_api_key=x_api_key, authorization=authorization)
        return auth
    except HTTPException:
        # If authentication fails, return a default operator context
        # This allows metrics to be accessed without authentication in development
        logger.debug("Metrics accessed without authentication - using default operator role")
        return AuthContext(subject="anonymous", role="operator", raw={})


def throttle(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"

    if not limiter.check(client_ip):
        logger.warning(
            "Rate limit exceeded",
            extra={
                "client_ip": client_ip,
                "correlation_id": current_request_id(),
            },
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ratelimit")
def health_ratelimit() -> dict[str, object]:
    limiter_stats = limiter.stats() if hasattr(limiter, "stats") else {
        "fail_open_count": 0,
        "consecutive_fail_open_count": 0,
        "last_fail_open_error": "",
    }
    return {
        "redis_available": limiter.is_available(),
        "limit_per_minute": settings.run_requests_per_minute,
        "fail_open_count": limiter_stats["fail_open_count"],
        "consecutive_fail_open_count": limiter_stats["consecutive_fail_open_count"],
        "last_fail_open_error": limiter_stats["last_fail_open_error"],
    }


@router.get("/metrics")
def get_metrics(
    auth: AuthContext = Depends(optional_auth),
    session: Session = Depends(get_session),
) -> dict:
    # Metrics endpoint allows any authenticated user or defaults to operator role
    # This makes metrics accessible in development environments without strict JWT requirements
    enforce_role(auth, {"admin", "operator", "reviewer"})
    return repository.get_system_metrics(session)


@router.get("/runs", response_model=RunListResponse)
def get_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, min_length=1, max_length=200),
    status: str | None = Query(default=None),
    started_from: datetime | None = Query(default=None),
    started_to: datetime | None = Query(default=None),
    min_cost_usd: float | None = Query(default=None, ge=0),
    max_cost_usd: float | None = Query(default=None, ge=0),
    auth: AuthContext = Depends(require_auth_or_context),
    session: Session = Depends(get_session),
) -> RunListResponse:
    enforce_role(auth, {"admin", "operator", "reviewer"})
    records = repository.list_runs(
        session,
        limit=limit,
        offset=offset,
        search=search,
        status=status,
        started_from=started_from,
        started_to=started_to,
        min_cost_usd=min_cost_usd,
        max_cost_usd=max_cost_usd,
    )
    total = repository.count_runs(
        session,
        search=search,
        status=status,
        started_from=started_from,
        started_to=started_to,
        min_cost_usd=min_cost_usd,
        max_cost_usd=max_cost_usd,
    )
    quota_subjects = {
        str(record.state_json.get("quota_subject", auth.subject))
        for record in records
    }
    quota_windows = repository.get_existing_daily_quotas(session, quota_subjects)
    runs = []
    for record in records:
        token_totals = repository.get_token_totals(session, record.run_id)
        quota_subject = str(record.state_json.get("quota_subject", auth.subject))
        quota = quota_windows.get(quota_subject)
        runs.append(RunStatusResponse(
            run_id=record.run_id,
            status=record.status,
            current_node=record.current_node,
            objective=record.objective,
            high_impact=record.high_impact,
            iteration_count=record.iteration_count,
            initial_token_budget=record.initial_token_budget,
            token_budget_remaining=record.token_budget_remaining,
            estimated_cost_usd=float(record.estimated_cost_usd or 0.0),
            latest_checkpoint_seq=None,
            latest_checkpoint_at=None,
            started_at=_as_utc(record.started_at),
            updated_at=_as_utc(record.updated_at),
            plan=str(record.state_json.get("plan", "") or ""),
            research_notes=list(record.state_json.get("research_notes", []) or []),
            analysis=str(record.state_json.get("analysis", "") or ""),
            draft=str(record.state_json.get("draft", "") or ""),
            critique=str(record.state_json.get("critique", "") or ""),
            final_output=str(record.state_json.get("final_output", "") or ""),
            human_decision=str(record.state_json.get("human_decision", "") or ""),
            human_reviewer=str(record.state_json.get("human_reviewer", "") or ""),
            human_notes=str(record.state_json.get("human_notes", "") or ""),
            metering_mode=str(token_totals.get("metering_mode", "estimated")),
            prompt_tokens_total=int(token_totals.get("prompt_tokens_total", 0)),
            completion_tokens_total=int(token_totals.get("completion_tokens_total", 0)),
            total_tokens_used=int(token_totals.get("total_tokens_used", 0)),
            quota_subject=quota_subject,
            quota_daily_limit=settings.quota_daily_tokens,
            quota_daily_used=int(quota.tokens_used) if quota is not None else 0,
            output=record.output,
        ))
    return RunListResponse(runs=runs, total=total)


@router.post("/runs/stream")
def create_run_stream(
    payload: RunCreateRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    auth: AuthContext = Depends(require_auth_or_context),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    enforce_role(auth, {"admin", "operator"})
    throttle(request)
    quota = repository.get_or_create_daily_quota(session, auth.subject)
    if settings.token_ledger_v2 and quota.tokens_used >= settings.quota_daily_tokens:
        raise HTTPException(status_code=402, detail="Daily token quota exceeded")

    cached_result = get_cached_run_result(
        session=session,
        objective=payload.objective,
        high_impact=payload.high_impact,
    )
    if cached_result is not None:
        logger.info(
            "Semantic cache hit for run stream",
            extra={
                "cache_hit": True,
                "objective_length": len(payload.objective),
                "high_impact": payload.high_impact,
                "correlation_id": current_request_id(),
            },
        )

        def cached_gen():
            cached_payload = dict(cached_result)
            cached_payload["cached"] = True
            yield sse_pack({"event": "run_finished", "data": cached_payload})

        return StreamingResponse(
            cached_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    logger.info(
        "Semantic cache miss for run stream",
        extra={
            "cache_hit": False,
            "objective_length": len(payload.objective),
            "high_impact": payload.high_impact,
            "correlation_id": current_request_id(),
        },
    )

    if settings.sse_resume_v2 and idempotency_key:
        dedupe_hash = sha256(f"start:{auth.subject}:{idempotency_key}:{payload.objective}:{payload.token_budget}".encode()).hexdigest()
        cached = repository.get_idempotency_record(session, "start", dedupe_hash)
        if cached:
            run_id = str(cached.run_id)
            def replay_gen():
                for event in repository.get_timeline(session, run_id):
                    yield sse_pack({"event": "timeline", "data": {
                        "run_id": event.run_id, "seq": event.seq, "status": cached.status, "current_node": cached.current_node,
                        "message": event.message, "event_type": event.event_type, "node": event.node, "data": event.data,
                    }})
            return StreamingResponse(replay_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def event_gen():
        try:
            stream_iter = orchestrator.start_stream(session, payload, actor_subject=auth.subject)
        except TypeError:
            stream_iter = orchestrator.start_stream(session, payload)
        for event in stream_iter:
            if event.get("event") == "run_finished":
                event_data = dict(event.get("data", {}))
                if str(event_data.get("status", "")) == "completed":
                    save_run_result_to_cache(
                        session=session,
                        objective=payload.objective,
                        result=event_data,
                        ttl_days=7,
                        high_impact=payload.high_impact,
                    )
            if event.get("event") == "run_started":
                event_data = event.get("data", {})
                if settings.sse_resume_v2 and idempotency_key:
                    dedupe_hash = sha256(f"start:{auth.subject}:{idempotency_key}:{payload.objective}:{payload.token_budget}".encode()).hexdigest()
                    repository.upsert_idempotency_record(
                        session=session,
                        scope="start",
                        key_hash=dedupe_hash,
                        run_id=str(event_data.get("run_id", "")),
                        status=str(event_data.get("status", "created")),
                        current_node=str(event_data.get("current_node", "planner")),
                        ttl_minutes=settings.idempotency_ttl_minutes,
                    )
                logger.info(
                    "Run creation accepted",
                    extra={
                        "run_id": event_data.get("run_id", ""),
                        "objective_length": len(payload.objective),
                        "high_impact": payload.high_impact,
                        "client_ip": request.client.host if request.client else "unknown",
                        "correlation_id": current_request_id(),
                    },
                )
            yield sse_pack(event)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs/{run_id}/resume/stream")
def resume_run_stream(
    run_id: str,
    payload: ApprovalDecisionRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    auth: AuthContext = Depends(require_auth_or_context),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    enforce_role(auth, {"admin", "reviewer"})
    throttle(request)
    _validate_run_id(run_id)

    logger.info(
        "Run resume requested",
        extra={
            "run_id": run_id,
            "decision": payload.decision,
            "reviewer": payload.reviewer,
            "correlation_id": current_request_id(),
        },
    )

    state = repository.get_run_state(session, run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if state.get("status") != "awaiting_human" or state.get("current_node") != "human_approval":
        raise HTTPException(status_code=409, detail="Run is not awaiting a human decision")
    if state.get("human_decision"):
        raise HTTPException(status_code=409, detail="Human decision already recorded")
    if settings.sse_resume_v2 and idempotency_key:
        resume_hash = sha256(f"resume:{auth.subject}:{run_id}:{idempotency_key}:{payload.decision}:{payload.notes}".encode()).hexdigest()
        if repository.get_idempotency_record(session, "resume", resume_hash):
            return StreamingResponse(iter([sse_pack({"event": "noop", "data": {"run_id": run_id, "message": "Duplicate resume ignored"}})]), media_type="text/event-stream")

    def event_gen():
        if settings.sse_resume_v2 and last_event_id:
            try:
                seq_floor = int(last_event_id)
                missed = [evt for evt in repository.get_timeline(session, run_id) if int(evt.seq) > seq_floor]
                for evt in missed:
                    yield sse_pack({"event": "timeline", "data": {
                        "run_id": evt.run_id, "seq": evt.seq, "status": state.get("status", "running"), "current_node": state.get("current_node", ""),
                        "message": evt.message, "event_type": evt.event_type, "node": evt.node, "data": evt.data,
                    }})
            except ValueError:
                pass
        try:
            stream_iter = orchestrator.resume_stream(session, run_id, payload, actor_subject=auth.subject)
        except TypeError:
            stream_iter = orchestrator.resume_stream(session, run_id, payload)
        for event in stream_iter:
            yield sse_pack(event)
        if settings.sse_resume_v2 and idempotency_key:
            resume_hash = sha256(f"resume:{auth.subject}:{run_id}:{idempotency_key}:{payload.decision}:{payload.notes}".encode()).hexdigest()
            repository.upsert_idempotency_record(
                session=session,
                scope="resume",
                key_hash=resume_hash,
                run_id=run_id,
                status="resumed",
                current_node=str(state.get("current_node", "")),
                ttl_minutes=settings.idempotency_ttl_minutes,
            )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run(
    run_id: str,
    auth: AuthContext = Depends(require_auth_or_context),
    session: Session = Depends(get_session),
) -> RunStatusResponse:
    enforce_role(auth, {"admin", "operator", "reviewer"})
    record = repository.get_run_record(session, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    latest_checkpoint = repository.get_latest_checkpoint(session, run_id)

    token_totals = repository.get_token_totals(session, run_id)
    quota_subject = str(record.state_json.get("quota_subject", auth.subject))
    quota = repository.get_existing_daily_quota(session, quota_subject)
    
    return RunStatusResponse(
        run_id=record.run_id,
        status=record.status,
        current_node=record.current_node,
        objective=record.objective,
        high_impact=record.high_impact,
        iteration_count=record.iteration_count,
        initial_token_budget=record.initial_token_budget,
        token_budget_remaining=record.token_budget_remaining,
        estimated_cost_usd=float(record.estimated_cost_usd or 0.0),
        latest_checkpoint_seq=latest_checkpoint.seq if latest_checkpoint else None,
        latest_checkpoint_at=_as_utc(latest_checkpoint.created_at) if latest_checkpoint else None,
        started_at=_as_utc(record.started_at),
        updated_at=_as_utc(record.updated_at),
        plan=str(record.state_json.get("plan", "") or ""),
        research_notes=list(record.state_json.get("research_notes", []) or []),
        analysis=str(record.state_json.get("analysis", "") or ""),
        draft=str(record.state_json.get("draft", "") or ""),
        critique=str(record.state_json.get("critique", "") or ""),
        final_output=str(record.state_json.get("final_output", "") or ""),
        human_decision=str(record.state_json.get("human_decision", "") or ""),
        human_reviewer=str(record.state_json.get("human_reviewer", "") or ""),
        human_notes=str(record.state_json.get("human_notes", "") or ""),
        metering_mode=str(token_totals.get("metering_mode", "estimated")),
        prompt_tokens_total=int(token_totals.get("prompt_tokens_total", 0)),
        completion_tokens_total=int(token_totals.get("completion_tokens_total", 0)),
        total_tokens_used=int(token_totals.get("total_tokens_used", 0)),
        quota_subject=quota_subject,
        quota_daily_limit=settings.quota_daily_tokens,
        quota_daily_used=int(quota.tokens_used) if quota is not None else 0,
        output=record.output,
    )


@router.get("/runs/{run_id}/timeline", response_model=RunTimelineResponse)
def get_timeline(
    run_id: str,
    auth: AuthContext = Depends(require_auth_or_context),
    session: Session = Depends(get_session),
) -> RunTimelineResponse:
    enforce_role(auth, {"admin", "operator", "reviewer"})
    record = repository.get_run_record(session, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    events = repository.get_timeline(session, run_id)
    timeline = [
        TimelineEvent(
            run_id=event.run_id,
            seq=event.seq,
            ts=_as_utc(event.created_at),
            event_type=event.event_type,
            node=event.node,
            message=event.message,
            data=event.data,
        )
        for event in events
    ]
    latest_checkpoint = repository.get_latest_checkpoint(session, run_id)

    token_totals = repository.get_token_totals(session, run_id)
    quota_subject = str(record.state_json.get("quota_subject", auth.subject))
    quota = repository.get_existing_daily_quota(session, quota_subject)

    return RunTimelineResponse(
        run_id=run_id,
        status=record.status,
        current_node=record.current_node,
        initial_token_budget=record.initial_token_budget,
        token_budget_remaining=record.token_budget_remaining,
        estimated_cost_usd=float(record.estimated_cost_usd or 0.0),
        latest_checkpoint_seq=latest_checkpoint.seq if latest_checkpoint else None,
        latest_checkpoint_at=_as_utc(latest_checkpoint.created_at) if latest_checkpoint else None,
        metering_mode=str(token_totals.get("metering_mode", "estimated")),
        prompt_tokens_total=int(token_totals.get("prompt_tokens_total", 0)),
        completion_tokens_total=int(token_totals.get("completion_tokens_total", 0)),
        total_tokens_used=int(token_totals.get("total_tokens_used", 0)),
        quota_subject=quota_subject,
        quota_daily_limit=settings.quota_daily_tokens,
        quota_daily_used=int(quota.tokens_used) if quota is not None else 0,
        events=timeline,
    )


@router.post("/runs/{run_id}/stop")
def stop_run(
    run_id: str,
    payload: StopRunRequest,
    auth: AuthContext = Depends(require_auth_or_context),
    session: Session = Depends(get_session),
) -> dict[str, object]:
    enforce_role(auth, {"admin", "operator", "reviewer"})

    state = repository.get_run_state(session, run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")

    status = str(state.get("status", ""))
    if status in TERMINAL_STATUSES:
        return {
            "run_id": run_id,
            "status": status,
            "stopped": False,
            "message": "Run is already terminal",
        }

    reason = str(payload.reason or "Stopped by operator")
    if status == "awaiting_human":
        now = datetime.now(timezone.utc)
        trace = list(state.get("trace", []))
        stop_event = {
            "seq": len(trace) + 1,
            "ts": now.isoformat(),
            "event_type": "run_stopped",
            "node": "orchestrator",
            "message": "Run stopped by operator",
            "data": {"reason": reason, "actor": auth.subject},
        }
        stopped_state = {
            **state,
            "status": "stopped",
            "current_node": "finalize",
            "updated_at": now,
            "stop_requested": True,
            "stop_requested_by": auth.subject,
            "stop_reason": reason,
            "stop_requested_at": now.isoformat(),
            "trace": [*trace, stop_event],
        }
        repository.persist_step(
            session,
            state=stopped_state,
            seq=int(stop_event["seq"]),
            event_type="run_stopped",
            node="orchestrator",
            message="Run stopped by operator",
            data={"reason": reason, "actor": auth.subject},
        )
        return {
            "run_id": run_id,
            "status": "stopped",
            "stopped": True,
            "message": "Run stopped",
        }

    repository.request_stop(session, run_id, actor=auth.subject, reason=reason)
    return {
        "run_id": run_id,
        "status": status,
        "stopped": True,
        "message": "Stop requested",
    }


@router.post("/runs/{run_id}/resume-budget/stream")
def resume_budget_stream(
    run_id: str,
    payload: BudgetResumeRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    auth: AuthContext = Depends(require_auth_or_context),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    enforce_role(auth, {"admin", "operator", "reviewer"})
    throttle(request)

    state = repository.get_run_state(session, run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if str(state.get("status", "")) != "budget_exhausted":
        raise HTTPException(status_code=409, detail="Run is not budget exhausted")
    if settings.sse_resume_v2 and idempotency_key:
        resume_hash = sha256(f"resume-budget:{auth.subject}:{run_id}:{idempotency_key}:{payload.additional_budget}".encode()).hexdigest()
        if repository.get_idempotency_record(session, "resume_budget", resume_hash):
            return StreamingResponse(iter([sse_pack({"event": "noop", "data": {"run_id": run_id, "message": "Duplicate budget resume ignored"}})]), media_type="text/event-stream")

    def event_gen():
        if settings.sse_resume_v2 and last_event_id:
            try:
                seq_floor = int(last_event_id)
                missed = [evt for evt in repository.get_timeline(session, run_id) if int(evt.seq) > seq_floor]
                for evt in missed:
                    yield sse_pack({"event": "timeline", "data": {
                        "run_id": evt.run_id, "seq": evt.seq, "status": state.get("status", "running"), "current_node": state.get("current_node", ""),
                        "message": evt.message, "event_type": evt.event_type, "node": evt.node, "data": evt.data,
                    }})
            except ValueError:
                pass
        try:
            stream_iter = orchestrator.resume_budget_stream(session, run_id, payload, actor_subject=auth.subject)
        except TypeError:
            stream_iter = orchestrator.resume_budget_stream(session, run_id, payload)
        for event in stream_iter:
            yield sse_pack(event)
        if settings.sse_resume_v2 and idempotency_key:
            resume_hash = sha256(f"resume-budget:{auth.subject}:{run_id}:{idempotency_key}:{payload.additional_budget}".encode()).hexdigest()
            repository.upsert_idempotency_record(
                session=session,
                scope="resume_budget",
                key_hash=resume_hash,
                run_id=run_id,
                status="resumed_budget",
                current_node=str(state.get("current_node", "")),
                ttl_minutes=settings.idempotency_ttl_minutes,
            )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
