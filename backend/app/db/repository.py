from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from fastapi.encoders import jsonable_encoder
from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.cost import estimate_cost
from app.db.tables import Base, IdempotencyRecord, QuotaWindow, RunCheckpoint, RunEvent, RunRecord, SemanticCache, TokenUsageLedger

logger = get_logger(__name__)


class _QuotaStub:
    def __init__(self, subject: str) -> None:
        self.subject = subject
        self.tokens_used = 0


def _quota_window_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or datetime.now(timezone.utc)
    day_start = datetime(year=current.year, month=current.month, day=current.day, tzinfo=timezone.utc)
    return day_start, day_start + timedelta(days=1)


def init_db(session: Session) -> None:
    bind = session.get_bind()
    if bind is None:
        raise RuntimeError("Database bind is not configured")
    Base.metadata.create_all(bind)
    _ensure_runs_schema(bind)


def _ensure_runs_schema(bind) -> None:
    if bind.dialect.name not in {"sqlite", "postgresql"}:
        return

    with bind.begin() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(runs)").all()} if bind.dialect.name == "sqlite" else set()
        if bind.dialect.name == "postgresql":
            result = connection.exec_driver_sql(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'runs'
                """
            )
            columns = {row[0] for row in result.fetchall()}

        if "initial_token_budget" not in columns:
            try:
                connection.exec_driver_sql(
                    "ALTER TABLE runs ADD COLUMN initial_token_budget INTEGER NOT NULL DEFAULT 8000"
                )
            except OperationalError:
                # If another process migrated the schema first, the app can continue normally.
                pass

        if "estimated_cost_usd" not in columns:
            try:
                connection.exec_driver_sql(
                    "ALTER TABLE runs ADD COLUMN estimated_cost_usd FLOAT NOT NULL DEFAULT 0.0"
                )
            except OperationalError:
                # If another process migrated the schema first, the app can continue normally.
                pass


def create_run(session: Session, state: dict[str, Any]) -> None:
    safe_state = jsonable_encoder(state)
    initial_budget = int(safe_state.get("initial_token_budget", safe_state.get("token_budget_remaining", 8000)))
    record = RunRecord(
        run_id=safe_state["run_id"],
        objective=safe_state["objective"],
        high_impact=bool(safe_state["high_impact"]),
        status=safe_state["status"],
        current_node=safe_state["current_node"],
        iteration_count=int(safe_state["iteration_count"]),
        initial_token_budget=initial_budget,
        token_budget_remaining=int(safe_state["token_budget_remaining"]),
        estimated_cost_usd=float(safe_state.get("estimated_cost_usd", 0.0) or 0.0),
        output=safe_state.get("final_output", ""),
        state_json=safe_state,
        started_at=state["started_at"],
        updated_at=state["updated_at"],
    )
    session.add(record)
    session.commit()


def update_run(session: Session, state: dict[str, Any]) -> None:
    safe_state = jsonable_encoder(state)
    record = session.get(RunRecord, state["run_id"])
    if record is None:
        raise KeyError("Run not found")

    record.status = safe_state["status"]
    record.current_node = safe_state["current_node"]
    record.iteration_count = int(safe_state["iteration_count"])
    record.token_budget_remaining = int(safe_state["token_budget_remaining"])
    record.estimated_cost_usd = float(safe_state.get("estimated_cost_usd", record.estimated_cost_usd) or 0.0)
    record.output = safe_state.get("final_output", "")
    record.state_json = safe_state
    record.updated_at = state["updated_at"]
    session.commit()


def get_run_state(session: Session, run_id: str) -> dict[str, Any] | None:
    record = session.get(RunRecord, run_id)
    if record is None:
        return None
    return dict(record.state_json)


def get_run_record(session: Session, run_id: str) -> RunRecord | None:
    return session.get(RunRecord, run_id)


def request_stop(session: Session, run_id: str, actor: str, reason: str) -> dict[str, Any]:
    record = session.get(RunRecord, run_id)
    if record is None:
        raise KeyError("Run not found")

    state = dict(record.state_json or {})
    state["stop_requested"] = True
    state["stop_requested_by"] = str(actor or "unknown")
    state["stop_reason"] = str(reason or "Stopped by operator")
    state["stop_requested_at"] = datetime.now(timezone.utc).isoformat()

    record.state_json = jsonable_encoder(state)
    record.updated_at = datetime.now(timezone.utc)
    session.commit()
    return state


def _run_filters(
    search: str | None,
    status: str | None,
    started_from: datetime | None,
    started_to: datetime | None,
    min_cost_usd: float | None,
    max_cost_usd: float | None,
) -> list:
    filters = []
    if search:
        like = f"%{search.strip()}%"
        filters.append((RunRecord.objective.ilike(like)) | (RunRecord.run_id.ilike(like)))
    if status:
        filters.append(RunRecord.status == status)
    if started_from:
        filters.append(RunRecord.started_at >= started_from)
    if started_to:
        filters.append(RunRecord.started_at <= started_to)
    if min_cost_usd is not None:
        filters.append(RunRecord.estimated_cost_usd >= float(min_cost_usd))
    if max_cost_usd is not None:
        filters.append(RunRecord.estimated_cost_usd <= float(max_cost_usd))
    return filters


def list_runs(
    session: Session,
    limit: int,
    offset: int,
    search: str | None = None,
    status: str | None = None,
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    min_cost_usd: float | None = None,
    max_cost_usd: float | None = None,
) -> list[RunRecord]:
    stmt = select(RunRecord)
    filters = _run_filters(search, status, started_from, started_to, min_cost_usd, max_cost_usd)
    if filters:
        stmt = stmt.where(*filters)
    stmt = stmt.order_by(RunRecord.started_at.desc()).offset(offset).limit(limit)
    return list(session.scalars(stmt).all())


def count_runs(
    session: Session,
    search: str | None = None,
    status: str | None = None,
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    min_cost_usd: float | None = None,
    max_cost_usd: float | None = None,
) -> int:
    stmt = select(func.count()).select_from(RunRecord)
    filters = _run_filters(search, status, started_from, started_to, min_cost_usd, max_cost_usd)
    if filters:
        stmt = stmt.where(*filters)
    return int(session.scalar(stmt) or 0)


def get_system_metrics(session: Session) -> dict[str, Any]:
    total_runs = int(session.scalar(select(func.count()).select_from(RunRecord)) or 0)

    status_rows = session.execute(
        select(RunRecord.status, func.count()).group_by(RunRecord.status)
    ).all()
    runs_by_status = {str(status): int(count) for status, count in status_rows}

    avg_token_usage = session.scalar(select(func.avg(RunRecord.initial_token_budget - RunRecord.token_budget_remaining)))
    avg_token_usage_per_run = float(avg_token_usage) if avg_token_usage is not None else 0.0

    avg_steps = session.scalar(select(func.avg(RunRecord.iteration_count)))
    avg_steps_per_run = float(avg_steps) if avg_steps is not None else 0.0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    runs_last_24h = int(
        session.scalar(select(func.count()).select_from(RunRecord).where(RunRecord.started_at >= cutoff)) or 0
    )

    total_cost_usd = float(session.scalar(select(func.coalesce(func.sum(RunRecord.estimated_cost_usd), 0.0))) or 0.0)

    avg_cost_per_run_usd = float(
        session.scalar(
            select(func.coalesce(func.avg(RunRecord.estimated_cost_usd), 0.0)).where(RunRecord.status == "completed")
        )
        or 0.0
    )

    provider_rows = session.execute(
        select(
            TokenUsageLedger.provider,
            TokenUsageLedger.model,
            func.coalesce(func.sum(TokenUsageLedger.prompt_tokens), 0),
            func.coalesce(func.sum(TokenUsageLedger.completion_tokens), 0),
        ).group_by(TokenUsageLedger.provider, TokenUsageLedger.model)
    ).all()

    cost_by_provider: dict[str, float] = {}
    for provider, model, prompt_tokens, completion_tokens in provider_rows:
        provider_key = str(provider or "unknown")
        event_cost = estimate_cost(provider_key, str(model or ""), int(prompt_tokens or 0), int(completion_tokens or 0))
        cost_by_provider[provider_key] = float(cost_by_provider.get(provider_key, 0.0) + event_cost)

    return {
        "total_runs": total_runs,
        "runs_by_status": runs_by_status,
        "avg_token_usage_per_run": avg_token_usage_per_run,
        "avg_steps_per_run": avg_steps_per_run,
        "runs_last_24h": runs_last_24h,
        "total_cost_usd": total_cost_usd,
        "avg_cost_per_run_usd": avg_cost_per_run_usd,
        "cost_by_provider": cost_by_provider,
    }


def get_timeline(session: Session, run_id: str) -> list[RunEvent]:
    stmt = select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq.asc())
    return list(session.scalars(stmt).all())


def get_latest_checkpoint(session: Session, run_id: str) -> RunCheckpoint | None:
    stmt = (
        select(RunCheckpoint)
        .where(RunCheckpoint.run_id == run_id)
        .order_by(RunCheckpoint.seq.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def append_token_usage(
    session: Session,
    run_id: str,
    seq: int,
    node: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    metering_mode: str,
    retry_count: int = 0,
    billable: bool = True,
) -> None:
    prompt_value = max(0, int(prompt_tokens))
    completion_value = max(0, int(completion_tokens))

    row = TokenUsageLedger(
        run_id=run_id,
        seq=seq,
        node=node,
        provider=provider,
        model=model,
        prompt_tokens=prompt_value,
        completion_tokens=completion_value,
        total_tokens=max(0, int(total_tokens)),
        metering_mode=metering_mode,
        retry_count=max(0, int(retry_count)),
        billable=bool(billable),
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)

    # Increment run cost using this event's contribution to avoid a full ledger rescan.
    run_record = session.get(RunRecord, run_id)
    if run_record:
        event_cost = estimate_cost(provider, model, prompt_value, completion_value)
        current_cost = float(run_record.estimated_cost_usd or 0.0)
        run_record.estimated_cost_usd = current_cost + float(event_cost)
        run_record.updated_at = datetime.now(timezone.utc)

    session.commit()


def get_token_totals(session: Session, run_id: str) -> dict[str, Any]:
    row = session.execute(
        select(
            func.coalesce(func.sum(TokenUsageLedger.prompt_tokens), 0),
            func.coalesce(func.sum(TokenUsageLedger.completion_tokens), 0),
            func.coalesce(func.sum(TokenUsageLedger.total_tokens), 0),
            func.max(TokenUsageLedger.metering_mode),
        ).where(TokenUsageLedger.run_id == run_id)
    ).first()
    if not row:
        return {"prompt_tokens_total": 0, "completion_tokens_total": 0, "total_tokens_used": 0, "metering_mode": "estimated"}
    return {
        "prompt_tokens_total": int(row[0] or 0),
        "completion_tokens_total": int(row[1] or 0),
        "total_tokens_used": int(row[2] or 0),
        "metering_mode": str(row[3] or "estimated"),
    }


def calculate_run_cost(session: Session, run_id: str) -> float:
    """Calculate total estimated cost in USD for all token usage in a run."""
    ledger_rows = session.execute(
        select(
            TokenUsageLedger.provider,
            TokenUsageLedger.model,
            func.sum(TokenUsageLedger.prompt_tokens).label("total_prompt"),
            func.sum(TokenUsageLedger.completion_tokens).label("total_completion"),
        )
        .where(TokenUsageLedger.run_id == run_id)
        .group_by(TokenUsageLedger.provider, TokenUsageLedger.model)
    ).all()
    
    total_cost = 0.0
    for provider, model, prompt_tokens, completion_tokens in ledger_rows:
        cost = estimate_cost(provider, model, prompt_tokens or 0, completion_tokens or 0)
        total_cost += cost
    
    return total_cost


def get_or_create_daily_quota(session: Session, subject: str) -> QuotaWindow | _QuotaStub:
    """Get or create daily quota window for subject.
    
    Returns a stub with zero balance if table initialization fails,
    allowing quota checks to degrade gracefully.
    """
    now = datetime.now(timezone.utc)
    day_start, day_end = _quota_window_bounds(now)
    stmt = select(QuotaWindow).where(
        QuotaWindow.subject == subject,
        QuotaWindow.window_start == day_start,
        QuotaWindow.window_end == day_end,
    )
    
    # Try to query existing quota
    try:
        existing = session.scalar(stmt)
        if existing:
            return existing
    except OperationalError as exc:
        logger.warning("Failed to query quota window; attempting table creation", extra={"error": str(exc), "subject": subject})
        # Try to create the table
        try:
            bind = session.get_bind()
            if bind is not None:
                Base.metadata.create_all(bind, tables=[QuotaWindow.__table__])
            # Retry query after table creation
            try:
                existing = session.scalar(stmt)
                if existing:
                    return existing
            except OperationalError as retry_exc:
                logger.error("Failed to query quota even after table creation", extra={"error": str(retry_exc), "subject": subject})
                return _QuotaStub(subject)
        except Exception as create_exc:
            logger.error("Failed to create quota table", extra={"error": str(create_exc), "subject": subject})
            return _QuotaStub(subject)
    
    # Create new quota window
    try:
        created = QuotaWindow(
            subject=subject,
            window_start=day_start,
            window_end=day_end,
            tokens_used=0,
            updated_at=now,
        )
        session.add(created)
        session.commit()
        session.refresh(created)
        return created
    except OperationalError as exc:
        logger.error("Failed to create quota window", extra={"error": str(exc), "subject": subject})
        return _QuotaStub(subject)


def get_existing_daily_quota(session: Session, subject: str) -> QuotaWindow | None:
    day_start, day_end = _quota_window_bounds()
    stmt = select(QuotaWindow).where(
        QuotaWindow.subject == subject,
        QuotaWindow.window_start == day_start,
        QuotaWindow.window_end == day_end,
    )
    return session.scalar(stmt)


def get_existing_daily_quotas(session: Session, subjects: Iterable[str]) -> dict[str, QuotaWindow]:
    normalized_subjects = {subject for subject in subjects if subject}
    if not normalized_subjects:
        return {}

    day_start, day_end = _quota_window_bounds()
    stmt = select(QuotaWindow).where(
        QuotaWindow.subject.in_(normalized_subjects),
        QuotaWindow.window_start == day_start,
        QuotaWindow.window_end == day_end,
    )
    return {row.subject: row for row in session.scalars(stmt).all()}


def consume_quota_tokens(session: Session, subject: str, tokens: int) -> QuotaWindow:
    row = get_or_create_daily_quota(session, subject)
    if isinstance(row, _QuotaStub):
        row.tokens_used += max(0, int(tokens))
        return row  # type: ignore[return-value]
    row.tokens_used = int(row.tokens_used) + max(0, int(tokens))
    row.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(row)
    return row


def get_idempotency_record(session: Session, scope: str, key_hash: str) -> IdempotencyRecord | None:
    now = datetime.now(timezone.utc)
    stmt = select(IdempotencyRecord).where(
        IdempotencyRecord.scope == scope,
        IdempotencyRecord.idempotency_key_hash == key_hash,
        IdempotencyRecord.expires_at >= now,
    )
    return session.scalar(stmt)


def upsert_idempotency_record(
    session: Session,
    scope: str,
    key_hash: str,
    run_id: str,
    status: str,
    current_node: str,
    response_json: dict[str, Any] | None = None,
    ttl_minutes: int = 120,
) -> None:
    now = datetime.now(timezone.utc)
    existing = get_idempotency_record(session, scope, key_hash)
    payload = response_json or {}
    if existing:
        existing.run_id = run_id
        existing.status = status
        existing.current_node = current_node
        existing.response_json = payload
        existing.expires_at = now + timedelta(minutes=max(1, ttl_minutes))
        session.commit()
        return
    row = IdempotencyRecord(
        scope=scope,
        idempotency_key_hash=key_hash,
        run_id=run_id,
        status=status,
        current_node=current_node,
        response_json=payload,
        expires_at=now + timedelta(minutes=max(1, ttl_minutes)),
        created_at=now,
    )
    session.add(row)
    session.commit()


def persist_step(
    session: Session,
    state: dict[str, Any],
    seq: int,
    event_type: str,
    node: str,
    message: str,
    data: dict[str, Any],
) -> None:
    safe_state = jsonable_encoder(state)
    safe_data = jsonable_encoder(data)

    record = session.get(RunRecord, state["run_id"])
    if record is None:
        raise KeyError("Run not found")

    record.status = safe_state["status"]
    record.current_node = safe_state["current_node"]
    record.iteration_count = int(safe_state["iteration_count"])
    record.token_budget_remaining = int(safe_state["token_budget_remaining"])
    record.output = safe_state.get("final_output", "")
    record.state_json = safe_state
    record.updated_at = state["updated_at"]

    event = RunEvent(
        run_id=state["run_id"],
        seq=seq,
        event_type=event_type,
        node=node,
        message=message,
        data=safe_data,
        created_at=datetime.now(timezone.utc),
    )
    checkpoint = RunCheckpoint(
        run_id=state["run_id"],
        seq=seq,
        node=node,
        state_json=safe_state,
        created_at=datetime.now(timezone.utc),
    )

    session.add(event)
    session.add(checkpoint)
    session.commit()


def get_exact_semantic_cache_record(
    session: Session,
    objective_hash: str,
) -> SemanticCache | None:
    now = datetime.now(timezone.utc)
    stmt = (
        select(SemanticCache)
        .where(SemanticCache.objective_hash == objective_hash)
        .where(SemanticCache.expires_at >= now)
        .order_by(SemanticCache.created_at.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def list_active_semantic_cache_records(session: Session) -> list[SemanticCache]:
    now = datetime.now(timezone.utc)
    stmt = (
        select(SemanticCache)
        .where(SemanticCache.expires_at >= now)
        .order_by(SemanticCache.created_at.desc())
    )
    return list(session.scalars(stmt).all())


def upsert_semantic_cache_record(
    session: Session,
    objective_hash: str,
    objective_embedding: dict[str, Any] | list[float] | None,
    result_summary: str,
    full_result_json: dict[str, Any],
    ttl_days: int = 7,
) -> None:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=max(1, int(ttl_days)))
    existing = get_exact_semantic_cache_record(session, objective_hash)
    if existing is not None:
        existing.objective_embedding = objective_embedding
        existing.result_summary = result_summary
        existing.full_result_json = full_result_json
        existing.expires_at = expires_at
        session.commit()
        return

    row = SemanticCache(
        objective_hash=objective_hash,
        objective_embedding=objective_embedding,
        result_summary=result_summary,
        full_result_json=full_result_json,
        created_at=now,
        expires_at=expires_at,
    )
    session.add(row)
    session.commit()


def delete_expired_semantic_cache(session: Session) -> int:
    now = datetime.now(timezone.utc)
    result = session.execute(
        delete(SemanticCache).where(SemanticCache.expires_at < now)
    )
    session.commit()
    return int(result.rowcount or 0)
