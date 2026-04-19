import json
import logging
from typing import Any
import time
import redis
from app.core.settings import settings

logger = logging.getLogger(__name__)
_last_semantic_cleanup_ts = 0.0
_semantic_cleanup_interval_seconds = 30 * 60

def get_redis():
    """Create a new Redis connection for the worker."""
    return redis.from_url(settings.redis_url)


def cleanup_semantic_cache_job() -> int:
    """Delete expired semantic cache rows."""
    from app.db import repository
    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        deleted = repository.delete_expired_semantic_cache(session)
        logger.info("Semantic cache cleanup completed", extra={"deleted_rows": deleted})
        return deleted
    finally:
        session.close()


def _maybe_run_semantic_cache_cleanup() -> None:
    global _last_semantic_cleanup_ts
    now = time.time()
    if now - _last_semantic_cleanup_ts < _semantic_cleanup_interval_seconds:
        return
    _last_semantic_cleanup_ts = now
    try:
        cleanup_semantic_cache_job()
    except Exception:
        logger.exception("Semantic cache cleanup job failed")

def run_orchestrator_job(state: dict[str, Any]) -> None:
    """Executes the graph run loop in a background RQ worker."""
    from app.core.orchestrator import Orchestrator
    from app.db.session import SessionLocal
    
    run_id = state.get("run_id")
    if not run_id:
        logger.error("Job received without a run_id.")
        return
        
    redis_conn = get_redis()
    orchestrator = Orchestrator()
    _maybe_run_semantic_cache_cleanup()
    
    # We must instantiate a local session for the background task
    session = SessionLocal()
    try:
        # Loop through the stream and publish every event back to the original request via PubSub
        for event in orchestrator._execute_stream(session, state):
            redis_conn.publish(f"run_events:{run_id}", json.dumps(event))
            
            # If the event signifies a terminal/pausing state, we can exit cleanly.
            # But the orchestrator loop automatically stops when graph stream is done.
    except Exception as e:
        logger.exception(f"Unhandled error executing run {run_id}")
        error_event = {
            "event": "run_error",
            "data": {
                "run_id": run_id,
                "error": str(e),
                "status": "failed"
            }
        }
        redis_conn.publish(f"run_events:{run_id}", json.dumps(error_event))
    finally:
        session.close()
