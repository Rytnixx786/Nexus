"""Semantic caching layer for LLM responses.

Simple in-memory cache indexed by hash(query + top_context).
Stores LLM responses to avoid redundant generations.
Cache enforces max size and entry TTL to prevent unbounded memory growth.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Configuration
CACHE_MAX_ENTRIES = 1000  # Max number of cache entries
CACHE_MAX_SIZE_MB = 100  # Max total cache size in MB
CACHE_ENTRY_TTL_SECONDS = 3600  # 1 hour TTL for each entry

# Module-level in-memory cache: {cache_key: (response, timestamp)}
_response_cache: dict[str, tuple[str, float]] = {}
_semantic_cache_instance = None


def _get_semantic_cache():
    global _semantic_cache_instance
    if _semantic_cache_instance is None:
        from app.core.semantic_cache import SemanticCache

        _semantic_cache_instance = SemanticCache()
    return _semantic_cache_instance


def _cleanup_expired_and_oversized() -> None:
    """Remove expired entries and enforce size limits.
    
    Called before each cache operation to maintain bounds:
    - Remove entries older than TTL
    - Remove oldest entries if size/count limits exceeded
    """
    now = time.time()
    
    # Remove expired entries
    expired_keys = [k for k, (_, ts) in _response_cache.items() if now - ts > CACHE_ENTRY_TTL_SECONDS]
    for k in expired_keys:
        del _response_cache[k]
        logger.debug("Removed expired cache entry", extra={"cache_key": k[:16]})
    
    # Remove oldest entries if too many
    if len(_response_cache) > CACHE_MAX_ENTRIES:
        remove_count = len(_response_cache) - CACHE_MAX_ENTRIES
        sorted_entries = sorted(_response_cache.items(), key=lambda x: x[1][1])[:remove_count]
        for k, _ in sorted_entries:
            del _response_cache[k]
        logger.debug("Removed oldest cache entries by count", extra={"removed": remove_count})
    
    # Remove oldest entries if too large
    total_size_bytes = sum(len(v[0]) for v in _response_cache.values())
    if total_size_bytes > CACHE_MAX_SIZE_MB * 1024 * 1024:
        sorted_entries = sorted(_response_cache.items(), key=lambda x: x[1][1])
        removed = 0
        for k, _ in sorted_entries:
            del _response_cache[k]
            removed += 1
            total_size_bytes = sum(len(v[0]) for v in _response_cache.values())
            if total_size_bytes <= CACHE_MAX_SIZE_MB * 1024 * 1024:
                break
        logger.debug("Removed oldest cache entries by size", extra={"removed": removed})


def _make_cache_key(query: str, top_context: str) -> str:
    """Generate cache key from query and top context.
    
    Args:
        query: Search query or objective string
        top_context: Top retrieved context snippet or uploaded context
    
    Returns:
        Hex hash of combined strings
    """
    combined = f"{query.strip()}|{top_context.strip()}"
    return hashlib.sha256(combined.encode()).hexdigest()


def get_cached_response(query: str, top_context: str) -> str | None:
    """Check cache for existing response.
    
    Args:
        query: Search query or objective string
        top_context: Top retrieved context or user-uploaded context
    
    Returns:
        Cached response string if hit, None if miss
    """
    _cleanup_expired_and_oversized()
    
    cache_key = _make_cache_key(query, top_context)
    cached_entry = _response_cache.get(cache_key)
    
    if cached_entry:
        cached_response, timestamp = cached_entry
        age_seconds = time.time() - timestamp
        if age_seconds <= CACHE_ENTRY_TTL_SECONDS:
            logger.debug(
                "Cache hit",
                extra={"cache_key": cache_key[:16], "value_length": len(cached_response), "total_cached": len(_response_cache), "age_seconds": int(age_seconds)}
            )
            return cached_response
        else:
            del _response_cache[cache_key]
    
    logger.debug(
        "Cache miss",
        extra={"cache_key": cache_key[:16], "total_cached": len(_response_cache)}
    )
    return None


def save_response_to_cache(query: str, top_context: str, response: str) -> None:
    """Save LLM response to cache.
    
    Args:
        query: Search query or objective string
        top_context: Top retrieved context or user-uploaded context
        response: LLM-generated response text
    """
    _cleanup_expired_and_oversized()
    
    cache_key = _make_cache_key(query, top_context)
    _response_cache[cache_key] = (response, time.time())
    
    logger.debug(
        "Cached response",
        extra={"cache_key": cache_key[:16], "value_length": len(response), "total_cached": len(_response_cache)}
    )


def clear_cache() -> None:
    """Clear all cached responses. Useful for testing."""
    global _response_cache
    _response_cache.clear()
    logger.debug("Cache cleared")


def get_cache_stats() -> dict[str, Any]:
    """Return cache statistics."""
    return {
        "total_entries": len(_response_cache),
        "approx_size_kb": sum(len(v[0]) for v in _response_cache.values()) // 1024,
        "max_entries": CACHE_MAX_ENTRIES,
        "max_size_mb": CACHE_MAX_SIZE_MB,
        "ttl_seconds": CACHE_ENTRY_TTL_SECONDS,
    }


def get_cached_run_result(session: Session, objective: str, high_impact: bool = False) -> dict[str, Any] | None:
    """Get cached run result for /api/runs/stream semantic lookup."""
    return _get_semantic_cache().get_cached_result(
        session=session,
        objective=objective,
        high_impact=high_impact,
    )


def save_run_result_to_cache(
    session: Session,
    objective: str,
    result: dict[str, Any],
    ttl_days: int = 7,
    high_impact: bool = False,
) -> None:
    """Store completed run output in semantic cache for stream-level reuse."""
    _get_semantic_cache().store_cached_result(
        session=session,
        objective=objective,
        result=result,
        ttl_days=ttl_days,
        high_impact=high_impact,
    )
