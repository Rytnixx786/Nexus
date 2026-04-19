"""Tests for semantic caching layer."""

from __future__ import annotations

import pytest

from app.core.cache import (
    clear_cache,
    get_cache_stats,
    get_cached_response,
    get_cached_run_result,
    save_response_to_cache,
    save_run_result_to_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache_before_each_test():
    """Clear cache before each test."""
    clear_cache()
    yield
    clear_cache()


def test_cache_miss_returns_none():
    """Test that cache miss returns None."""
    result = get_cached_response("test query", "test context")
    assert result is None


def test_cache_hit_returns_saved_response():
    """Test that cache hit returns the saved response."""
    query = "What is artificial intelligence?"
    context = "AI is a field of computer science"
    expected_response = "AI is a transformative field that enables machines to think"

    # Save to cache
    save_response_to_cache(query, context, expected_response)

    # Retrieve from cache
    cached = get_cached_response(query, context)
    assert cached == expected_response


def test_different_query_bypasses_cache():
    """Test that different query doesn't return cached result."""
    query1 = "What is machine learning?"
    context = "ML is a subset of AI"
    response = "Machine learning enables computers to learn from data"

    # Save with query1
    save_response_to_cache(query1, context, response)

    # Try to retrieve with different query
    query2 = "What is deep learning?"
    cached = get_cached_response(query2, context)
    assert cached is None


def test_different_context_bypasses_cache():
    """Test that different context doesn't return cached result."""
    query = "How does neural networks work?"
    context1 = "Neural networks are inspired by biological neurons"
    response = "Neural networks are computational models..."

    # Save with context1
    save_response_to_cache(query, context1, response)

    # Try to retrieve with different context
    context2 = "A different context about neural networks"
    cached = get_cached_response(query, context2)
    assert cached is None


def test_cache_with_empty_context():
    """Test caching with empty context."""
    query = "What is data science?"
    context = ""
    response = "Data science combines statistics and programming"

    # Save and retrieve
    save_response_to_cache(query, context, response)
    cached = get_cached_response(query, context)
    assert cached == response


def test_cache_with_empty_query():
    """Test caching with empty query."""
    query = ""
    context = "Some context"
    response = "A response"

    # Save and retrieve
    save_response_to_cache(query, context, response)
    cached = get_cached_response(query, context)
    assert cached == response


def test_cache_with_multiline_response():
    """Test caching with multiline response."""
    query = "What are the steps to build an AI?"
    context = "Building AI requires multiple steps"
    response = """Step 1: Define the problem
Step 2: Collect data
Step 3: Train the model
Step 4: Evaluate and iterate
Step 5: Deploy"""

    # Save and retrieve
    save_response_to_cache(query, context, response)
    cached = get_cached_response(query, context)
    assert cached == response


def test_cache_stats_empty():
    """Test cache stats when cache is empty."""
    stats = get_cache_stats()
    assert stats["total_entries"] == 0
    assert stats["approx_size_kb"] == 0


def test_cache_stats_with_entries():
    """Test cache stats with entries."""
    query1 = "Query 1"
    context = "Context"
    response1 = "Response 1" * 100  # Make it reasonably sized

    query2 = "Query 2"
    response2 = "Response 2" * 50

    save_response_to_cache(query1, context, response1)
    save_response_to_cache(query2, context, response2)

    stats = get_cache_stats()
    assert stats["total_entries"] == 2
    assert stats["approx_size_kb"] > 0


def test_cache_overwrites_existing_key():
    """Test that saving with same key/context overwrites previous value."""
    query = "What is NLP?"
    context = "Natural language context"
    response1 = "First response about NLP"
    response2 = "Updated response about NLP"

    # Save first response
    save_response_to_cache(query, context, response1)
    assert get_cached_response(query, context) == response1

    # Save second response with same key
    save_response_to_cache(query, context, response2)
    assert get_cached_response(query, context) == response2


def test_cache_key_generation_is_deterministic():
    """Test that cache key generation is deterministic."""
    query = "Same query"
    context = "Same context"

    # Save once
    response = "A response"
    save_response_to_cache(query, context, response)
    result1 = get_cached_response(query, context)

    # Retrieve multiple times - should be consistent
    result2 = get_cached_response(query, context)
    result3 = get_cached_response(query, context)

    assert result1 == result2 == result3 == response


def test_cache_with_special_characters():
    """Test caching with special characters in query and context."""
    query = "What is AI? #@$%^&*()_+-=[]{}|;:,.<>?"
    context = "Context with special chars: 你好 🎉 café"
    response = "Response with symbols: ™ © ®"

    save_response_to_cache(query, context, response)
    cached = get_cached_response(query, context)
    assert cached == response


def test_clear_cache_removes_all_entries():
    """Test that clear_cache removes all cached entries."""
    # Add some entries
    save_response_to_cache("Query 1", "Context", "Response 1")
    save_response_to_cache("Query 2", "Context", "Response 2")
    assert get_cache_stats()["total_entries"] == 2

    # Clear cache
    clear_cache()
    assert get_cache_stats()["total_entries"] == 0

    # Verify entries are gone
    assert get_cached_response("Query 1", "Context") is None
    assert get_cached_response("Query 2", "Context") is None


def test_run_cache_lookup_delegates_to_semantic_cache(monkeypatch):
    """Run-level cache lookup should delegate to semantic cache class."""

    class _FakeSemanticCache:
        def __init__(self):
            self.called = False

        def get_cached_result(self, session, objective, high_impact=False):
            self.called = True
            return {"status": "completed", "output": "cached", "cached": True}

    fake = _FakeSemanticCache()
    monkeypatch.setattr("app.core.cache._get_semantic_cache", lambda: fake)

    result = get_cached_run_result(session=None, objective="objective", high_impact=True)
    assert result is not None
    assert result["cached"] is True
    assert fake.called is True


def test_run_cache_store_delegates_to_semantic_cache(monkeypatch):
    """Run-level cache stores should delegate with ttl days."""

    calls: list[dict[str, object]] = []

    class _FakeSemanticCache:
        def store_cached_result(self, session, objective, result, ttl_days=7, high_impact=False):
            calls.append(
                {
                    "objective": objective,
                    "result": result,
                    "ttl_days": ttl_days,
                    "high_impact": high_impact,
                }
            )

    monkeypatch.setattr("app.core.cache._get_semantic_cache", lambda: _FakeSemanticCache())

    save_run_result_to_cache(
        session=None,
        objective="objective",
        result={"status": "completed", "output": "cached"},
        ttl_days=9,
        high_impact=True,
    )

    assert len(calls) == 1
    assert calls[0]["ttl_days"] == 9
    assert calls[0]["high_impact"] is True
