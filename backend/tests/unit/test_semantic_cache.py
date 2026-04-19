from __future__ import annotations

from app.core.semantic_cache import SemanticCache


def test_exact_hash_hit_without_embedding(monkeypatch):
    cache = SemanticCache()

    class _Row:
        full_result_json = {"status": "completed", "output": "exact hit"}

    monkeypatch.setattr("app.core.semantic_cache.repository.get_exact_semantic_cache_record", lambda session, objective_hash: _Row())

    result = cache.get_cached_result(session=None, objective="objective", high_impact=False)

    assert result is not None
    assert result["cache_match"] == "exact"
    assert result["cached"] is True


def test_semantic_hit_above_threshold(monkeypatch):
    cache = SemanticCache(similarity_threshold=0.95)

    class _Row:
        objective_embedding = [1.0, 0.0]
        full_result_json = {"status": "completed", "output": "semantic hit"}

    monkeypatch.setattr("app.core.semantic_cache.repository.get_exact_semantic_cache_record", lambda session, objective_hash: None)
    monkeypatch.setattr("app.core.semantic_cache.repository.list_active_semantic_cache_records", lambda session: [_Row()])
    monkeypatch.setattr(cache, "_embed_text", lambda text: [1.0, 0.0])

    result = cache.get_cached_result(session=None, objective="objective", high_impact=False)

    assert result is not None
    assert result["cache_match"] == "semantic"
    assert result["cached"] is True


def test_semantic_miss_below_threshold(monkeypatch):
    cache = SemanticCache(similarity_threshold=0.95)

    class _Row:
        objective_embedding = [0.0, 1.0]
        full_result_json = {"status": "completed", "output": "semantic hit"}

    monkeypatch.setattr("app.core.semantic_cache.repository.get_exact_semantic_cache_record", lambda session, objective_hash: None)
    monkeypatch.setattr("app.core.semantic_cache.repository.list_active_semantic_cache_records", lambda session: [_Row()])
    monkeypatch.setattr(cache, "_embed_text", lambda text: [1.0, 0.0])

    result = cache.get_cached_result(session=None, objective="objective", high_impact=False)

    assert result is None


def test_store_cached_result_uses_ttl(monkeypatch):
    cache = SemanticCache()
    calls = []

    monkeypatch.setattr(cache, "_embed_text", lambda text: [0.1, 0.2, 0.3])

    def _capture_upsert(session, objective_hash, objective_embedding, result_summary, full_result_json, ttl_days):
        calls.append(
            {
                "objective_hash": objective_hash,
                "objective_embedding": objective_embedding,
                "result_summary": result_summary,
                "full_result_json": full_result_json,
                "ttl_days": ttl_days,
            }
        )

    monkeypatch.setattr("app.core.semantic_cache.repository.upsert_semantic_cache_record", _capture_upsert)

    cache.store_cached_result(
        session=None,
        objective="objective",
        result={"status": "completed", "output": "cached output"},
        ttl_days=10,
        high_impact=True,
    )

    assert len(calls) == 1
    assert calls[0]["ttl_days"] == 10
    assert calls[0]["result_summary"] == "cached output"


def test_semantic_match_respects_high_impact_flag(monkeypatch):
    cache = SemanticCache(similarity_threshold=0.95)

    class _Row:
        objective_embedding = [1.0, 0.0]
        full_result_json = {"status": "completed", "output": "semantic hit", "cache_high_impact": False}

    monkeypatch.setattr("app.core.semantic_cache.repository.get_exact_semantic_cache_record", lambda session, objective_hash: None)
    monkeypatch.setattr("app.core.semantic_cache.repository.list_active_semantic_cache_records", lambda session: [_Row()])
    monkeypatch.setattr(cache, "_embed_text", lambda text: [1.0, 0.0])

    result = cache.get_cached_result(session=None, objective="objective", high_impact=True)
    assert result is None
