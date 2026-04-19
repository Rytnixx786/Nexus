# Semantic Cache

## Purpose

Nexus Researcher now supports run-level semantic caching on `POST /api/runs/stream`.
If a new objective is highly similar to a previous successful run, the API returns
cached results immediately instead of re-running the graph.

## Similarity Threshold

- Threshold: `0.95` cosine similarity
- Reasoning: this is a precision-first threshold to avoid accidental reuse of near-miss objectives.

A lower threshold increases hit rate but also increases the risk of serving stale or
semantically different output.

## Cache Key Strategy

The cache key is generated from:

- normalized objective (`strip().lower()`)
- `high_impact` flag

This prevents reuse between high-impact and non-high-impact flows.

## TTL Strategy

- Default TTL: 7 days
- Entries are stored with `created_at` and `expires_at`
- Expired entries are removed by periodic cleanup job and can be deleted on demand via repository cleanup helpers

## Degradation Behavior

Semantic caching uses sentence-transformers embeddings when available. If any of the
following is unavailable or fails, the system degrades to exact-match hash lookup:

- embedding model loading
- vector generation
- pgvector extension availability

This ensures caching remains functional and safe even when semantic/vector features are down.

## Storage Model

`semantic_cache` stores:

- `objective_hash`
- `objective_embedding`
- `result_summary`
- `full_result_json`
- `created_at`
- `expires_at`

The API marks cache-served SSE terminal events with `cached: true`.
