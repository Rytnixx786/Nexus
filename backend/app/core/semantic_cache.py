from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db import repository

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.95


@dataclass
class _EmbeddingState:
    model: Any | None
    available: bool


class SemanticCache:
    """DB-backed semantic cache with hash fallback.

    Uses sentence-transformers embeddings when available. If embedding support is
    unavailable, lookup degrades to exact hash matching.
    """

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self._embedding_state = _EmbeddingState(model=None, available=False)

    def _normalize_objective(self, objective: str, high_impact: bool) -> str:
        normalized = (objective or "").strip().lower()
        return f"high_impact={int(bool(high_impact))}|{normalized}"

    def _hash_objective(self, normalized_objective: str) -> str:
        return hashlib.sha256(normalized_objective.encode("utf-8")).hexdigest()

    def _load_embedding_model(self) -> Any | None:
        if self._embedding_state.available:
            return self._embedding_state.model
        if self._embedding_state.model is None and not self._embedding_state.available:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_state.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self._embedding_state.available = True
                logger.info("Semantic cache embedding model loaded")
            except Exception as exc:
                self._embedding_state.model = None
                self._embedding_state.available = False
                logger.warning("Semantic embedding model unavailable; using hash fallback", extra={"error": str(exc)})
        return self._embedding_state.model

    def _embed_text(self, text: str) -> list[float] | None:
        model = self._load_embedding_model()
        if model is None:
            return None
        try:
            vector = model.encode(text, normalize_embeddings=True)
            return [float(value) for value in vector]
        except Exception as exc:
            logger.warning("Semantic embedding generation failed", extra={"error": str(exc)})
            return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        numerator = sum(x * y for x, y in zip(a, b))
        denom_a = math.sqrt(sum(x * x for x in a))
        denom_b = math.sqrt(sum(y * y for y in b))
        if denom_a == 0.0 or denom_b == 0.0:
            return -1.0
        return numerator / (denom_a * denom_b)

    def _extract_embedding(self, payload: Any) -> list[float] | None:
        if isinstance(payload, list):
            try:
                return [float(v) for v in payload]
            except (TypeError, ValueError):
                return None
        return None

    def get_cached_result(self, session: Session, objective: str, high_impact: bool = False) -> dict[str, Any] | None:
        normalized = self._normalize_objective(objective, high_impact)
        objective_hash = self._hash_objective(normalized)

        # Fast exact match path first.
        exact = repository.get_exact_semantic_cache_record(session, objective_hash)
        if exact is not None:
            payload = dict(exact.full_result_json or {})
            payload.pop("cache_high_impact", None)
            payload["cached"] = True
            payload.setdefault("cache_similarity", 1.0)
            payload.setdefault("cache_match", "exact")
            return payload

        query_embedding = self._embed_text(normalized)
        if query_embedding is None:
            return None

        best_payload: dict[str, Any] | None = None
        best_similarity = -1.0
        for row in repository.list_active_semantic_cache_records(session):
            if bool((row.full_result_json or {}).get("cache_high_impact", False)) != bool(high_impact):
                continue
            row_embedding = self._extract_embedding(row.objective_embedding)
            if row_embedding is None:
                continue
            similarity = self._cosine_similarity(query_embedding, row_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_payload = dict(row.full_result_json or {})

        if best_payload is not None and best_similarity >= self.similarity_threshold:
            best_payload.pop("cache_high_impact", None)
            best_payload["cached"] = True
            best_payload["cache_similarity"] = float(best_similarity)
            best_payload["cache_match"] = "semantic"
            return best_payload
        return None

    def store_cached_result(
        self,
        session: Session,
        objective: str,
        result: dict[str, Any],
        ttl_days: int = 7,
        high_impact: bool = False,
    ) -> None:
        normalized = self._normalize_objective(objective, high_impact)
        objective_hash = self._hash_objective(normalized)
        embedding = self._embed_text(normalized)
        summary = str(result.get("output", "") or "")[:1000]
        stored_payload = dict(result)
        stored_payload["cache_high_impact"] = bool(high_impact)

        repository.upsert_semantic_cache_record(
            session=session,
            objective_hash=objective_hash,
            objective_embedding=embedding,
            result_summary=summary,
            full_result_json=stored_payload,
            ttl_days=ttl_days,
        )
