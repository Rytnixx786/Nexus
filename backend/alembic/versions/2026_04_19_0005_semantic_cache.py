"""add semantic cache table

Revision ID: 0005_semantic_cache
Revises: 0004_runs_estimated_cost
Create Date: 2026-04-19
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

try:
    from pgvector.sqlalchemy import Vector

    _EMBEDDING_TYPE = Vector(384)
except Exception:  # pragma: no cover
    _EMBEDDING_TYPE = sa.JSON()


revision = "0005_semantic_cache"
down_revision = "0004_runs_estimated_cost"
branch_labels = None
depends_on = None


def _maybe_enable_pgvector(bind) -> None:
    if bind.dialect.name != "postgresql":
        return
    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception:
        # Degrade gracefully: runtime will use exact-hash fallback if vector infra is unavailable.
        pass


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    _maybe_enable_pgvector(bind)

    if "semantic_cache" not in inspector.get_table_names():
        op.create_table(
            "semantic_cache",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("objective_hash", sa.String(length=128), nullable=False),
            sa.Column("objective_embedding", _EMBEDDING_TYPE, nullable=True),
            sa.Column("result_summary", sa.Text(), nullable=False, server_default=""),
            sa.Column("full_result_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
        )
        inspector = sa.inspect(bind)

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("semantic_cache")}
    if "ix_semantic_cache_objective_hash" not in existing_indexes:
        op.create_index("ix_semantic_cache_objective_hash", "semantic_cache", ["objective_hash"], unique=False)
    if "ix_semantic_cache_expires_at" not in existing_indexes:
        op.create_index("ix_semantic_cache_expires_at", "semantic_cache", ["expires_at"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "semantic_cache" in inspector.get_table_names():
        existing_indexes = {idx["name"] for idx in inspector.get_indexes("semantic_cache")}
        if "ix_semantic_cache_objective_hash" in existing_indexes:
            op.drop_index("ix_semantic_cache_objective_hash", table_name="semantic_cache")
        if "ix_semantic_cache_expires_at" in existing_indexes:
            op.drop_index("ix_semantic_cache_expires_at", table_name="semantic_cache")
        op.drop_table("semantic_cache")
