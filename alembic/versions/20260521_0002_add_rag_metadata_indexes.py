"""add rag metadata indexes

Revision ID: 20260521_0002
Revises: 20260520_0001
Create Date: 2026-05-21
"""

from alembic import op


revision = "20260521_0002"
down_revision = "20260520_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("create index if not exists ix_rag_documents_metadata on rag_documents using gin (metadata)")
    op.execute("create index if not exists ix_rag_chunks_metadata on rag_chunks using gin (metadata)")
    op.execute("create index if not exists ix_rag_documents_repo on rag_documents ((metadata->>'repo'))")
    op.execute("create index if not exists ix_rag_documents_path on rag_documents ((metadata->>'path'))")


def downgrade() -> None:
    op.execute("drop index if exists ix_rag_documents_path")
    op.execute("drop index if exists ix_rag_documents_repo")
    op.execute("drop index if exists ix_rag_chunks_metadata")
    op.execute("drop index if exists ix_rag_documents_metadata")
