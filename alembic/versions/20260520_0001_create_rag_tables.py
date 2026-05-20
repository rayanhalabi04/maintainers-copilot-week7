"""create rag tables

Revision ID: 20260520_0001
Revises:
Create Date: 2026-05-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260520_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("create extension if not exists vector")
    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("source_type", "source_id", name="uq_rag_documents_source"),
    )
    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("document_id", sa.Text(), sa.ForeignKey("rag_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_rag_chunks_document_index"),
    )
    op.execute("alter table rag_chunks alter column embedding type vector(384) using embedding::vector")
    op.create_index("ix_rag_documents_source_type", "rag_documents", ["source_type"])
    op.create_index("ix_rag_chunks_document_id", "rag_chunks", ["document_id"])
    op.execute(
        "create index ix_rag_chunks_embedding on rag_chunks using ivfflat (embedding vector_cosine_ops)"
    )
    op.execute(
        "create index ix_rag_chunks_text_fts on rag_chunks using gin (to_tsvector('english', text))"
    )


def downgrade() -> None:
    op.drop_index("ix_rag_chunks_text_fts", table_name="rag_chunks")
    op.drop_index("ix_rag_chunks_embedding", table_name="rag_chunks")
    op.drop_index("ix_rag_chunks_document_id", table_name="rag_chunks")
    op.drop_index("ix_rag_documents_source_type", table_name="rag_documents")
    op.drop_table("rag_chunks")
    op.drop_table("rag_documents")
