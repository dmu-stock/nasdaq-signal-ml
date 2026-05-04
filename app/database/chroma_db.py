"""ChromaDB 기반 뉴스 벡터스토어 (RAG 의 R)."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.services.schema import NewsItem


COLLECTION = "stock_news"


class NewsVectorStore:
    """ticker / 날짜 메타필터 + 시맨틱 검색을 제공하는 Chroma 래퍼."""

    def __init__(self, settings: Settings):
        self.settings = settings
        embeddings = OpenAIEmbeddings(
            model=settings.embed_model,
            api_key=settings.openai_api_key,
        )
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    # ── 입력 ──
    def add(self, items: Iterable[NewsItem]) -> int:
        items = list(items)
        if not items:
            return 0

        existing = set(self._existing_ids([it.id for it in items]))
        new_items = [it for it in items if it.id not in existing]
        if not new_items:
            return 0

        docs: list[Document] = []
        ids: list[str] = []
        for it in new_items:
            docs.append(Document(page_content=it.text(), metadata=it.to_metadata()))
            ids.append(it.id)

        self.store.add_documents(docs, ids=ids)
        return len(new_items)

    def _existing_ids(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        res = self.store.get(ids=ids)
        return res.get("ids", []) or []

    # ── 조회 ──
    def fetch_by_window(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> list[Document]:
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        where = {
            "$and": [
                {"ticker": ticker},
                {"published_ts": {"$gte": start_ts}},
                {"published_ts": {"$lt": end_ts}},
            ]
        }
        res = self.store.get(where=where)
        out: list[Document] = []
        for content, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            out.append(Document(page_content=content, metadata=meta or {}))
        return out

    def search(
        self,
        query: str,
        ticker: str | None = None,
        k: int = 5,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[tuple[Document, float]]:
        conds: list[dict] = []
        if ticker:
            conds.append({"ticker": ticker})
        if start is not None:
            conds.append({"published_ts": {"$gte": int(start.timestamp())}})
        if end is not None:
            conds.append({"published_ts": {"$lt": int(end.timestamp())}})

        where: dict | None
        if not conds:
            where = None
        elif len(conds) == 1:
            where = conds[0]
        else:
            where = {"$and": conds}

        return self.store.similarity_search_with_score(query, k=k, filter=where)
