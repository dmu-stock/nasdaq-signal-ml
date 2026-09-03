"""뉴스 RAG 체인: 질문 → ChromaDB 검색 → 검색된 기사 근거로 LLM 답변."""
from __future__ import annotations
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.database.chroma_db import NewsVectorStore

# 한국어 질문 쿼리 영어로 번역
from langchain_core.output_parsers import StrOutputParser

# ── 출력 스키마 ──
class Citation(BaseModel):
    source: str = Field(..., description="언론사명")
    published: str = Field(..., description="게시일 YYYY-MM-DD")

class RAGAnswer(BaseModel):
    answer: str = Field(..., description="질문에 대한 한국어 답 (검색된 기사 근거)")
    citations: list[Citation] = Field(..., description="답의 근거로 쓴 출처")
    enough: str = Field(..., description="근거 충분성: 충분/부분적/부족")


_SYSTEM = (
    "너는 미국 주식 뉴스 리서치 어시스턴트다. 아래 [검색된 기사]에 있는 내용만 "
    "근거로 한국어로 답한다.\n"
    "1) 기사에 없는 내용은 절대 지어내지 않는다.\n"
    "2) 근거가 부족하면 enough='부족'으로 정직하게 답한다.\n"
    "3) 답에 사용한 기사 출처를 citations에 넣는다.\n"
    "4) 정중체(~습니다).\n"
)
_USER = "질문: {question}\n\n[검색된 기사]\n{context}\n\n위 기사만 근거로 답하라."

_REWRITE = (
    "Convert the user's stock question into a concise English search query "
    "for a financial news database. Output ONLY the query text, no quotes.\n"
    "질문: {question}"
)

def _format_docs(hits: list[tuple[Document, float]]) -> str:
    """검색결과(Document 목록) → 프롬프트에 넣을 텍스트."""
    if not hits:
        return "(검색 결과 없음)"
    out = []
    for i, (doc, score) in enumerate(hits, 1):
        m = doc.metadata or {}
        pub = (m.get("published_at", "") or "")[:10]
        out.append(f"[{i}] ({pub}, {m.get('source', '?')})\n{doc.page_content}")
    return "\n\n".join(out)


class NewsRAGChain:
    def __init__(self, settings: Settings):
        self.vs  = NewsVectorStore(settings)               # ← ① 검색 담당
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
        ).with_structured_output(RAGAnswer)                # ← ② 생성 담당
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM), ("user", _USER),
        ])
        self.chain = self.prompt | self.llm                # LCEL (요약체인과 동일)

        # 추가: 질문 → 영어 검색어 (temperature=0 = 창의성 끄고 정확히)
        self.rewrite_chain = (
            ChatPromptTemplate.from_messages([("user", _REWRITE)])
            | ChatOpenAI(model=settings.chat_model,
                         api_key=settings.openai_api_key, temperature=0)
            | StrOutputParser()                       # ← LLM 답에서 순수 텍스트만 뽑음
        )

    def ask(self, question: str, k: int = 5) -> dict:
        #영어 검색어로 변환
        en_query = self.rewrite_chain.invoke({"question": question}).strip()

        # ① 검색 — LLM 아님, 벡터 유사도
        hits = self.vs.search(en_query, ticker=None, k=k)  # vintage라 ticker=None
        if not hits:
            return {"answer": "관련 기사를 찾지 못했습니다.", "citations": [],
                    "enough": "부족", "search_query": en_query}
        # ② 생성 — 검색된 기사만 근거로
        try:
            result: RAGAnswer = self.chain.invoke({
                "question": question,
                "context": _format_docs(hits),
            })
            out = result.model_dump()
            out["search_query"] = en_query  
            return out
        except Exception as e:
            return {"answer": f"생성 실패 ({type(e).__name__}): {str(e)[:150]}",
                    "citations": [], "enough": "부족", "search_query": en_query}


@lru_cache
def get_news_rag_chain() -> NewsRAGChain:
    return NewsRAGChain(get_settings())