"""졸작 로보어드바이저 API — 뉴스요약 / RAG리서치 / 아침브리핑 / 챗봇.

대시보드는 이 엔드포인트를 HTTP로만 호출한다 (모델 직접 로드 금지 = MSA).
"""
from __future__ import annotations
from datetime import date

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.collector.news_fetch import fetch_recent_news
from app.services.news_summary_chain import get_news_summary_chain
from app.services.news_rag_chain import get_news_rag_chain
from app.services.market_recap_chain import get_market_recap_chain
from app.services.chatbot import StockChatbot

router = APIRouter()

# 세션별 챗봇 (대화 메모리를 세션 단위로 분리 — 전역 공유 방지)
_chatbots: dict[str, StockChatbot] = {}


def _session_bot(session_id: str) -> StockChatbot:
    if session_id not in _chatbots:
        _chatbots[session_id] = StockChatbot(get_settings())
    return _chatbots[session_id]


# ── 헬스체크 ──
@router.get("/health")
def health():
    """서버 상태 + 사용 가능한 기능."""
    return {
        "status": "ok",
        "features": ["news_summary", "research_rag", "market_recap", "chat"],
    }


# ── 뉴스 요약 (라이브) ──
class SummaryRequest(BaseModel):
    ticker: str = Field(..., examples=["NVDA"])
    limit: int = Field(8, ge=1, le=20)


@router.post("/news/summary")
def news_summary(body: SummaryRequest):
    """티커의 최신 뉴스를 요약 (출처 포함)."""
    docs = fetch_recent_news(body.ticker, limit=body.limit)
    return get_news_summary_chain().summarize(body.ticker, docs, today=str(date.today()))


# ── RAG 리서치 (과거 아카이브) ──
class ResearchRequest(BaseModel):
    question: str = Field(..., examples=["엔비디아가 작년에 왜 올랐어?"])
    top_k: int = Field(5, ge=1, le=15)


@router.post("/research")
def research(body: ResearchRequest):
    """과거 뉴스 아카이브를 검색해 질문에 답 (출처 인용)."""
    return get_news_rag_chain().ask(body.question, top_k=body.top_k)


# ── 아침 증시 브리핑 ──
@router.get("/market/recap")
def market_recap():
    """전날 미국증시 요약 (지수·특징주·매크로·지정학)."""
    return get_market_recap_chain().generate()


# ── 챗봇 ──
class ChatRequest(BaseModel):
    message: str = Field(..., examples=["엔비디아 요즘 어때?"])
    session_id: str = Field("default", description="대화 세션 구분 (같은 값이면 대화 이어짐)")


@router.post("/chat")
def chat(body: ChatRequest):
    """도구 호출 에이전트 챗봇. session_id로 대화 메모리를 분리한다."""
    return {"reply": _session_bot(body.session_id).chat(body.message)}
