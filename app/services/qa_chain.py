"""LangChain 기반 주식 Q&A 파이프라인.

디스코드 봇의 `a.<질문>` 명령이 호출하는 백엔드.

흐름:
  1) 분류 체인 (LLM 가드레일): 질문이 "주식/금융/시장/기업/거시경제" 관련인지 판단.
     - 관련 없음 → 거부 메시지 반환, LLM 호출 종료.
  2) RAG 검색: ChromaDB 에서 관련 뉴스 top-k 회수 (티커 힌트가 있으면 필터).
  3) 답변 체인: 시스템 프롬프트 + 검색된 뉴스 컨텍스트로 답변 생성.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.rag_service import RAGService, get_rag_service


# ══════════════════════════════════════
# 1) 분류 체인 — 주식 관련 질문인지 판별
# ══════════════════════════════════════
class StockTopicCheck(BaseModel):
    is_stock_related: bool = Field(
        ...,
        description=(
            "주식 / 종목 / 미국 증시 / 기업 실적 / 거시경제(금리·인플레·환율 등) / "
            "투자 전략 / 시장 동향 관련 질문이면 True. 일반 잡담·코딩·요리·연예 "
            "등 무관한 질문이면 False."
        ),
    )
    ticker_hint: Optional[str] = Field(
        None,
        description=(
            "질문에서 특정 종목이 언급됐다면 그 종목의 미국 티커 (예: AAPL, TSLA). "
            "없으면 null."
        ),
    )
    reason: str = Field(..., description="한 줄 판단 근거 (한국어).")


_CLASSIFIER_SYSTEM = (
    "너는 디스코드 주식 분석 봇의 질문 분류기다. 사용자의 질문이 "
    "**주식·금융시장·기업분석·투자·거시경제** 와 관련 있는지 판단한다. "
    "정치/사회/스포츠/연예/일반 잡담은 시장 영향이 명확하지 않다면 무관으로 본다. "
    "코딩/요리/일상 질문은 무조건 무관(False)."
)

_CLASSIFIER_USER = "질문: {question}\n\n위 질문을 구조화 형식으로 분류하라."


# ══════════════════════════════════════
# 2) 답변 체인 — 주식 관련 질문 RAG 응답
# ══════════════════════════════════════
_ANSWER_SYSTEM = (
    "너는 미국 주식 시장을 전문으로 하는 AI 애널리스트다. "
    "지원 종목: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO, COST, NFLX. "
    "사용자의 질문에 한국어로, 간결하고 정확하게 답한다.\n\n"
    "원칙:\n"
    "1) 아래 [참고 뉴스] 컨텍스트가 있으면 우선 활용하고, 사실은 뉴스에 근거해 답한다.\n"
    "2) **반드시 컨텍스트의 항목 번호로 인용한다**. 예: `엔비디아는 데이터센터 매출이 급증했다[1][3].`\n"
    "   - 컨텍스트에 없는 일반 지식 문장은 인용하지 말 것.\n"
    "   - 답변 끝에 별도의 '출처' 섹션을 만들지는 말 것 (디스코드 임베드가 따로 붙임).\n"
    "3) 컨텍스트에 없는 정보는 일반 지식으로 보충하되, 모르는 것은 모른다고 말한다.\n"
    "4) 투자 추천(매수/매도 단정)은 하지 않는다. 대신 호재/악재 요인을 정리한다.\n"
    "5) 가격/실적 등 수치는 정확하지 않을 수 있으므로 '최신 시세는 /c.{{TICKER}}.시세 로 확인' 안내."
)

_ANSWER_USER = (
    "[참고 뉴스]\n{context}\n\n"
    "[질문]\n{question}\n\n"
    "위 정보를 바탕으로 한국어로 답하라. 컨텍스트 항목은 [번호] 로 인용. "
    "디스코드 메시지로 보낼 거니까 1500자 이내."
)


# ══════════════════════════════════════
# 거부 응답 (주식 무관 질문)
# ══════════════════════════════════════
REJECT_MESSAGE = (
    "⚠️ 이건 주식 이야기가 아니에요.\n"
    "저는 미국 주식 / 시장 / 기업 분석 전용 AI 봇이에요. "
    "주식·종목·거시경제 관련 질문을 `a.` 뒤에 적어주세요.\n"
    "예) `a. 엔비디아 최근 호재가 뭐야?`, `a. 미국 금리 인하가 빅테크에 미치는 영향은?`"
)


# ══════════════════════════════════════
# 한글 / 영문 회사명 → 티커 백업 매핑
# (분류기가 ticker_hint 를 못 채울 때 폴백)
# ══════════════════════════════════════
_TICKER_ALIASES = {
    "AAPL":  ["apple", "애플"],
    "MSFT":  ["microsoft", "마이크로소프트"],
    "GOOGL": ["google", "alphabet", "구글", "알파벳"],
    "AMZN":  ["amazon", "아마존"],
    "NVDA":  ["nvidia", "엔비디아"],
    "META":  ["meta", "메타", "facebook", "페이스북"],
    "TSLA":  ["tesla", "테슬라"],
    "AVGO":  ["broadcom", "브로드컴"],
    "COST":  ["costco", "코스트코"],
    "NFLX":  ["netflix", "넷플릭스"],
}


def _guess_ticker(question: str) -> Optional[str]:
    q = (question or "").lower()
    for tk, names in _TICKER_ALIASES.items():
        if tk.lower() in q:
            return tk
        for name in names:
            if name.lower() in q:
                return tk
    return None


class StockQAChain:
    """LangChain 으로 묶인 (분류 → RAG → 답변) 파이프라인."""

    def __init__(self, settings: Settings, rag: RAGService):
        self.settings = settings
        self.rag = rag

        # 분류용 LLM (구조화 출력)
        self.classifier_llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        ).with_structured_output(StockTopicCheck)
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", _CLASSIFIER_SYSTEM),
            ("user", _CLASSIFIER_USER),
        ])
        self.classifier_chain = self.classifier_prompt | self.classifier_llm

        # 답변용 LLM (자유 텍스트)
        self.answer_llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        )
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", _ANSWER_SYSTEM),
            ("user", _ANSWER_USER),
        ])
        self.answer_chain = self.answer_prompt | self.answer_llm | StrOutputParser()

    # ── 분류 ──
    def classify(self, question: str) -> StockTopicCheck:
        return self.classifier_chain.invoke({"question": question})

    # ── RAG 컨텍스트 ──
    def _retrieve_context(
        self,
        question: str,
        ticker: Optional[str],
        k: int = 5,
    ) -> tuple[str, list[dict]]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        try:
            hits = self.rag.search(
                query=question,
                ticker=ticker,
                k=k,
                start=start,
                end=end,
            )
        except Exception:
            hits = []

        # Chroma 가 비어있고 티커가 있으면 yfinance 로 라이브 보강
        if not hits and ticker:
            hits = self._fetch_yfinance_news(ticker, k=k)

        if not hits:
            return "(관련 뉴스 컨텍스트 없음 — 일반 지식으로 답하라.)", []

        lines = []
        sources = []
        for i, h in enumerate(hits, 1):
            title = h.get("title") or "(제목 없음)"
            src = h.get("source") or ""
            pub = h.get("published_at") or ""
            snippet = (h.get("snippet") or "").strip().replace("\n", " ")
            lines.append(f"[{i}] ({src} | {pub}) {title}\n    {snippet}")
            sources.append({
                "title": title,
                "url": h.get("url"),
                "source": src,
                "published_at": pub,
                "ticker": h.get("ticker") or ticker,
            })
        return "\n".join(lines), sources

    # ── yfinance 라이브 폴백 (Chroma 비었을 때) ──
    def _fetch_yfinance_news(self, ticker: str, k: int = 5) -> list[dict]:
        try:
            import yfinance as yf
            items = yf.Ticker(ticker).news or []
        except Exception:
            return []

        def _dget(d, key):
            v = d.get(key) if isinstance(d, dict) else None
            return v.get("url") if isinstance(v, dict) else None

        out: list[dict] = []
        for item in items[: k * 2]:
            content = item.get("content") or item
            title = content.get("title") or item.get("title") or ""
            if not title:
                continue
            summary = content.get("summary") or content.get("description") or ""
            url = (
                _dget(content, "canonicalUrl")
                or _dget(content, "clickThroughUrl")
                or item.get("link")
                or ""
            )
            provider = content.get("provider") or {}
            source = provider.get("displayName") if isinstance(provider, dict) else ""
            pub = content.get("pubDate") or item.get("providerPublishTime") or ""
            out.append({
                "title": title,
                "snippet": (summary or title)[:300],
                "url": url,
                "source": source or "Yahoo Finance",
                "published_at": str(pub),
                "ticker": ticker,
            })
            if len(out) >= k:
                break
        return out

    # ── 메인 ──
    def ask(self, question: str, ticker: Optional[str] = None) -> dict:
        question = (question or "").strip()
        if not question:
            return {
                "ok": False,
                "is_stock_related": False,
                "answer": "⚠️ 질문이 비어있어요. `a. 무엇이 궁금한가요?` 형태로 적어주세요.",
                "ticker": None,
                "sources": [],
            }

        # 1) 분류
        try:
            check = self.classify(question)
        except Exception as e:
            return {
                "ok": False,
                "is_stock_related": False,
                "answer": f"⚠️ 분류기 호출에 실패했어요: {e}",
                "ticker": None,
                "sources": [],
            }

        if not check.is_stock_related:
            return {
                "ok": True,
                "is_stock_related": False,
                "answer": REJECT_MESSAGE,
                "ticker": None,
                "sources": [],
                "reason": check.reason,
            }

        # 2) RAG 컨텍스트 회수
        final_ticker = ticker or check.ticker_hint or _guess_ticker(question)
        context, sources = self._retrieve_context(question, final_ticker)

        # 3) 답변 생성
        try:
            answer = self.answer_chain.invoke({
                "question": question,
                "context": context,
            })
        except Exception as e:
            return {
                "ok": False,
                "is_stock_related": True,
                "answer": f"⚠️ 답변 생성 중 오류: {e}",
                "ticker": final_ticker,
                "sources": sources,
            }

        return {
            "ok": True,
            "is_stock_related": True,
            "answer": answer.strip(),
            "ticker": final_ticker,
            "sources": sources,
            "reason": check.reason,
        }


@lru_cache
def get_qa_chain() -> StockQAChain:
    return StockQAChain(get_settings(), get_rag_service())
