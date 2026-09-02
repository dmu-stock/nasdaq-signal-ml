"""LangChain 뉴스 요약 체인.

입력: ticker + NewsDoc 목록
출력: 구조화 한국어 요약 {overall_tone, key_points[], notable_events, as_of}
     각 key_point에 출처 포함 → 미션38 'citation' 요건 충족.
"""
from __future__ import annotations
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.collector.news_fetch import NewsDoc


# ── 출력 스키마 (LLM이 이 모양으로 답하도록 강제) ──
class KeyPoint(BaseModel):
    point: str = Field(..., description="핵심 요지 1문장 (한국어)")
    source: str = Field(..., description="근거 출처 (언론사명 또는 URL)")


class NewsSummary(BaseModel):
    overall_tone: str = Field(..., description="전반 톤: 긍정/중립/부정 + 한 줄 근거")
    key_points: list[KeyPoint] = Field(..., description="핵심 3~5개, 각각 출처 표기")
    notable_events: str = Field(..., description="실적·규제·급등락 등 특이 이벤트 (없으면 '특이사항 없음')")
    as_of: str = Field(..., description="요약 기준일 YYYY-MM-DD")


_SYSTEM = (
    "너는 미국 주식 뉴스 분석가다. 주어진 최신 뉴스만 근거로 한국어 요약을 작성한다.\n"
    "원칙:\n"
    "1) 주어진 뉴스에 있는 내용만 사용. 없는 사실 창작 금지.\n"
    "2) 매수/매도 단정 금지. 사실 요약 위주.\n"
    "3) 각 핵심 요지엔 반드시 출처를 붙인다.\n"
    "4) 정중체(~습니다).\n"
)
_USER = (
    "종목: {ticker}\n오늘: {today}\n\n"
    "최신 뉴스 목록:\n{news_block}\n\n"
    "위 뉴스를 종합해 overall_tone / key_points / notable_events / as_of 를 채워라."
)


def _format_news(docs: list[NewsDoc]) -> str:
    """NewsDoc 목록 → 프롬프트에 넣을 텍스트 블록."""
    if not docs:
        return "(뉴스 없음)"
    return "\n".join(
        f"[{i}] ({d.date}, {d.source}) {d.title}\n    {d.summary}\n    출처: {d.url}"
        for i, d in enumerate(docs, 1)
    )


class NewsSummaryChain:
    def __init__(self, settings: Settings):
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        ).with_structured_output(NewsSummary)      # ← 출력이 무조건 NewsSummary 형태로
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user", _USER),
        ])
        self.chain = self.prompt | self.llm         # ← LCEL: 프롬프트 → LLM 파이프

    def summarize(self, ticker: str, docs: list[NewsDoc], today: str = "") -> dict:
        if not docs:
            return {"overall_tone": "중립 (뉴스 없음)", "key_points": [],
                    "notable_events": "수집된 뉴스가 없습니다.", "as_of": today}
        try:
            result: NewsSummary = self.chain.invoke({
                "ticker": ticker,
                "today": today,
                "news_block": _format_news(docs),
            })
            return result.model_dump()
        except Exception as e:
            return {"overall_tone": f"요약 실패 ({type(e).__name__})", "key_points": [],
                    "notable_events": "-", "as_of": today}


@lru_cache
def get_news_summary_chain() -> NewsSummaryChain:
    return NewsSummaryChain(get_settings())