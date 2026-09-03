"""매일 아침 전날 미국증시 요약 체인."""
from __future__ import annotations
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.collector.market_data import (
    IndexMove, Mover, fetch_index_moves, fetch_movers, fetch_macro_moves,
)
from app.collector.news_fetch import fetch_recent_news, NewsDoc


class MarketRecap(BaseModel):
    headline: str = Field(..., description="오늘의 진짜 동인을 담은 구체적 한 줄. '상승세 지속' 같은 공허한 표현 금지")
    index_summary: str = Field(..., description="3대 지수 + VIX 등락")
    market_theme: str = Field(..., description="급등·급락 종목 패턴에서 유추한 핵심 테마 (예: 사이버보안 다수 하락→소프트웨어 매도세)")
    notable_movers: str = Field(..., description="주요 급등·급락주와 등락률 (제공 데이터만)")
    macro_geopolitics: str = Field(..., description="유가·금·금리·달러 움직임과 지정학(전쟁·긴장)·거시 이슈. 없으면 '특이 거시 이슈 없음'")
    key_points: list[str] = Field(..., description="주요 이슈 3~4개 (출처)")
    watch_today: str = Field(..., description="오늘 관전포인트. 근거 없으면 '특이 일정 미확인'")
    as_of: str


_SYSTEM = (
    "너는 미국 증시 아침 브리핑 작성자다. 지수·매크로·특징주·뉴스만 근거로 한국어로 쓴다.\n"
    "1) 숫자는 입력값만 사용. 창작 금지.\n"
    "2) headline은 '오늘의 진짜 동인'을 담아 구체적으로. '상승세 지속' 같은 공허한 말 금지.\n"
    "3) market_theme은 급등·급락 '패턴'에서 유추 (예: PANW·CRWD·DDOG 하락 → 사이버보안/SW 매도).\n"
    "4) key_points엔 출처를 붙인다.\n"
    "5) watch_today는 뉴스 근거 있을 때만. 없으면 '예정 일정 미확인'.\n"
    "6) macro_geopolitics: 유가·금·달러·금리 움직임과 뉴스에서 지정학(전쟁/긴장)·금리·유가 이슈를 짚는다. "
    "금·유가 동반 급등 + 안전자산 선호는 지정학 리스크 신호로 해석한다.\n"
    "7) 매수/매도 단정 금지. 정중체.\n"
)
_USER = (
    "전날: {as_of}\n\n"
    "[지수]\n{indexes}\n\n"
    "[매크로]\n{macro}\n\n"
    "[특징주]\n{movers}\n\n"
    "[시장·지정학 뉴스]\n{news}\n\n"
    "각 필드를 채워라."
)


def _format_moves(moves: list[IndexMove]) -> str:
    return "\n".join(f"- {move.name}: {move.close:,} ({move.change_pct:+.2f}%)" for move in moves) or "(없음)"


def _format_news(docs: list[NewsDoc]) -> str:
    return "\n".join(f"- ({doc.date}, {doc.source}) {doc.title}" for doc in docs) or "(없음)"


def _format_movers(gainers: list[Mover], losers: list[Mover]) -> str:
    up = ", ".join(f"{move.ticker} {move.change_pct:+.1f}%" for move in gainers)
    down = ", ".join(f"{move.ticker} {move.change_pct:+.1f}%" for move in losers)
    return f"상승: {up}\n하락: {down}"


class MarketRecapChain:
    def __init__(self, settings: Settings):
        self.llm = ChatOpenAI(
            model=settings.chat_model, api_key=settings.openai_api_key, temperature=0.3,
        ).with_structured_output(MarketRecap)
        self.prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("user", _USER)])
        self.chain = self.prompt | self.llm

    def generate(self) -> dict:
        index_moves = fetch_index_moves()
        macro_moves = fetch_macro_moves()
        gainers, losers = fetch_movers()

        news: list[NewsDoc] = []
        for etf in ("QQQ", "SPY"):                       # 시장 뉴스
            news += fetch_recent_news(etf, limit=4)
        for macro_ticker in ("CL=F", "GLD"):             # 유가·금 뉴스 = 지정학 소스
            news += fetch_recent_news(macro_ticker, limit=4)

        as_of = index_moves[0].date if index_moves else ""
        result = self.chain.invoke({
            "as_of": as_of,
            "indexes": _format_moves(index_moves),
            "macro": _format_moves(macro_moves),
            "movers": _format_movers(gainers, losers),
            "news": _format_news(news),
        })
        return result.model_dump()


@lru_cache
def get_market_recap_chain() -> MarketRecapChain:
    return MarketRecapChain(get_settings())
