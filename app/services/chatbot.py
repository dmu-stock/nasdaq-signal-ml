"""주식 리서치 챗봇 (도구 호출 에이전트, 직접 구현)."""
from __future__ import annotations
import json
from datetime import date
from functools import lru_cache

import yfinance as yf
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.collector.news_fetch import fetch_recent_news
from app.services.news_summary_chain import get_news_summary_chain
from app.services.news_rag_chain import get_news_rag_chain
from app.services.market_recap_chain import get_market_recap_chain
from app.services.signal_service import get_buy_picks, get_ticker_signal


@tool
def summarize_recent_news(ticker: str) -> str:
    """미국 주식 티커의 '최신' 뉴스를 요약. '지금/요즘/오늘' 질문에 사용.
    ticker는 심볼이어야 함 (엔비디아→NVDA, 애플→AAPL)."""
    docs = fetch_recent_news(ticker, limit=8)
    result = get_news_summary_chain().summarize(ticker, docs, today=str(date.today()))
    return json.dumps(result, ensure_ascii=False)


@tool
def research_past_news(question: str) -> str:
    """2024~2025 뉴스 아카이브를 검색해 '과거' 질문에 답.
    '작년에/2024년에/왜 올랐·빠졌' 같은 과거 사건 질문에 사용."""
    return json.dumps(get_news_rag_chain().ask(question), ensure_ascii=False)


@tool
def get_stock_price(ticker: str) -> str:
    """미국 주식의 현재가와 전일 대비 등락률 조회. '가격/얼마/현재가/등락' 질문에 사용.
    ticker는 심볼이어야 함 (엔비디아→NVDA)."""
    try:
        info = yf.Ticker(ticker).fast_info
        last, prev = info.get("lastPrice"), info.get("previousClose")
        change_pct = round((last / prev - 1) * 100, 2) if (last and prev) else None
        return json.dumps({"ticker": ticker, "price": last, "change_pct": change_pct},
                          ensure_ascii=False)
    except Exception as error:
        return f"{ticker} 가격 조회 실패: {error}"


@tool
def get_market_recap() -> str:
    """전날 미국증시 전반 요약 — 지수·특징주(급등락)·유가·금·금리·지정학(전쟁/긴장).
    '증시 어때/시장 상황/전쟁 여파/유가·금/거시' 같은 시장 전반 질문에 사용."""
    return json.dumps(get_market_recap_chain().generate(), ensure_ascii=False)

@tool
def buy_signal() -> str:
    """오늘의 ML 매수 시그널(top 종목). '매수 시그널/오늘 뭐 사?'에 사용.
    (처음 호출은 모델 추론으로 수십초 걸림)"""
    return json.dumps(get_buy_picks(), ensure_ascii=False)

@tool
def ticker_signal(ticker: str) -> str:
    """특정 종목의 ML 매수 점수(LGBM·LSTM 확률). '엔비디아 시그널 어때?'에 사용. ticker=심볼."""
    return json.dumps(get_ticker_signal(ticker), ensure_ascii=False)


_SYSTEM = (
    "너는 미국 주식 리서치 어시스턴트다. 한국어로 정중히 답한다.\n"
    "- 회사명은 티커로 바꿔 도구에 넘긴다 (엔비디아→NVDA).\n"
    "- '가격/얼마/현재가/등락' 질문 → get_stock_price\n"
    "- '지금/요즘' 종목 뉴스 질문 → summarize_recent_news\n"
    "- '증시/시장/전쟁/지정학/유가·금/거시' 질문 → get_market_recap\n"
    "- '작년/과거/왜 올랐·빠졌' 질문 → research_past_news\n"
    "- 여러 도구가 필요하면 순서대로 호출해도 된다.\n"
    "- 도구 결과의 출처를 답에 밝히고, 없는 사실은 지어내지 않는다.\n"
    "- '매수 시그널/오늘 뭐 사/시그널 점수' 질문 → buy_signal 또는 ticker_signal\n"
)


class StockChatbot:
    def __init__(self, settings: Settings):
        self.tools = [summarize_recent_news, research_past_news, get_stock_price, get_market_recap, buy_signal, ticker_signal]
        self.by_name = {tool_obj.name: tool_obj for tool_obj in self.tools}
        self.llm = ChatOpenAI(
            model=settings.chat_model, api_key=settings.openai_api_key, temperature=0,
        ).bind_tools(self.tools)                # ← LLM에게 도구 메뉴를 쥐여줌
        self.history = [SystemMessage(content=_SYSTEM)]   # ← 대화 메모리

    def chat(self, user_input: str, max_steps: int = 5) -> str:
        self.history.append(HumanMessage(content=user_input))
        for _ in range(max_steps):
            ai_message = self.llm.invoke(self.history)      # LLM 판단
            self.history.append(ai_message)
            if not ai_message.tool_calls:                   # 도구 안 부름 = 최종 답
                return ai_message.content
            for tool_call in ai_message.tool_calls:         # 도구 실행 후 결과를 히스토리에
                print(f"  [tool] {tool_call['name']}({tool_call['args']})")   # 생각 로그 (Windows cp949 콘솔 때문에 이모지 X)
                tool_fn = self.by_name.get(tool_call["name"])
                try:
                    tool_result = (tool_fn.invoke(tool_call["args"]) if tool_fn
                                   else f"알 수 없는 도구: {tool_call['name']}")
                except Exception as error:
                    tool_result = f"도구 실행 오류: {type(error).__name__}: {error}"
                # 성공/실패 무관하게 반드시 응답 메시지를 붙인다 (히스토리 깨짐 방지)
                self.history.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        return "(도구 반복 과다로 중단)"


@lru_cache
def get_chatbot() -> StockChatbot:
    return StockChatbot(get_settings())