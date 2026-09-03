"""주식 리서치 챗봇 (도구 호출 에이전트, 직접 구현)."""
from __future__ import annotations
import json
from datetime import date
from functools import lru_cache

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.collector.news_fetch import fetch_recent_news
from app.services.news_summary_chain import get_news_summary_chain
from app.services.news_rag_chain import get_news_rag_chain


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


_SYSTEM = (
    "너는 미국 주식 리서치 어시스턴트다. 한국어로 정중히 답한다.\n"
    "- 회사명은 티커로 바꿔 도구에 넘긴다 (엔비디아→NVDA).\n"
    "- '지금/요즘' 질문 → summarize_recent_news\n"
    "- '작년/과거/왜' 질문 → research_past_news\n"
    "- 도구 결과의 출처를 답에 밝히고, 없는 사실은 지어내지 않는다.\n"
)


class StockChatbot:
    def __init__(self, settings: Settings):
        self.tools = [summarize_recent_news, research_past_news]
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
                print(f"  🔧 {tool_call['name']}({tool_call['args']})")   # 생각 로그
                tool_result = self.by_name[tool_call["name"]].invoke(tool_call["args"])
                self.history.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        return "(도구 반복 과다로 중단)"


@lru_cache
def get_chatbot() -> StockChatbot:
    return StockChatbot(get_settings())