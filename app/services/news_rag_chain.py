"""뉴스 RAG 체인: 질문 → ChromaDB 검색 → 검색된 기사 근거로 LLM 답변."""
from __future__ import annotations
from functools import lru_cache
from datetime import date 

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

_REFINE = (
    "이전 시도들이 근거를 못 찾았다. 아래 질문에 '완전히 다른 각도'의 영어 검색어를 만들어라.\n"
    "- 이전 검색어에 단어만 덧붙이지 마라.\n"
    "- 다른 핵심 키워드·동의어·관련 이벤트·더 넓은 상위 개념을 시도하라.\n"
    "검색어만 출력.\n"
    "질문: {question}\n이미 실패한 검색어들: {tried}"       
)

_REWRITE = (
    "Today is {today}. Convert the user's stock question into a concise English "
    "search query for a financial news database. Resolve relative dates using today "
    "(작년=last year, 지난달=last month, 요즘=recent 등) into explicit years. "
    "Output ONLY the query text, no quotes.\n"
    "질문: {question}"
)

def _format_docs(hits: list[tuple[Document, float]]) -> str:
    """검색결과(Document 목록) → 프롬프트에 넣을 텍스트."""
    if not hits:
        return "(검색 결과 없음)"
    lines = []
    for i, (doc, _score) in enumerate(hits, 1):
        meta = doc.metadata or {}
        published = (meta.get("published_at", "") or "")[:10]
        lines.append(f"[{i}] ({published}, {meta.get('source', '?')})\n{doc.page_content}")
    return "\n\n".join(lines)


class NewsRAGChain:
    def __init__(self, settings: Settings):
        self.vs  = NewsVectorStore(settings)               # 검색 담당
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
        ).with_structured_output(RAGAnswer)                # 생성 담당
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM), ("user", _USER),
        ])
        self.chain = self.prompt | self.llm                # LCEL (요약체인과 동일)

        # 추가: 질문 → 영어 검색어 (temperature=0 = 창의성 끄고 정확히)
        self.rewrite_chain = (
            ChatPromptTemplate.from_messages([("user", _REWRITE)])
            | ChatOpenAI(model=settings.chat_model,
                         api_key=settings.openai_api_key, temperature=0)
            | StrOutputParser()                       # LLM 답에서 순수 텍스트만 뽑음
        )

        self.refine_chain = (
            ChatPromptTemplate.from_messages([("user", _REFINE)])
            | ChatOpenAI(model=settings.chat_model,
                         api_key=settings.openai_api_key, temperature=0.5)
            | StrOutputParser()
        )
         
    def _search_and_answer(self, question: str, query: str, top_k: int) -> dict:
        hits = self.vs.search(query, ticker=None, k=top_k)
        if not hits:
            return {"answer": "관련 기사를 찾지 못했습니다.", "citations": [], "enough": "부족"}
        try:
            result: RAGAnswer = self.chain.invoke(
                {"question": question, "context": _format_docs(hits)})
            return result.model_dump()
        except Exception as error:
            return {"answer": f"생성 실패 ({type(error).__name__})", "citations": [], "enough": "부족"}
        
    def ask(self, question: str, top_k: int = 5, max_tries: int = 3) -> dict:
        query = self.rewrite_chain.invoke({"question": question,"today": str(date.today())}).strip()  # 한→영
        trace, result = [], None
        for attempt in range(1, max_tries + 1):
            result = self._search_and_answer(question, query, top_k)       # 검색+생성
            trace.append({"attempt": attempt, "query": query, "enough": result.get("enough")})
            if result.get("enough") == "충분":
                break 
            tried = ", ".join(t["query"] for t in trace)                                                  
            query = self.refine_chain.invoke(                            
                {"question": question, "tried": tried}).strip()
        result["search_query"] = trace[-1]["query"]
        result["attempts"] = trace          # ← self-correction 과정 (덤: reasoning trace)
        return result


@lru_cache
def get_news_rag_chain() -> NewsRAGChain:
    return NewsRAGChain(get_settings())