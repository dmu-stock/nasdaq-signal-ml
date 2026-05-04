"""LangChain 기반 뉴스 감성분석.

입력: NewsItem
출력: score ∈ [-1, +1], rationale (한 문장)
캐시: sentiment_cache.json (article id 기준)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from app.core.config import Settings
from app.services.schema import NewsItem


CACHE_PATH = Path(__file__).resolve().parents[2] / "sentiment_cache.json"


class SentimentResult(BaseModel):
    score: float = Field(
        ...,
        description=(
            "주식 시장 영향 관점의 감성 점수. "
            "-1 = 매우 부정 (악재), 0 = 중립, +1 = 매우 긍정 (호재). "
            "소수점 두 자리까지."
        ),
    )
    rationale: str = Field(..., description="한 문장으로 판단 근거.")


_SYSTEM = (
    "너는 주식 시장 분석가다. 주어진 뉴스 한 건이 해당 종목/시장의 단기 주가에 "
    "미칠 영향을 -1.0(강한 악재) ~ +1.0(강한 호재) 사이의 실수로 평가한다. "
    "광고/홍보성·중복기사·관련 없는 기사는 0에 가깝게 둔다. "
    "정치/사회 관련은 시장 영향 관점에서만 본다."
)

_USER = (
    "종목/대상: {ticker}\n"
    "발행시각(UTC): {published}\n"
    "출처: {source}\n"
    "제목: {title}\n"
    "본문: {content}\n\n"
    "위 뉴스의 감성 점수를 구조화 형식으로 답하라."
)


class SentimentScorer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        ).with_structured_output(SentimentResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user", _USER),
        ])
        self.chain = self.prompt | self.llm
        self._cache = self._load_cache()

    # ── 캐시 ──
    def _load_cache(self) -> dict[str, dict]:
        if CACHE_PATH.exists():
            try:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        CACHE_PATH.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_cached(self, item_id: str) -> dict | None:
        return self._cache.get(item_id)

    # ── 점수 ──
    def score_one(self, item: NewsItem) -> dict:
        cached = self._cache.get(item.id)
        if cached is not None:
            return cached
        result: SentimentResult = self.chain.invoke({
            "ticker": item.ticker,
            "published": item.published_at.isoformat(),
            "source": item.source,
            "title": item.title,
            "content": item.content or "(본문 없음)",
        })
        record = {
            "id": item.id,
            "ticker": item.ticker,
            "published_at": item.published_at.isoformat(),
            "score": float(result.score),
            "rationale": result.rationale,
        }
        self._cache[item.id] = record
        return record

    def score_batch(
        self,
        items: Iterable[NewsItem],
        save_every: int = 20,
        show_progress: bool = True,
    ) -> list[dict]:
        items = list(items)
        out: list[dict] = []
        new_count = 0
        iterator = tqdm(items, desc="감성분석", ncols=80) if show_progress else items
        for it in iterator:
            try:
                pre = self._cache.get(it.id) is not None
                rec = self.score_one(it)
                out.append(rec)
                if not pre:
                    new_count += 1
                if new_count and new_count % save_every == 0:
                    self._save_cache()
            except Exception as e:
                out.append({
                    "id": it.id,
                    "ticker": it.ticker,
                    "published_at": it.published_at.isoformat(),
                    "score": 0.0,
                    "rationale": f"[ERROR] {e}",
                })
        self._save_cache()
        return out
