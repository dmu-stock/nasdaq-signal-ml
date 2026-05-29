"""LLM 기반 종합 시그널 산출 (CSV + OHLCV 입력 → 구조화 수치 출력).

이 모듈은 **휴리스틱(compute_signal)을 LLM으로 대체** 한다.

흐름:
  1) yfinance 로 최근 60일 OHLCV 받아 CSV 텍스트 화 (LLM 입력)
  2) `data/news_*.csv` 에서 해당 ticker 언급 뉴스 최근 30일분 추출
  3) 두 데이터를 한 프롬프트에 넣고 LLM 에 다음을 요청 (structured output):
       - signal: up | down | neutral
       - probability: 0~1 (상승확률)
       - confidence: 0~1 (신뢰도)
       - support, resistance_1, resistance_2, recommended_buy
       - reasoning: 한국어 근거 1~2문장
  4) 호출 측에서 휴리스틱 결과와 병행 보관, LLM 우선

LLM 이 산술을 정확히 못하는 한계가 있어 호출 측은 LLM 출력이 비정상이면
heuristic 으로 폴백할 수 있도록 `ok` 플래그를 포함시킨다.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.technical_analysis import fetch_ohlcv


# ══════════════════════════════════════
# 구조화 출력 스키마
# ══════════════════════════════════════
class CSVSignalResult(BaseModel):
    signal: str = Field(..., description='"up" | "down" | "neutral" 중 하나')
    probability: float = Field(
        ...,
        description="상승확률 (0.0 ~ 1.0). 가격 추세·이평 정합·뉴스 톤을 종합한 판단",
    )
    confidence: float = Field(
        ...,
        description=(
            "신뢰도 (0.0 ~ 1.0). 가격 지표들과 뉴스 신호가 얼마나 같은 방향을 "
            "가리키는지의 정합 강도. 통계적 신뢰구간이 아니라 정성적 일관성."
        ),
    )
    support: float = Field(..., description="지지선 (USD). 최근 저점 또는 피벗 기반.")
    resistance_1: float = Field(..., description="1차 저항선 (USD). 최근 고점 또는 피벗 R1.")
    resistance_2: float = Field(..., description="2차 저항선 (USD). 그 위 다음 저항대.")
    recommended_buy: float = Field(
        ...,
        description="추천 매수가 (USD). 보수적 진입 — 현재가와 지지선 사이가 일반적.",
    )
    reasoning: str = Field(
        ...,
        description="한국어 1~2문장 근거. 어떤 지표/뉴스를 근거로 했는지 간략히.",
    )


_SYSTEM = (
    "너는 미국 주식 단기 기술적·심리적 분석가다. 사용자가 제공한 (1) 최근 60일 OHLCV CSV, "
    "(2) 해당 종목 관련 뉴스 헤드라인을 모두 검토해서 다음 구조화 수치를 출력한다.\n\n"
    "원칙:\n"
    "1) signal/probability/confidence/support/resistance_1/resistance_2/recommended_buy 모두 산출.\n"
    "2) 가격 레벨은 OHLCV 의 실제 값 범위 안에서 산출 — 현재가 ±20% 범위를 크게 벗어나지 말 것.\n"
    "3) probability 와 confidence 는 0~1 실수. probability 가 0.5 면 중립.\n"
    "4) 뉴스 톤이 부정적이고 가격이 이평선 아래면 down/낮은 prob. 둘 다 긍정이면 up/높은 prob.\n"
    "5) reasoning 은 한국어 1~2문장으로 어떤 근거를 봤는지 짧게."
)

_USER = (
    "종목: {ticker}\n"
    "기준일: {as_of}\n"
    "현재가: ${current_price}\n\n"
    "[최근 60일 OHLCV — CSV 형식]\n"
    "{ohlcv_csv}\n\n"
    "[관련 뉴스 헤드라인 — 최근순, 최대 {news_count}건]\n"
    "{news_block}\n\n"
    "위 두 정보를 종합해서 구조화 수치를 산출하라."
)


# ══════════════════════════════════════
# 뉴스 CSV 로더 (싱글톤 캐시)
# ══════════════════════════════════════
class NewsCSVStore:
    """뉴스 CSV 를 메모리에 한 번만 로드해서 ticker 별 필터를 빠르게."""

    _TICKER_ALIASES = {
        "AAPL":  ["AAPL", "Apple", "애플"],
        "MSFT":  ["MSFT", "Microsoft", "마이크로소프트"],
        "GOOGL": ["GOOGL", "GOOG", "Google", "Alphabet", "구글"],
        "AMZN":  ["AMZN", "Amazon", "아마존"],
        "NVDA":  ["NVDA", "Nvidia", "NVIDIA", "엔비디아"],
        "META":  ["META", "Meta", "Facebook", "메타", "페이스북"],
        "TSLA":  ["TSLA", "Tesla", "테슬라"],
        "AVGO":  ["AVGO", "Broadcom", "브로드컴"],
        "COST":  ["COST", "Costco", "코스트코"],
        "NFLX":  ["NFLX", "Netflix", "넷플릭스"],
    }

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self.df is not None:
            return self.df
        if not self.csv_path.exists():
            self.df = pd.DataFrame(columns=["date", "title", "source", "content"])
            return self.df
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        # 날짜 정규화
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
        self.df = df
        return df

    def recent_for_ticker(self, ticker: str, max_rows: int = 15) -> pd.DataFrame:
        df = self._load()
        if df.empty:
            return df
        aliases = self._TICKER_ALIASES.get(ticker.upper(), [ticker])
        pat = "|".join(map(lambda s: s.replace("$", r"\$"), aliases))
        title_mask = df["title"].astype(str).str.contains(pat, case=False, na=False, regex=True)
        content_mask = df["content"].astype(str).str.contains(pat, case=False, na=False, regex=True)
        hit = df[title_mask | content_mask].head(max_rows)
        return hit


# ══════════════════════════════════════
# 메인 체인
# ══════════════════════════════════════
class CSVAnalysisChain:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.news = NewsCSVStore(settings.news_csv_path)
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        ).with_structured_output(CSVSignalResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user", _USER),
        ])
        self.chain = self.prompt | self.llm

    # ── OHLCV → CSV 텍스트 ──
    def _ohlcv_to_csv_text(self, ohlcv: pd.DataFrame, n: int = 60) -> tuple[str, dict]:
        if ohlcv is None or ohlcv.empty:
            return "(데이터 없음)", {}
        df = ohlcv.tail(n).copy()
        # 컬럼 정리
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].round(2)
        df.index.name = "date"
        # 인덱스가 datetime 이면 날짜만
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        csv_text = df.to_csv()
        meta = {
            "as_of": df.index[-1],
            "current_price": float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0,
            "rows": len(df),
        }
        return csv_text, meta

    # ── 뉴스 → 텍스트 블록 ──
    def _news_to_block(self, news_df: pd.DataFrame) -> tuple[str, int]:
        if news_df.empty:
            return "(관련 뉴스 없음)", 0
        lines = []
        for _, r in news_df.iterrows():
            date = str(r.get("date", ""))[:10]
            title = str(r.get("title", ""))[:140]
            source = str(r.get("source", ""))[:30]
            lines.append(f"- [{date} · {source}] {title}")
        return "\n".join(lines), len(news_df)

    # ── 메인 호출 ──
    def analyze(self, ticker: str) -> dict:
        # 1) OHLCV
        ohlcv = fetch_ohlcv(ticker, period="3mo")
        csv_text, meta = self._ohlcv_to_csv_text(ohlcv, n=60)
        if not meta:
            return {"ok": False, "error": f"{ticker} OHLCV 없음"}

        # 2) 뉴스
        news_df = self.news.recent_for_ticker(ticker, max_rows=15)
        news_block, news_count = self._news_to_block(news_df)

        # 3) LLM 호출
        try:
            result: CSVSignalResult = self.chain.invoke({
                "ticker": ticker,
                "as_of": meta["as_of"],
                "current_price": meta["current_price"],
                "ohlcv_csv": csv_text,
                "news_block": news_block,
                "news_count": news_count,
            })
        except Exception as e:
            return {
                "ok": False,
                "error": f"LLM 호출 실패: {type(e).__name__}: {str(e)[:200]}",
            }

        # 4) 결과 후처리 — 비정상 값은 ok=False 처리해서 호출측이 폴백할 수 있게
        out = result.model_dump()
        cp = meta["current_price"]
        valid = (
            0.0 <= out["probability"] <= 1.0
            and 0.0 <= out["confidence"] <= 1.0
            and out["support"] < cp < out["resistance_1"] < out["resistance_2"]
            and 0.5 * cp < out["support"] < 1.5 * cp  # 50% 범위 안
        )
        out.update({
            "ok": True,
            "valid": valid,  # 호출측이 valid=False 면 휴리스틱으로 폴백
            "current_price": cp,
            "as_of": meta["as_of"],
            "news_count_used": news_count,
            "source": "csv_llm",
        })
        return out


@lru_cache
def get_csv_analysis_chain() -> CSVAnalysisChain:
    return CSVAnalysisChain(get_settings())
