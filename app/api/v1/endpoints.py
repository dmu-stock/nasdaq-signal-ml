from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.services.rag_service import RAGService, get_rag_service
from app.services.schema import NewsItemIn

router = APIRouter()


# ══════════════════════════════════════
# 헬스 / 테스트
# ══════════════════════════════════════
@router.get("/test")
async def test():
    """테스트 API 엔드포인트."""
    return {"message": "테스트 성공"}


# ══════════════════════════════════════
# RAG 라우트
# ══════════════════════════════════════
class IngestRequest(BaseModel):
    items: list[NewsItemIn]
    score: bool = True


@router.post("/rag/ingest")
async def rag_ingest(
    body: IngestRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """뉴스 ingest: Chroma 임베딩 저장 + LangChain 감성점수.

    수집 담당(`app/collector/news_crawler.py`) 이 결과를 여기로 POST 하거나,
    내부 코드에서 RAGService.ingest() 를 직접 호출해도 됨.
    """
    return rag.ingest(body.items, score=body.score)


@router.get("/rag/features")
async def rag_features(
    ticker: str = Query(...),
    start: str | None = Query(None, description="YYYY-MM-DD (default: 30일 전)"),
    end: str | None = Query(None, description="YYYY-MM-DD (default: 오늘)"),
    rag: RAGService = Depends(get_rag_service),
):
    """일별 sentiment 피처 (Sentiment_lag1, Sentiment_rolling_3) 반환."""
    s = _parse_date(start) if start else datetime.now(timezone.utc) - timedelta(days=30)
    e = _parse_date(end) if end else datetime.now(timezone.utc)
    df = rag.features(ticker=ticker, start=s, end=e)
    rows = []
    if not df.empty:
        df = df.copy()
        df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
        rows = df.to_dict(orient="records")
    return {"ticker": ticker, "rows": rows}


@router.get("/rag/search")
async def rag_search(
    query: str = Query(...),
    ticker: str | None = None,
    k: int = 5,
    start: str | None = None,
    end: str | None = None,
    rag: RAGService = Depends(get_rag_service),
):
    """RAG 시맨틱 검색."""
    s = _parse_date(start) if start else None
    e = _parse_date(end) if end else None
    results = rag.search(query=query, ticker=ticker, k=k, start=s, end=e)
    return {"query": query, "ticker": ticker, "results": results}


# ══════════════════════════════════════
# 예측 (모델 담당이 채울 곳)
# ══════════════════════════════════════
@router.get("/predict")
async def predict(
    ticker: str = Query(...),
    rag: RAGService = Depends(get_rag_service),
):
    """T+1 방향 예측. 디스코드 봇의 `/c.{ticker}.분석` 이 호출.

    현재는 sentiment 만 반환하는 임시 구현. 모델 담당(`app/models/predictor.py`)
    완성되면 그걸 호출해서 signal/probability 채우면 됨.
    """
    sent = rag.latest_sentiment(ticker, end=datetime.now(timezone.utc))
    return {
        "signal": "neutral",  # TODO: predictor.predict(ticker) 결과로 교체
        "probability": None,
        "sentiment_lag1": sent["sentiment_lag1"],
        "sentiment_rolling_3": sent["sentiment_rolling_3"],
        "n_articles": sent["n_articles"],
        "rationale": (
            f"[임시] 모델 미구현. "
            f"최근 뉴스 감성 lag1={sent['sentiment_lag1']:.3f}, "
            f"rolling_3={sent['sentiment_rolling_3']:.3f} ({sent['n_articles']} articles)."
        ),
    }


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
