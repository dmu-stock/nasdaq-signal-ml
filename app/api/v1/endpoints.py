from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from app.services import watchlist_service
from app.services.analysis_chain import AnalysisChain, get_analysis_chain
from app.services.chart_service import render_candle_png
from app.services.csv_analysis_chain import CSVAnalysisChain, get_csv_analysis_chain
from app.services.qa_chain import StockQAChain, get_qa_chain
from app.services.rag_service import RAGService, get_rag_service
from app.services.schema import NewsItemIn
from app.services.technical_analysis import full_technical

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
    analysis: AnalysisChain = Depends(get_analysis_chain),
    csv_chain: CSVAnalysisChain = Depends(get_csv_analysis_chain),
):
    """종합 분석. 디스코드 봇의 `/c.{ticker}.분석` 이 호출.

    파이프라인:
      1) yfinance OHLCV + sentiment 로 휴리스틱 시그널 계산 (compute_signal)
      2) **LLM 이 CSV+OHLCV 직접 보고 시그널 산출** (csv_analysis_chain)
      3) LLM 결과가 valid 면 우선, 아니면 휴리스틱으로 폴백
      4) 별도 LLM 으로 한국어 코멘트 (say/result/caution/upside_trigger) 생성

    응답:
      - signal / probability / confidence  (LLM 우선, 폴백 가능)
      - 가격 레벨 (recommended_buy / resistance_1,2 / support / current_price)
      - 이동평균 / RSI / trend / flags  (휴리스틱 산출, 참고용)
      - sentiment (lag1, rolling_3)
      - say / result / caution / upside_trigger  (LangChain 구조화 코멘트)
      - signal_source: "llm_csv" | "heuristic"  (어떤 경로로 결정됐는지)
      - llm_reasoning: LLM 의 근거 (LLM 경로일 때만)
    """
    sent = rag.latest_sentiment(ticker, end=datetime.now(timezone.utc))
    sentiment_score = float(sent.get("sentiment_lag1") or 0.0)

    # 1) 휴리스틱 (지표 산출 + 폴백용)
    tech = full_technical(ticker, sentiment_score=sentiment_score)
    if not tech:
        return {
            "ticker": ticker,
            "ok": False,
            "error": f"{ticker} OHLCV 데이터를 가져올 수 없어요 (yfinance 응답 없음).",
        }

    # 2) LLM 기반 CSV+OHLCV 분석 (병행)
    llm_sig = csv_chain.analyze(ticker)

    # 3) 결과 선택: LLM 이 유효하면 LLM, 아니면 휴리스틱
    use_llm = bool(llm_sig.get("ok") and llm_sig.get("valid"))
    if use_llm:
        signal = llm_sig["signal"]
        probability = round(float(llm_sig["probability"]), 3)
        confidence = round(float(llm_sig["confidence"]), 3)
        support = round(float(llm_sig["support"]), 2)
        resistance_1 = round(float(llm_sig["resistance_1"]), 2)
        resistance_2 = round(float(llm_sig["resistance_2"]), 2)
        recommended_buy = round(float(llm_sig["recommended_buy"]), 2)
        signal_source = "llm_csv"
        llm_reasoning = llm_sig.get("reasoning", "")
    else:
        signal = tech["signal"]
        probability = tech["probability_up"]
        confidence = tech["confidence"]
        support = tech["support"]
        resistance_1 = tech["resistance_1"]
        resistance_2 = tech["resistance_2"]
        recommended_buy = tech["recommended_buy"]
        signal_source = "heuristic"
        llm_reasoning = ""

    # 4) 코멘트 LLM — 위에서 선택된 시그널 기준으로 작성
    tech_for_comment = {**tech,
        "signal": signal,
        "probability_up": probability,
        "confidence": confidence,
        "support": support,
        "resistance_1": resistance_1,
        "resistance_2": resistance_2,
        "recommended_buy": recommended_buy,
    }
    commentary = analysis.generate(
        ticker=ticker,
        tech=tech_for_comment,
        sentiment_lag1=sent["sentiment_lag1"],
        sentiment_rolling_3=sent["sentiment_rolling_3"],
    )

    return {
        "ticker": ticker,
        "ok": True,
        # 최종 신호 (LLM 또는 휴리스틱)
        "signal": signal,
        "probability": probability,
        "confidence": confidence,
        "trend": tech["trend"],
        # 가격 레벨
        "current_price": tech["current_price"],
        "recommended_buy": recommended_buy,
        "resistance_1": resistance_1,
        "resistance_2": resistance_2,
        "support": support,
        # 지표 (참고)
        "sma_5": tech["sma_5"],
        "sma_20": tech["sma_20"],
        "sma_60": tech["sma_60"],
        "rsi_14": tech["rsi_14"],
        "flags": tech["flags"],
        "as_of": tech["as_of"],
        # 감성
        "sentiment_lag1": sent["sentiment_lag1"],
        "sentiment_rolling_3": sent["sentiment_rolling_3"],
        "n_articles": sent["n_articles"],
        # 시그널 출처 + LLM 근거
        "signal_source": signal_source,
        "llm_reasoning": llm_reasoning,
        "llm_news_count_used": llm_sig.get("news_count_used", 0),
        # 휴리스틱 결과도 같이 (디버깅/비교용)
        "heuristic": {
            "signal": tech["signal"],
            "probability": tech["probability_up"],
            "confidence": tech["confidence"],
            "support": tech["support"],
            "resistance_1": tech["resistance_1"],
            "resistance_2": tech["resistance_2"],
            "recommended_buy": tech["recommended_buy"],
        },
        # LLM 코멘트
        "say": commentary["say"],
        "result": commentary["result"],
        "caution": commentary["caution"],
        "upside_trigger": commentary["upside_trigger"],
    }


@router.get("/chart")
async def chart(
    ticker: str = Query(...),
    period: str = Query("6mo", description="yfinance period (예: 3mo, 6mo, 1y, 2y)"),
):
    """캔들 + 5/10/20/60일 이평선 + 거래량 PNG 반환."""
    png = render_candle_png(ticker, period=period)
    if not png:
        return Response(
            content=b"",
            media_type="image/png",
            status_code=404,
            headers={"X-Error": f"no OHLCV for {ticker}"},
        )
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},  # 5분 캐시
    )


# ══════════════════════════════════════
# Q&A — 디스코드 `a.` 명령이 호출
# ══════════════════════════════════════
class AskRequest(BaseModel):
    question: str
    ticker: str | None = None


@router.post("/ask")
async def ask(
    body: AskRequest,
    qa: StockQAChain = Depends(get_qa_chain),
):
    """LangChain Q&A.

    - 1단계: 분류기가 '주식 관련 질문인가?' 판정 (LLM 가드레일)
    - 2단계: 관련이면 RAG (Chroma) 로 최근 30일 뉴스 회수
    - 3단계: 컨텍스트 + 질문으로 LLM 답변
    """
    return qa.ask(body.question, ticker=body.ticker)


@router.get("/ask")
async def ask_get(
    q: str = Query(..., description="질문"),
    ticker: str | None = Query(None, description="선택: 종목 티커 힌트"),
    qa: StockQAChain = Depends(get_qa_chain),
):
    """GET 버전 — 브라우저/curl 테스트 편의용."""
    return qa.ask(q, ticker=ticker)


# ══════════════════════════════════════
# 워치리스트 (Member / MemberStock) CRUD
#   디스코드 봇의 /c.내주식.* 명령이 호출
# ══════════════════════════════════════
class WatchlistAddRequest(BaseModel):
    ticker: str
    username: str | None = None
    quantity: float | None = None
    avg_buy_price: float | None = None


class WatchlistUpdateRequest(BaseModel):
    quantity: float | None = None
    avg_buy_price: float | None = None


@router.get("/members/{discord_id}/watchlist")
async def watchlist_list(discord_id: str):
    """사용자의 워치리스트 + 포트폴리오 (수량, 평단 포함, 최신 추가 순)."""
    items = watchlist_service.list_watchlist(discord_id)
    return {"discord_id": discord_id, "count": len(items), "items": items}


@router.post("/members/{discord_id}/watchlist")
async def watchlist_add(discord_id: str, body: WatchlistAddRequest):
    """ticker 추가 (member 자동 생성). 이미 있고 quantity/avg_buy_price 가
    들어오면 업데이트, 안 들어오면 already=True."""
    return watchlist_service.add_to_watchlist(
        discord_id,
        body.ticker,
        username=body.username,
        quantity=body.quantity,
        avg_buy_price=body.avg_buy_price,
    )


@router.patch("/members/{discord_id}/watchlist/{ticker}")
async def watchlist_update(discord_id: str, ticker: str, body: WatchlistUpdateRequest):
    """기존 종목의 수량 / 평단가 업데이트."""
    return watchlist_service.update_watchlist_item(
        discord_id, ticker,
        quantity=body.quantity,
        avg_buy_price=body.avg_buy_price,
    )


@router.delete("/members/{discord_id}/watchlist/{ticker}")
async def watchlist_remove(discord_id: str, ticker: str):
    """워치리스트에서 ticker 제거."""
    return watchlist_service.remove_from_watchlist(discord_id, ticker)


@router.delete("/members/{discord_id}/watchlist")
async def watchlist_clear(discord_id: str):
    """워치리스트 전체 삭제."""
    return watchlist_service.clear_watchlist(discord_id)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
