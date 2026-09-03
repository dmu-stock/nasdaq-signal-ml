"""전날 미국 지수·매크로 자산 등락 (아침 브리핑용)."""
from __future__ import annotations
from dataclasses import dataclass
import yfinance as yf
from app.config.config import TICKERS

_INDEXES = {"^GSPC": "S&P500", "^IXIC": "NASDAQ", "^DJI": "Dow", "^VIX": "VIX"}
_MACRO = {"CL=F": "WTI유가", "GC=F": "금", "^TNX": "미10년물금리", "DX-Y.NYB": "달러지수"}


@dataclass
class IndexMove:
    name: str
    close: float
    change_pct: float
    date: str


@dataclass
class Mover:
    ticker: str
    change_pct: float


def fetch_movers(tickers: list[str] | None = None, top_n: int = 5) -> tuple[list[Mover], list[Mover]]:
    """유니버스 전날 등락 → (상위 상승, 상위 하락)."""
    tickers = tickers or TICKERS
    close_prices = yf.download(tickers, period="5d", progress=False)["Close"]   # 배치 다운로드 (빠름)
    moves: list[Mover] = []
    for ticker in tickers:
        try:
            closes = close_prices[ticker].dropna()
            if len(closes) >= 2:
                change_pct = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
                moves.append(Mover(ticker, round(change_pct, 2)))
        except Exception:
            continue
    moves.sort(key=lambda move: move.change_pct, reverse=True)
    return moves[:top_n], list(reversed(moves[-top_n:]))     # (상승, 하락)


def _fetch_asset_moves(symbols: dict[str, str]) -> list[IndexMove]:
    """심볼→표시이름 딕셔너리의 전날 종가·등락률 수집 (지수/매크로 공용)."""
    moves: list[IndexMove] = []
    for symbol, display_name in symbols.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if len(hist) < 2:
                continue
            prev_close, last_close = hist["Close"].iloc[-2], hist["Close"].iloc[-1]
            moves.append(IndexMove(
                name=display_name,
                close=round(float(last_close), 2),
                change_pct=round((last_close / prev_close - 1) * 100, 2),
                date=str(hist.index[-1].date()),          # 실제 세션 날짜
            ))
        except Exception as error:
            print(f"[market_data] {symbol} 실패: {error}")
    return moves


def fetch_index_moves() -> list[IndexMove]:
    """최근 완료된 미국 세션의 지수 등락 (S&P500·나스닥·다우·VIX)."""
    return _fetch_asset_moves(_INDEXES)


def fetch_macro_moves() -> list[IndexMove]:
    """유가·금·금리·달러 등 매크로 자산 등락 (지정학·거시 신호)."""
    return _fetch_asset_moves(_MACRO)
