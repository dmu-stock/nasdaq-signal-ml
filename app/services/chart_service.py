"""캔들 차트 PNG 렌더링 (5/10/20/60일 이동평균 + 거래량)."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")  # 서버용 백엔드 (GUI 없이 PNG)
import mplfinance as mpf

from app.services.technical_analysis import fetch_ohlcv


# 한국 주식 앱 스타일 (빨강=상승, 파랑=하락) 가 아니라
# 디스코드 다크 테마에 어울리는 yahoo 스타일 (녹색=상승)
_STYLE = mpf.make_mpf_style(
    base_mpf_style="yahoo",
    rc={
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    },
)


def render_candle_png(ticker: str, period: str = "6mo") -> bytes:
    """캔들+이평선+거래량 차트를 PNG 바이트로 반환. 데이터 없으면 빈 bytes."""
    df = fetch_ohlcv(ticker, period=period)
    if df.empty:
        return b""

    buf = io.BytesIO()
    try:
        mpf.plot(
            df,
            type="candle",
            mav=(5, 10, 20, 60),
            volume=True,
            style=_STYLE,
            title=f"{ticker}  ·  {df.index[-1].date()}",
            ylabel="Price ($)",
            ylabel_lower="Volume",
            figratio=(16, 9),
            figscale=1.1,
            tight_layout=True,
            savefig=dict(fname=buf, dpi=130, bbox_inches="tight"),
        )
    except Exception:
        return b""
    buf.seek(0)
    return buf.read()
