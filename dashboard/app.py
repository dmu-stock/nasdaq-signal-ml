"""로보어드바이저 대시보드 (Streamlit).

FastAPI(http://localhost:8000)를 HTTP로만 호출한다 — 모델 직접 로드 X (MSA).
실행: streamlit run dashboard/app.py
"""
import os
import uuid

import requests
import streamlit as st

# 로컬: localhost, 배포: 환경변수 API_URL(배포된 API 주소) 자동 사용
API = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="US 주식 AI 리서치", page_icon="📈", layout="centered")
st.title("📈 US 주식 AI 리서치")
st.caption("FastAPI 백엔드를 HTTP로 호출하는 대시보드 (MSA)")


def api_post(path: str, payload: dict):
    try:
        res = requests.post(f"{API}{path}", json=payload, timeout=90)
        res.raise_for_status()
        return res.json(), None
    except Exception as error:
        return None, str(error)


def api_get(path: str):
    try:
        res = requests.get(f"{API}{path}", timeout=90)
        res.raise_for_status()
        return res.json(), None
    except Exception as error:
        return None, str(error)


tab_signal, tab_news, tab_research, tab_recap, tab_chat = st.tabs(
    ["🎯 매수 시그널", "📰 뉴스 요약", "🔎 과거 리서치", "🌅 아침 브리핑", "💬 챗봇"]
)

# ── 탭: ML 매수 시그널 ──
with tab_signal:
    st.subheader("ML 매수 시그널")
    st.caption(
        "논문 모델(LightGBM 단기타이밍 + Dual-LSTM 중기순위) 재정렬 앙상블. "
        "GBM이 후보(3일 +2.5%)를 선정하면 LSTM이 순위(5일 상위30%)를 재정렬합니다."
    )
    if st.button("오늘의 시그널 조회", key="signal_btn"):
        with st.spinner("모델 추론 중... (첫 호출은 41종목 예측으로 수십초 걸립니다)"):
            data, error = api_get("/signal")
        if error:
            st.error(error)
        elif data:
            st.caption(f"VIX {data.get('vix', '-')} · 가드레일: VIX ≥ 30이면 매수 중단")
            picks = data.get("picks", [])
            if not picks:
                st.warning(data.get("reason") or "오늘 매수 후보 없음 — 현금 보유 권장")
            else:
                for rank, pick in enumerate(picks, 1):
                    badge = " · 🧪 확장(참고)" if pick.get("extra") else " · 🎓 학습종목"
                    st.markdown(
                        f"### {rank}. {pick['ticker']}{badge}\n"
                        f"종합점수 **{pick['final_prob']:.3f}** · "
                        f"GBM(단기) {pick['prob_lgb']:.3f} · LSTM(중기) {pick['prob_lstm']:.3f}"
                    )
                    st.progress(min(max(pick["final_prob"], 0.0), 1.0))
                st.caption("🎓 학습종목 = 검증된 유니버스 · 🧪 확장종목 = 학습분포 밖(참고용)")
            st.caption("⚠️ 동일 유니버스 내 상대 순위 신호입니다. 절대 수익을 보장하지 않으며 교육 목적입니다.")

# ── 탭 1: 최신 뉴스 요약 ──
with tab_news:
    st.subheader("최신 뉴스 요약")
    ticker = st.text_input("종목 티커", "NVDA", key="news_ticker")
    if st.button("요약하기", key="news_btn"):
        with st.spinner("뉴스 수집·요약 중..."):
            data, error = api_post("/news/summary", {"ticker": ticker})
        if error:
            st.error(error)
        elif data:
            st.info(f"**전반 톤** — {data.get('overall_tone', '')}")
            st.markdown("**핵심 포인트**")
            for point in data.get("key_points", []):
                st.markdown(f"- {point['point']}  \n  ↳ 출처: {point['source']}")
            st.caption(f"특이 이벤트: {data.get('notable_events', '-')} · 기준일 {data.get('as_of', '')}")

# ── 탭 2: 과거 아카이브 리서치 (RAG) ──
with tab_research:
    st.subheader("과거 뉴스 아카이브 리서치 (RAG)")
    question = st.text_input("질문", "엔비디아가 작년에 왜 올랐어?", key="rag_q")
    if st.button("검색·분석", key="rag_btn"):
        with st.spinner("검색·분석 중..."):
            data, error = api_post("/research", {"question": question, "top_k": 5})
        if error:
            st.error(error)
        elif data:
            st.markdown(data.get("answer", ""))
            st.caption(f"근거 충분성: {data.get('enough', '')} · 검색어: {data.get('search_query', '')}")
            with st.expander("출처 보기"):
                for cite in data.get("citations", []):
                    st.write(f"- {cite['source']} ({cite['published']})")

            # 에이전트의 자기교정(Self-Correction) 과정 노출 = reasoning trace
            attempts = data.get("attempts", [])
            if attempts:
                with st.expander(f"🔎 검색 과정 · {len(attempts)}회 시도 (self-correction)"):
                    for step in attempts:
                        mark = "✅ 충분" if step.get("enough") == "충분" else "🔁 부족 → 재검색"
                        st.markdown(f"**{step['attempt']}차** · `{step['query']}` → {mark}")
                    if attempts[-1].get("enough") != "충분":
                        st.caption("여러 각도로 재검색했지만 근거를 못 찾아 정직하게 '부족'으로 답했습니다.")

# ── 탭 3: 아침 증시 브리핑 ──
with tab_recap:
    st.subheader("전날 미국증시 브리핑")
    if st.button("브리핑 생성", key="recap_btn"):
        with st.spinner("지수·특징주·매크로 종합 중... (몇 초 걸립니다)"):
            data, error = api_get("/market/recap")
        if error:
            st.error(error)
        elif data:
            st.header(data.get("headline", ""))
            st.write(f"**지수** — {data.get('index_summary', '')}")
            st.write(f"**테마** — {data.get('market_theme', '')}")
            st.write(f"**특징주** — {data.get('notable_movers', '')}")
            st.warning(f"**매크로·지정학** — {data.get('macro_geopolitics', '')}")
            st.markdown("**주요 이슈**")
            for point in data.get("key_points", []):
                st.markdown(f"- {point}")
            st.caption(f"관전포인트: {data.get('watch_today', '')} · {data.get('as_of', '')}")

# ── 탭 4: 챗봇 (이어서 대화) ──
with tab_chat:
    st.subheader("리서치 챗봇")
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("예: 엔비디아 요즘 어때?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                data, error = api_post(
                    "/chat", {"message": prompt, "session_id": st.session_state.session_id}
                )
            reply = data.get("reply", "") if data else f"오류: {error}"
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

st.divider()
st.caption(
    "⚠️ 본 시스템은 교육 목적으로 개발되었으며 실제 투자 조언이 아닙니다. "
    "과거 성과가 미래 수익을 보장하지 않습니다."
)
