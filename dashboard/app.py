"""로보어드바이저 대시보드 (Streamlit).

FastAPI(http://localhost:8000)를 HTTP로만 호출한다 — 모델 직접 로드 X (MSA).
실행: streamlit run dashboard/app.py
"""
import uuid

import requests
import streamlit as st

API = "http://localhost:8000/api/v1"

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


tab_news, tab_research, tab_recap, tab_chat = st.tabs(
    ["📰 뉴스 요약", "🔎 과거 리서치", "🌅 아침 브리핑", "💬 챗봇"]
)

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
