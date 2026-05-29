"""LangChain 기반 종합 분석 코멘트 생성.

입력: technical_analysis.full_technical() 결과 + sentiment 정보
출력: { say, result, caution, upside_trigger } 구조화 텍스트
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings


class AnalysisSay(BaseModel):
    say: str = Field(
        ...,
        description=(
            "AI 분석 요약 (2-3문장). 상승확률/신뢰도를 자연스럽게 녹여 한국어로. "
            "예: '전체적으로 상승 확률이 높고, 투자자 신뢰도 또한 강력합니다…'"
        ),
    )
    result: str = Field(
        ...,
        description=(
            "분석결과 (1-2문장). 현재 추세 + 이동평균/RSI 정합을 종합한 결론. "
            "예: '기술적 분석 결과, 전체적으로 상승세가 지속되고 매수 신호가 강하게 나타나고 있습니다.'"
        ),
    )
    caution: str = Field(
        ...,
        description=(
            "주의사항 (1-2문장). 단기 조정 위험, 과매수, 저항선 부근 경계 등. "
            "예: '저항선 1차 부근에서 단기 조정이 발생할 수 있습니다.'"
        ),
    )
    upside_trigger: str = Field(
        ...,
        description=(
            "상승요건 (1-2문장). 추가 상승을 위해 필요한 조건. 가격 레벨 명시. "
            "예: '가격이 230달러를 돌파하고 거래량이 동반되면 추가 상승 모멘텀이 확인됩니다.'"
        ),
    )


_SYSTEM = (
    "너는 미국 주식 기술적 분석가다. 주어진 지표(현재가, 지지/저항, 이동평균, RSI, "
    "추세, 상승확률, 감성점수)를 바탕으로 한국어 분석 코멘트를 작성한다.\n\n"
    "원칙:\n"
    "1) 매수/매도 단정 X. '강하게 나타난다', '가능성이 높다' 같은 가능 표현.\n"
    "2) 가격 레벨은 입력으로 받은 숫자만 그대로 사용 (창작 금지).\n"
    "3) 모든 문장 정중체로 (∼습니다, ∼됩니다).\n"
    "4) 한 항목은 1~2문장. 길지 않게.\n"
)

_USER = (
    "종목: {ticker}\n"
    "기준일: {as_of}\n"
    "현재가: ${current_price}\n"
    "추천 매수가: ${recommended_buy}\n"
    "저항선: 1차 ${resistance_1} / 2차 ${resistance_2}\n"
    "지지선: ${support}\n"
    "이동평균: 5일 ${sma_5} / 20일 ${sma_20} / 60일 ${sma_60}\n"
    "RSI(14): {rsi_14}\n"
    "추세: {trend}\n"
    "상승확률: {probability_up} (신뢰도 {confidence})\n"
    "신호 정합 플래그: {flags}\n"
    "뉴스 감성: lag1={sentiment_lag1}, rolling_3={sentiment_rolling_3}\n\n"
    "위 지표를 종합해서 say / result / caution / upside_trigger 네 항목을 구조화 형식으로 답하라."
)


class AnalysisChain:
    def __init__(self, settings: Settings):
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        ).with_structured_output(AnalysisSay)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user", _USER),
        ])
        self.chain = self.prompt | self.llm

    def generate(
        self,
        ticker: str,
        tech: dict,
        sentiment_lag1: float = 0.0,
        sentiment_rolling_3: float = 0.0,
    ) -> dict:
        try:
            result: AnalysisSay = self.chain.invoke({
                "ticker": ticker,
                "as_of": tech.get("as_of", ""),
                "current_price": tech.get("current_price", "-"),
                "recommended_buy": tech.get("recommended_buy", "-"),
                "resistance_1": tech.get("resistance_1", "-"),
                "resistance_2": tech.get("resistance_2", "-"),
                "support": tech.get("support", "-"),
                "sma_5": tech.get("sma_5", "-"),
                "sma_20": tech.get("sma_20", "-"),
                "sma_60": tech.get("sma_60", "-"),
                "rsi_14": tech.get("rsi_14", "-"),
                "trend": tech.get("trend", "-"),
                "probability_up": tech.get("probability_up", "-"),
                "confidence": tech.get("confidence", "-"),
                "flags": ", ".join(tech.get("flags", []) or []) or "-",
                "sentiment_lag1": sentiment_lag1,
                "sentiment_rolling_3": sentiment_rolling_3,
            })
            return result.model_dump()
        except Exception as e:
            return {
                "say": f"AI 분석 코멘트 생성 실패 ({type(e).__name__}). 기술 지표는 위 수치 참고하세요.",
                "result": "지표만 산출됨.",
                "caution": "LLM 호출 실패로 상세 코멘트 미생성.",
                "upside_trigger": "-",
            }


@lru_cache
def get_analysis_chain() -> AnalysisChain:
    return AnalysisChain(get_settings())
