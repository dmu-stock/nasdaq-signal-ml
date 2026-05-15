"""
headline_preprocessor.py

[역할]
- 뉴스 헤드라인을 감정 분석(FinBERT)에 적합하도록 LLM으로 전처리
- 긍정/부정 방향은 유지하면서 노이즈(기자 의견, 완충 표현 등)만 제거
- FinBERT 입력용 '정제된 한 문장(clean_headline)' 생성

[중요한 설계 원칙]
- 이 모듈은 감정을 판단하지 않음
- 긍정/부정/중립 분류는 FinBERT의 역할
- 본 모듈은 감정 분석 입력 품질을 높이는 전처리 단계에만 집중

[수정 이력 요약]
- os.getenv("") → os.getenv("OPENAI_API_KEY") 수정 (환경변수 버그 수정)
- 잘못된 OpenAI 메서드 호출 수정
- 단건 호출 → 배치 호출 방식 추가 (API 호출 횟수 감소)
"""

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


# ──────────────────────────────────────────────
# 환경 변수 로드 및 OpenAI 클라이언트 초기화
#
# ※ 반드시 .env 파일 또는 시스템 환경변수에
#    OPENAI_API_KEY가 설정되어 있어야 함
#
#   Mac / Linux:
#     export OPENAI_API_KEY="sk-..."
#
#   Windows (CMD):
#     set OPENAI_API_KEY=sk-...
# ──────────────────────────────────────────────

load_dotenv()  # .env 파일에서 환경변수 로드
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def preprocess_headline(headline: str) -> str:
    """
    단일 뉴스 헤드라인을 LLM으로 전처리

    ※ 여러 개의 헤드라인을 처리할 경우
       preprocess_headlines_batch() 사용 권장
       (API 호출 횟수 절감 목적)

    Parameters
    ----------
    headline : str
        Yahoo Finance 등에서 수집한 원본 뉴스 헤드라인

    Returns
    -------
    str
        FinBERT 감정 분석에 적합하도록 정제된 헤드라인
    """

    # 빈 문자열 또는 공백만 있는 입력 방어
    if not headline or not headline.strip():
        return headline

    # LLM 프롬프트 구성
    # 핵심 포인트:
    # - 감정 방향 유지
    # - 요약/해석 금지
    # - 노이즈만 제거
    prompt = f"""
다음은 금융 뉴스 헤드라인이다.
이 문장을 감정 분석에 적합하도록 정제하라.

규칙:
1. 원래 헤드라인의 긍정/부정 톤은 반드시 유지할 것
2. 기자의 의견, 완충 표현("analysts say", "reportedly" 등), 부가 설명은 제거할 것
3. 요약하거나 새로운 해석을 추가하지 말 것
4. 한 문장으로 유지할 것
5. 감정 판단(긍정/부정/중립)을 직접 언급하지 말 것
6. 정제된 문장만 출력하고, 설명이나 따옴표는 붙이지 말 것

헤드라인:
{headline}
"""

    # ── OpenAI API 호출 ─────────────────────────────
    # chat.completions.create:
    #  - 표준 채팅 기반 텍스트 생성 API
    #  - gpt-4o-mini: 비용 대비 성능이 좋아 헤드라인 전처리에 적합
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                # system 역할: 모델의 행동 규칙 정의
                "role": "system",
                "content": (
                    "너는 금융 뉴스 헤드라인을 감정 분석용으로 정제하는 전문가다. "
                    "지시에 따라 정제된 문장만 반환한다."
                ),
            },
            {
                # user 역할: 실제 작업 지시
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,  # 창의성 최소화 → 항상 일관된 출력
        max_tokens=200,   # 헤드라인은 짧으므로 200 토큰이면 충분
    )

    # 응답에서 정제된 텍스트만 추출
    return response.choices[0].message.content.strip()


def preprocess_headlines_batch(headlines: List[str]) -> List[str]:
    """
    여러 개의 뉴스 헤드라인을 한 번의 API 호출로 일괄 전처리

    [장점]
    - 단건 호출 대비 API 호출 횟수 대폭 감소
    - 뉴스 개수가 많아질수록 비용/속도 면에서 유리

    [동작 방식]
    1. 헤드라인을 번호 목록 형태로 묶어서 LLM에 전달
    2. LLM이 번호 순서대로 정제된 결과 반환
    3. 결과를 파싱하여 원본 순서 그대로 재배치

    [안전 장치]
    - API 호출 실패 시 → 원본 헤드라인 그대로 반환
    - 파싱 결과 수 불일치 시 → 원본 유지

    Parameters
    ----------
    headlines : List[str]
        전처리할 뉴스 헤드라인 리스트

    Returns
    -------
    List[str]
        전처리된 헤드라인 리스트
        (입력과 동일한 순서 및 길이 유지)
    """

    # 빈 리스트 입력 방어
    if not headlines:
        return []

    # 비어있지 않은 헤드라인의 인덱스만 추출
    valid_indices = [i for i, h in enumerate(headlines) if h and h.strip()]

    # 전부 빈 문자열이면 그대로 반환
    if not valid_indices:
        return headlines

    # ── 배치 프롬프트 구성 ─────────────────────────
    # "번호. 헤드라인" 형식으로 나열
    numbered_headlines = "\n".join(
        [f"{i + 1}. {headlines[idx]}" for i, idx in enumerate(valid_indices)]
    )

    prompt = f"""
아래 금융 뉴스 헤드라인들을 각각 감정 분석에 적합하도록 정제하라.

규칙:
1. 각 헤드라인의 긍정/부정 톤은 반드시 유지할 것
2. 감정 및 시장 방향성과 관련 없는 불필요한 수식만 제거할 것
3. 요약하거나 새로운 해석을 추가하지 말 것
4. 각각 한 문장으로 유지할 것
5. 감정 판단(긍정/부정/중립)을 직접 언급하지 말 것
6. 감정 단서를 제거하지 말고 유지하라
7. 원문의 언어를 절대 변경하지 말 것
8. 영어는 영어로 유지할 것
9. 번역하지 말 것

출력 형식 (반드시 준수):
- 번호. 정제된 헤드라인
- 설명이나 추가 텍스트 없이 번호 목록만 출력

헤드라인 목록:
{numbered_headlines}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 금융 뉴스 헤드라인을 감정 분석용으로 정제하는 전문가다. "
                        "지시에 따라 번호 목록 형식으로만 반환한다."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=1000,  # 배치 처리이므로 토큰 여유 확보
        )

        raw_output = response.choices[0].message.content.strip()

        # ── 응답 파싱 ───────────────────────────────
        # "1. 텍스트" → "텍스트" 형태로 추출
        parsed_lines = []
        for line in raw_output.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line[0].isdigit() and ". " in line:
                cleaned = line.split(". ", 1)[1].strip()
                parsed_lines.append(cleaned)

        # ── 결과 재조합 ─────────────────────────────
        # 기본값: 원본 헤드라인 유지
        result = list(headlines)

        if len(parsed_lines) == len(valid_indices):
            for i, idx in enumerate(valid_indices):
                result[idx] = parsed_lines[i]
        else:
            print(
                f"[경고] 배치 전처리 파싱 수 불일치 "
                f"(기대: {len(valid_indices)}, 실제: {len(parsed_lines)}) → 원본 유지"
            )

        return result

    except Exception as e:
        # API 호출 실패 시 전체 파이프라인 중단 방지
        print(f"[에러] 배치 전처리 실패 → 원본 사용: {e}")
        return headlines


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_headlines = [
        "Tesla shares rise despite weak deliveries, analysts cautious",
        "NVIDIA stock plunges after mixed earnings outlook",
        "Apple rallies on strong iPhone demand in China",
        "",  # 빈 문자열 처리 테스트
    ]

    print("=== 배치 전처리 테스트 ===\n")
    results = preprocess_headlines_batch(test_headlines)
    
    for before, after in zip(test_headlines, results):
        print(f"원본    : {before}")
        print(f"전처리  : {after}")
        print("-" * 60)
