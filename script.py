# 샘플 Python 스크립트입니다.

# Shift+F10을(를) 눌러 실행하거나 내 코드로 바꿉니다.
# 클래스, 파일, 도구 창, 액션 및 설정을 어디서나 검색하려면 Shift 두 번을(를) 누릅니다.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NK→SK 농업 적성 추천: FastAPI 백엔드 (단일 요청에 12문항 모두 수신)
- POST /recommendations : 12개 문항(q1~q12)을 한 번에 받아 프롬프트를 구성하고 LLM을 호출하여 JSON 결과 반환
- 옵션: run_model=false 로 주면 LLM 호출 없이 프롬프트만 반환
- 환경변수: OPENAI_API_KEY, MODEL_NAME(기본: gpt-5-2025-08-07)

실행 예)
  uvicorn app:app --reload --port 8000

테스트 예)
  curl -X POST http://localhost:8000/recommendations \
    -H 'Content-Type: application/json' \
    -d '{
      "q1":"Y",
      "q2":["논농사(쌀)", "시설채소(온실·비닐하우스)"],
      "q3":5,
      "q4":["경작/재배", "기계·설비 운전(트랙터·양수기 등)"],
      "q5":["노지 재배", "관수(점적/스프링클러) 운용"],
      "q6":["야외(논·밭)", "온실/시설"],
      "q7":"보통",
      "q8":["스마트팜"],
      "q9":"중간 정도가 좋다",
      "q10":"토목 보조",
      "q11":"충청",
      "q12":"취업(농장 근로)",
      "run_model": true
    }'
"""

from typing import List, Dict, Any, Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import json
import os

# -----------------------------
# FastAPI 초기화
# -----------------------------
app = FastAPI(title="NK→SK 농업 적성 추천 API", version="1.1.0")

# CORS (로컬 프런트엔드용)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 특정 오리진으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 최근 matches 문자열 캐시(POST 완료 후 업데이트, GET에서 제공)
LAST_MATCHES_STR: Optional[str] = None

# -----------------------------
# 데이터 모델(12문항을 한 번에 수신)
# -----------------------------

YesNo = Literal["Y", "N"]

class Answers(BaseModel):
    # 1
    q1: YesNo = Field(..., description="농업 경험 여부: 'Y' 또는 'N'")
    # 2
    q2: List[str] = Field(default_factory=list, description="수행 분야(복수)")
    # 3
    q3: int = Field(..., ge=0, le=80, description="경력 기간(년)")
    # 4
    q4: List[str] = Field(default_factory=list, description="맡았던 역할(복수)")
    # 5
    q5: List[str] = Field(default_factory=list, description="재배 시스템/방식(복수)")
    # 6
    q6: List[str] = Field(default_factory=list, description="선호 작업 환경(복수)")
    # 7
    q7: Literal["높음", "보통", "낮음"]
    # 8
    q8: List[str] = Field(default_factory=list, description="배우고 싶은 분야(복수)")
    # 9
    q9: Literal["바빠도 괜찮다(높음 가능)", "중간 정도가 좋다", "낮은 노동 강도 선호"]
    # 10
    q10: Optional[str] = Field(default="", description="농업 외 직업 경험(서술형)")
    # 11
    q11: Literal["수도권", "강원", "충청", "전라", "경상", "제주", "상관없음"]
    # 12
    q12: Literal[
        "취업(농장 근로)",
        "창업(귀농·작물 재배)",
        "기술직(스마트팜 운영·드론 등)",
        "안정적인 단순작업",
        "아직 모르겠다",
    ]

    # 실행 옵션
    run_model: bool = Field(default=True, description="LLM 호출 여부")

    @field_validator("q10", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return v or ""

# -----------------------------
# 프롬프트 빌더 (원본 CLI 로직을 백엔드에 맞게 그대로 유지)
# -----------------------------

def build_system_prompt() -> str:
    return (
        "당신은 북한 농업 경력 기반으로 남한에서의 농업 직무·작목·학습경로를 매칭하는 취업/정착 코치입니다. "
        "답변은 한국어로 간결하지만 실무적으로 제시하세요. "
        "남한의 제도·안전·위생 기준, 기후·유통 구조, 스마트농업/스마트축산 전환 추세를 고려하세요."
    )


def build_user_prompt(answers: Dict[str, Any]) -> str:
    out = [
        "## 사용자 응답 요약",
        f"1. 농업 경험 여부: {answers['q1']}",
        f"2. 수행 분야(복수): {', '.join(answers['q2']) if answers['q2'] else '무응답'}",
        f"3. 경력 기간(년): {answers['q3']}",
        f"4. 맡았던 역할(복수): {', '.join(answers['q4']) if answers['q4'] else '무응답'}",
        f"5. 활용 재배 시스템/방식(복수): {', '.join(answers['q5']) if answers['q5'] else '무응답'}",
        f"6. 선호 작업 환경(복수): {', '.join(answers['q6']) if answers['q6'] else '무응답'}",
        f"7. 자기평가 체력: {answers['q7']}",
        f"8. 배우고 싶은 분야(복수): {', '.join(answers['q8']) if answers['q8'] else '무응답'}",
        f"9. 노동 강도 선호: {answers['q9']}",
        f"10. 농업 외 직업 경험: {answers['q10'] or '무응답'}",
        f"11. 정착 희망 지역: {answers['q11']}",
        f"12. 장기 목표 유형: {answers['q12']}",
        "",
        "## 지시사항",
        "- 아래 '결과 포맷'에 맞춰 남한에서의 적합 분야 상위 3가지를 제시하세요.",
        "- 후보마다: 적합 이유(기술 이전성, 안전/위생·법규 차이, 노동강도), 예상 진입경로(취업/교육/귀농), "
        "권장 지역·작목 예시, 필요 역량/장비/자본 범위를 간단히 정리하세요.",
        "- 가능하면 공공교육/자격(예: 농기계, PLS, HACCP, 스마트팜)과 단기 현장실습(일학습병행/농업마이스터대학/지자체 센터) 제안을 포함하세요.",
        "",
        "## 결과 포맷(JSON)",
        "{",
        '  "matches": [',
        "    {",
        '      "분야": "예: 시설채소 스마트팜",',
        '      "적합_이유": "짧은 문장 2~3개",',
        '      "권장_지역_작목": ["지역-작목1", "지역-작목2"],',
        '      "진입경로": ["현장취업", "교육/자격", "창업(초기)"],',
        '      "필요역량_장비_자본": "핵심만 요약",',
        '      "다음단계": ["기관/과정 제안", "체험/실습 제안"]',
        "    },",
        "    { ... },",
        "    { ... }",
        "  ]",
        "}",
        "",
        "출력은 반드시 유효한 JSON만 반환하세요. 설명 문장이나 마크다운을 추가하지 마세요.",
    ]
    return "\n".join(out)

# -----------------------------
# LLM 호출 유틸
# -----------------------------

def call_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """OpenAI Chat Completions 호출 후 JSON 파싱 시도."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = os.getenv("MODEL_NAME", "gpt-5-2025-08-07")

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = completion.choices[0].message.content.strip()

    # 모델이 유효 JSON을 반환하도록 지시했지만, 안전하게 파싱 시도
    parsed: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    try:
        parsed = json.loads(content)
    except Exception as e:
        error = f"응답 JSON 파싱 실패: {e}"

    return {
        "raw": content,
        "json": parsed,
        "parse_error": error,
        "model": model_name,
    }

# -----------------------------
# 엔드포인트
# -----------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

# 프런트엔드가 간단히 가져갈 수 있게 matches만 문자열로 반환
# - Content-Type: text/plain
# - POST /recommendations 성공 후 최신 결과를 제공 (서버 메모리 캐시)
from fastapi import Response, status

@app.get("/matches")
async def get_matches_plain():
    if LAST_MATCHES_STR is None:
        return Response("no matches yet", media_type="text/plain", status_code=status.HTTP_404_NOT_FOUND)
    return Response(LAST_MATCHES_STR, media_type="text/plain")


@app.post("/recommendations")
async def recommendations(payload: Answers):
    # 1) 프롬프트 구성
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload.dict())

    # 2) run_model 플래그에 따라 호출
    if not payload.run_model:
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "run_model": False,
            "note": "run_model=false 이므로 LLM을 호출하지 않았습니다.",
        }

    # 3) LLM 호출
    try:
        result = call_llm(system_prompt, user_prompt)
        # matches만 뽑아 문자열로 캐시 (프런트 GET 용)
        global LAST_MATCHES_STR
        LAST_MATCHES_STR = None
        try:
            if result.get("json") and isinstance(result["json"], dict) and "matches" in result["json"]:
                LAST_MATCHES_STR = json.dumps(result["json"]["matches"], ensure_ascii=False)
            else:
                # 파싱 실패 시 원문 전체 저장
                LAST_MATCHES_STR = result.get("raw", "")
        except Exception:
            LAST_MATCHES_STR = result.get("raw", "")

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "run_model": True,
            "llm_result": result,
            "matches_string": LAST_MATCHES_STR,
        }
    except Exception as e:
        # 호출 실패 시 프롬프트라도 반환
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "run_model": True,
            "error": str(e),
            "hint": "OPENAI_API_KEY, MODEL_NAME, 요금제/모델 접근 권한을 확인하세요.",
        }


# -----------------------------
# 로컬 실행 진입점 (uvicorn)
# -----------------------------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


