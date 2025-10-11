"""PydanticAI-based paper analyzer."""

import os
from typing import Dict

from pydantic_ai import Agent

from .models import MethodologyStructure, PaperText


async def analyze_paper(paper_text: PaperText) -> MethodologyStructure:
    """
    LLM을 사용하여 논문의 방법론을 구조화된 형태로 추출합니다.

    Args:
        paper_text: 추출된 논문 텍스트

    Returns:
        MethodologyStructure: 구조화된 방법론 정보
    """
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. "
            "API 키를 설정해주세요."
        )

    # PydanticAI 에이전트 설정
    agent = Agent(
        'openai:gpt-4o',  # GPT-4o 사용 (더 빠르고 비용 효율적)
        result_type=MethodologyStructure,
        system_prompt="""
        당신은 NLP 논문 분석 전문가입니다. 논문의 방법론을 분석하여 다음을 추출하세요:

        1. 주요 컴포넌트 (모델, 레이어, 프로세스, 데이터)
        2. 컴포넌트 간 데이터 흐름과 연결 관계
        3. 핵심 혁신 포인트와 기여
        4. 기존 방법과의 차이점

        특히 다음을 중점적으로 분석해주세요:
        - Transformer 기반 모델의 구조 (Encoder, Decoder, Attention)
        - Generator-Discriminator 구조 (GAN, ELECTRA 등)
        - Pre-training과 Fine-tuning 전략
        - Loss function과 최적화 방법
        - 데이터 처리 파이프라인

        컴포넌트 ID는 영문 소문자와 언더스코어만 사용해주세요 (예: generator, discriminator, token_embedding).
        연결 관계는 데이터 흐름을 명확하게 표현해주세요.
        """
    )

    # 분석 요청
    result = await agent.run(f"""
    논문 제목: {paper_text.title}

    논문 초록:
    {paper_text.abstract}

    논문 방법론:
    {paper_text.method}

    이 논문의 핵심 방법론을 구조화해주세요. 다음 사항에特别注意(특별히 주의)해주세요:

    1. 모델 구조의 핵심 컴포넌트를 정확히 식별
    2. 데이터가 어떻게 흐르는지 명확하게 표현
    3. 기존 방법(BERT, GPT 등)과의 차이점을 명확히 설명
    4. 혁신적인 부분을 강조

    예를 들어 ELECTRA 논문의 경우:
    - Generator: MLM으로 마스킹된 토큰 예측
    - Discriminator: 토큰이 진짜인지 가짜인지 분류
    - RTD (Replaced Token Detection): 핵심 학습 목표
    - Generator와 Discriminator의 차이점 (크기, 목적)
    """)

    return result.data


async def analyze_paper_batch(paper_texts: list[PaperText]) -> list[MethodologyStructure]:
    """
    여러 논문을 동시에 분석합니다.

    Args:
        paper_texts: 분석할 논문 텍스트 목록

    Returns:
        list[MethodologyStructure]: 구조화된 방법론 정보 목록
    """
    # 병렬 처리를 위해 각 논문을 동시에 분석
    import asyncio

    tasks = [analyze_paper(paper_text) for paper_text in paper_texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 예외 처리
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"논문 {i+1} 분석 중 오류 발생: {result}")
            # 기본 구조 생성
            valid_results.append(MethodologyStructure(
                title=paper_texts[i].title,
                main_contribution="분석 실패",
                components=[],
                connections=[],
                key_differences="분석 실패"
            ))
        else:
            valid_results.append(result)

    return valid_results