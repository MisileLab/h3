"""Pydantic models for paper methodology structure."""

from pydantic import BaseModel, Field
from typing import List, Literal


class Component(BaseModel):
    """논문 방법론의 구성 요소"""
    id: str = Field(description="고유 식별자 (예: 'generator', 'discriminator')")
    name: str = Field(description="컴포넌트 이름")
    description: str = Field(description="간단한 설명")
    type: Literal["model", "layer", "process", "data"] = Field(
        description="컴포넌트 타입"
    )


class Connection(BaseModel):
    """컴포넌트 간 연결"""
    from_id: str = Field(description="시작 컴포넌트 ID")
    to_id: str = Field(description="도착 컴포넌트 ID")
    label: str = Field(description="연결 설명 (예: 'generates tokens')")


class MethodologyStructure(BaseModel):
    """논문의 방법론 구조"""
    title: str = Field(description="논문 제목")
    main_contribution: str = Field(description="핵심 기여/혁신")
    components: List[Component] = Field(description="주요 구성 요소 목록")
    connections: List[Connection] = Field(description="컴포넌트 간 연결")
    key_differences: str = Field(description="기존 방법과의 차이점")


class PaperText(BaseModel):
    """논문 텍스트 구조"""
    title: str = Field(description="논문 제목")
    abstract: str = Field(description="초록")
    method: str = Field(description="방법론 섹션")
    full_text: str = Field(description="전체 텍스트")