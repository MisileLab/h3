"""Tests for Mermaid diagram generation."""

import pytest

from src.models import Component, Connection, MethodologyStructure
from src.visualizer import (
    generate_mermaid_diagram,
    generate_comparison_diagram,
    generate_markdown_report,
    validate_mermaid_syntax
)


def test_generate_mermaid_diagram():
    """Mermaid 다이어그램 생성 테스트"""
    components = [
        Component(id="generator", name="Generator", description="MLM component", type="model"),
        Component(id="discriminator", name="Discriminator", description="RTD component", type="model"),
        Component(id="input", name="Input Tokens", description="Input data", type="data")
    ]
    connections = [
        Connection(from_id="input", to_id="generator", label="processed by"),
        Connection(from_id="generator", to_id="discriminator", label="feeds")
    ]

    structure = MethodologyStructure(
        title="Test Paper",
        main_contribution="New method",
        components=components,
        connections=connections,
        key_differences="Better approach"
    )

    diagram = generate_mermaid_diagram(structure)

    assert "```mermaid" in diagram
    assert "graph TD" in diagram
    assert "generator" in diagram
    assert "discriminator" in diagram
    assert "input" in diagram


def test_generate_comparison_diagram():
    """비교 다이어그램 생성 테스트"""
    components = [
        Component(id="generator", name="Generator", description="New approach", type="model"),
        Component(id="discriminator", name="Discriminator", description="New discriminator", type="model")
    ]

    structure = MethodologyStructure(
        title="Test Paper",
        main_contribution="Innovation",
        components=components,
        connections=[],
        key_differences="Major improvement"
    )

    comparison = generate_comparison_diagram(structure)

    assert "```mermaid" in comparison
    assert "graph LR" in comparison
    assert "Traditional Approach" in comparison
    assert "Proposed Method" in comparison


def test_generate_markdown_report():
    """마크다운 리포트 생성 테스트"""
    components = [
        Component(id="test", name="Test Component", description="Test", type="model")
    ]

    structure = MethodologyStructure(
        title="Test Paper",
        main_contribution="Test contribution",
        components=components,
        connections=[],
        key_differences="Test differences"
    )

    report = generate_markdown_report(structure)

    assert "# Test Paper" in report
    assert "Test contribution" in report
    assert "Test Component" in report
    assert "```mermaid" in report


def test_validate_mermaid_syntax():
    """Mermaid 문법 검증 테스트"""
    valid_diagram = "```mermaid\ngraph TD\n    A --> B\n```"
    invalid_diagram = "graph TD\n    A --> B"

    assert validate_mermaid_syntax(valid_diagram) == True
    assert validate_mermaid_syntax(invalid_diagram) == False


def test_component_style_mapping():
    """컴포넌트 스타일 매핑 테스트"""
    from src.visualizer import get_component_style

    assert get_component_style("model") == "model"
    assert get_component_style("layer") == "layer"
    assert get_component_style("process") == "process"
    assert get_component_style("data") == "data"
    assert get_component_style("unknown") == "model"  # fallback