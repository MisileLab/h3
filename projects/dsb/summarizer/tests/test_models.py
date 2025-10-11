"""Tests for Pydantic models."""

import pytest

from src.models import Component, Connection, MethodologyStructure, PaperText


def test_component_creation():
    """Component 모델 생성 테스트"""
    component = Component(
        id="generator",
        name="Generator",
        description="MLM pre-training component",
        type="model"
    )
    assert component.id == "generator"
    assert component.type == "model"


def test_connection_creation():
    """Connection 모델 생성 테스트"""
    connection = Connection(
        from_id="generator",
        to_id="discriminator",
        label="generates tokens for"
    )
    assert connection.from_id == "generator"
    assert connection.to_id == "discriminator"


def test_methodology_structure_creation():
    """MethodologyStructure 모델 생성 테스트"""
    components = [
        Component(id="gen", name="Generator", description="Gen component", type="model"),
        Component(id="dis", name="Discriminator", description="Dis component", type="model")
    ]
    connections = [
        Connection(from_id="gen", to_id="dis", label="feeds")
    ]

    structure = MethodologyStructure(
        title="Test Paper",
        main_contribution="New approach",
        components=components,
        connections=connections,
        key_differences="Better than BERT"
    )

    assert len(structure.components) == 2
    assert len(structure.connections) == 1
    assert structure.title == "Test Paper"


def test_paper_text_creation():
    """PaperText 모델 생성 테스트"""
    paper = PaperText(
        title="Test Paper",
        abstract="This is a test abstract",
        method="This is the method section",
        full_text="Full paper text here"
    )
    assert paper.title == "Test Paper"
    assert len(paper.abstract) > 0