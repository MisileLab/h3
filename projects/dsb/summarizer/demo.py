#!/usr/bin/env python3
"""
Paper Visualizer Demo Script

This script demonstrates how to use the paper visualizer without requiring a PDF file.
It creates sample data to show the visualization functionality.
"""

import asyncio
from src.models import Component, Connection, MethodologyStructure, PaperText
from src.visualizer import generate_markdown_report, generate_mermaid_diagram


def create_electra_example() -> MethodologyStructure:
    """Create a sample ELECTRA methodology structure for demonstration."""
    components = [
        Component(
            id="input_tokens",
            name="Input Tokens",
            description="Original input sequence from corpus",
            type="data"
        ),
        Component(
            id="generator",
            name="Generator",
            description="Small masked language model (MLM) that replaces masked tokens",
            type="model"
        ),
        Component(
            id="replaced_tokens",
            name="Replaced Tokens",
            description="Tokens where some are replaced by generator predictions",
            type="data"
        ),
        Component(
            id="discriminator",
            name="Discriminator",
            description="Full-size model that predicts if each token is original or replaced",
            type="model"
        ),
        Component(
            id="token_predictions",
            name="Token Predictions",
            description="Binary classification: original vs replaced token",
            type="data"
        )
    ]

    connections = [
        Connection(
            from_id="input_tokens",
            to_id="generator",
            label="masked input (15%)"
        ),
        Connection(
            from_id="generator",
            to_id="replaced_tokens",
            label="replaces masked tokens"
        ),
        Connection(
            from_id="replaced_tokens",
            to_id="discriminator",
            label="classifies each token"
        ),
        Connection(
            from_id="discriminator",
            to_id="token_predictions",
            label="outputs original/replaced"
        )
    ]

    return MethodologyStructure(
        title="ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        main_contribution="Efficient pre-training by replacing token generation with a discriminative task (Replaced Token Detection)",
        components=components,
        connections=connections,
        key_differences="Unlike BERT's MLM (which predicts 15% of tokens), ELECTRA's discriminator analyzes all tokens, making it more computationally efficient while achieving better performance"
    )


def main():
    """Run the demonstration."""
    print("🔬 Paper Visualizer Demo")
    print("=" * 50)
    print()

    # Create sample ELECTRA structure
    structure = create_electra_example()

    print(f"📝 Paper: {structure.title}")
    print(f"🎯 Main Contribution: {structure.main_contribution}")
    print(f"🔧 Components: {len(structure.components)}")
    print(f"🔗 Connections: {len(structure.connections)}")
    print()

    # Generate Mermaid diagram
    print("🎨 Generated Mermaid Diagram:")
    print("-" * 30)
    diagram = generate_mermaid_diagram(structure)
    print(diagram)
    print()

    # Generate full markdown report (truncated for demo)
    print("📊 Generated Markdown Report:")
    print("-" * 30)
    report = generate_markdown_report(structure)
    # Show first part of report
    lines = report.split('\n')
    for i, line in enumerate(lines[:50]):  # Show first 50 lines
        print(line)
    if len(lines) > 50:
        print("... (truncated)")
    print()

    print("✨ Demo complete! To use with real papers:")
    print("   1. Set OPENAI_API_KEY environment variable")
    print("   2. Run: uv run paper-visualizer --input your_paper.pdf")


if __name__ == "__main__":
    main()