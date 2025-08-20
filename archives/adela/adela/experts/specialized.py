"""Specialized expert networks for different chess aspects."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from adela.experts.base import ConvolutionalExpert


class PhaseExpert(ConvolutionalExpert):
    """Expert specialized for a specific game phase."""

    def __init__(
        self, 
        phase: str,
        num_filters: int = 256, 
        num_blocks: int = 10
    ) -> None:
        """Initialize the phase expert.

        Args:
            phase: Game phase ("opening", "middlegame", or "endgame").
            num_filters: Number of filters in convolutional layers.
            num_blocks: Number of residual blocks.
        """
        super().__init__(f"{phase}_expert", num_filters, num_blocks)
        self.phase = phase
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get expert metadata.

        Returns:
            Dictionary with expert metadata.
        """
        metadata = super().get_metadata()
        metadata["phase"] = self.phase
        return metadata


class StyleExpert(ConvolutionalExpert):
    """Expert specialized for a specific playing style."""

    def __init__(
        self, 
        style: str,
        num_filters: int = 256, 
        num_blocks: int = 10
    ) -> None:
        """Initialize the style expert.

        Args:
            style: Playing style ("tactical", "positional", "attacking", or "defensive").
            num_filters: Number of filters in convolutional layers.
            num_blocks: Number of residual blocks.
        """
        super().__init__(f"{style}_expert", num_filters, num_blocks)
        self.style = style
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get expert metadata.

        Returns:
            Dictionary with expert metadata.
        """
        metadata = super().get_metadata()
        metadata["style"] = self.style
        return metadata


class AdaptationExpert(ConvolutionalExpert):
    """Expert specialized for adapting to specific opponent types."""

    def __init__(
        self, 
        opponent_type: str,
        num_filters: int = 256, 
        num_blocks: int = 10
    ) -> None:
        """Initialize the adaptation expert.

        Args:
            opponent_type: Opponent type ("anti_engine", "anti_human", or "counter_style").
            num_filters: Number of filters in convolutional layers.
            num_blocks: Number of residual blocks.
        """
        super().__init__(f"{opponent_type}_expert", num_filters, num_blocks)
        self.opponent_type = opponent_type
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get expert metadata.

        Returns:
            Dictionary with expert metadata.
        """
        metadata = super().get_metadata()
        metadata["opponent_type"] = self.opponent_type
        return metadata


def create_phase_experts(
    num_filters: int = 256, 
    num_blocks: int = 10
) -> Dict[str, PhaseExpert]:
    """Create a set of phase experts.

    Args:
        num_filters: Number of filters in convolutional layers.
        num_blocks: Number of residual blocks.

    Returns:
        Dictionary of phase experts.
    """
    phases = ["opening", "middlegame", "endgame"]
    return {
        phase: PhaseExpert(phase, num_filters, num_blocks)
        for phase in phases
    }


def create_style_experts(
    num_filters: int = 256, 
    num_blocks: int = 10
) -> Dict[str, StyleExpert]:
    """Create a set of style experts.

    Args:
        num_filters: Number of filters in convolutional layers.
        num_blocks: Number of residual blocks.

    Returns:
        Dictionary of style experts.
    """
    styles = ["tactical", "positional", "attacking", "defensive"]
    return {
        style: StyleExpert(style, num_filters, num_blocks)
        for style in styles
    }


def create_adaptation_experts(
    num_filters: int = 256, 
    num_blocks: int = 10
) -> Dict[str, AdaptationExpert]:
    """Create a set of adaptation experts.

    Args:
        num_filters: Number of filters in convolutional layers.
        num_blocks: Number of residual blocks.

    Returns:
        Dictionary of adaptation experts.
    """
    opponent_types = ["anti_engine", "anti_human", "counter_style"]
    return {
        opponent_type: AdaptationExpert(opponent_type, num_filters, num_blocks)
        for opponent_type in opponent_types
    }
