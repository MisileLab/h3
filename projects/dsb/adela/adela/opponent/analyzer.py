"""Opponent analysis module for adapting to specific opponents."""

from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from adela.core.chess_shim import chess, require_pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adela.core.board import BoardRepresentation


class OpponentProfile:
    """Profile of an opponent's playing style."""

    def __init__(self, name: str) -> None:
        """Initialize the opponent profile.

        Args:
            name: Opponent name or identifier.
        """
        self.name = name
        
        # Opening repertoire
        self.white_openings: Dict[str, int] = defaultdict(int)
        self.black_openings: Dict[str, int] = defaultdict(int)
        
        # Move timing statistics
        self.avg_move_time: float = 0.0
        self.move_time_variance: float = 0.0
        self.time_pressure_errors: int = 0
        
        # Style metrics (0-1 scale)
        self.tactical_score: float = 0.5  # 0 = positional, 1 = tactical
        self.risk_score: float = 0.5  # 0 = conservative, 1 = aggressive
        self.complexity_preference: float = 0.5  # 0 = simple, 1 = complex
        
        # Phase performance
        self.opening_performance: float = 0.0
        self.middlegame_performance: float = 0.0
        self.endgame_performance: float = 0.0
        
        # Error patterns
        self.error_rate: float = 0.0
        self.blunder_rate: float = 0.0
        self.error_phase_distribution: Dict[str, float] = {
            "opening": 0.0,
            "middlegame": 0.0,
            "endgame": 0.0
        }
        
        # Game history
        self.games_analyzed: int = 0
        self.rating: Optional[int] = None
        
    def to_feature_vector(self) -> np.ndarray:
        """Convert the profile to a feature vector for the neural network.

        Returns:
            Feature vector representing the opponent profile.
        """
        features = [
            self.tactical_score,
            self.risk_score,
            self.complexity_preference,
            self.opening_performance,
            self.middlegame_performance,
            self.endgame_performance,
            self.error_rate,
            self.blunder_rate,
            self.error_phase_distribution["opening"],
            self.error_phase_distribution["middlegame"],
            self.error_phase_distribution["endgame"],
            self.avg_move_time,
            self.move_time_variance,
            float(self.time_pressure_errors) / max(1, self.games_analyzed),
            float(self.rating or 1500) / 3000  # Normalize rating
        ]
        
        return np.array(features, dtype=np.float32)


class OpponentAnalyzer:
    """Analyzer for opponent's playing style and weaknesses."""

    def __init__(self) -> None:
        """Initialize the opponent analyzer."""
        self.profiles: Dict[str, OpponentProfile] = {}
        
    def create_profile(self, name: str) -> OpponentProfile:
        """Create a new opponent profile.

        Args:
            name: Opponent name or identifier.

        Returns:
            New opponent profile.
        """
        profile = OpponentProfile(name)
        self.profiles[name] = profile
        return profile
    
    def get_profile(self, name: str) -> OpponentProfile:
        """Get an opponent profile.

        Args:
            name: Opponent name or identifier.

        Returns:
            Opponent profile.
        """
        if name not in self.profiles:
            return self.create_profile(name)
        return self.profiles[name]
    
    def analyze_game(
        self, 
        pgn: str, 
        opponent_name: str,
        opponent_color: chess.Color,
        engine_eval: Optional[List[float]] = None
    ) -> None:
        """Analyze a game to update an opponent's profile.

        Args:
            pgn: PGN string of the game.
            opponent_name: Opponent name or identifier.
            opponent_color: Color played by the opponent.
            engine_eval: Optional list of engine evaluations for each position.
        """
        # Get or create opponent profile
        profile = self.get_profile(opponent_name)
        
        # Parse PGN
        _pgn = require_pgn()
        game = _pgn.read_game(pgn)
        if game is None:
            return
        
        # Update games analyzed
        profile.games_analyzed += 1
        
        # Extract metadata
        if "WhiteElo" in game.headers and opponent_color == chess.WHITE:
            profile.rating = int(game.headers["WhiteElo"])
        elif "BlackElo" in game.headers and opponent_color == chess.BLACK:
            profile.rating = int(game.headers["BlackElo"])
        
        # Extract opening information
        if "ECO" in game.headers:
            eco_code = game.headers["ECO"]
            if opponent_color == chess.WHITE:
                profile.white_openings[eco_code] += 1
            else:
                profile.black_openings[eco_code] += 1
        
        # Analyze moves
        board = game.board()
        moves = list(game.mainline_moves())
        move_times = []
        error_count = 0
        blunder_count = 0
        
        # Phase statistics
        phase_move_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
        phase_error_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
        
        for i, move in enumerate(moves):
            # Skip moves not made by the opponent
            if board.turn != opponent_color:
                board.push(move)
                continue
            
            # Get board representation
            board_rep = BoardRepresentation(board.fen())
            phase = self._get_phase_name(board_rep.get_phase())
            phase_move_counts[phase] += 1
            
            # Check for errors if engine evaluations are available
            if engine_eval and i + 1 < len(engine_eval):
                eval_before = engine_eval[i]
                eval_after = engine_eval[i + 1]
                
                # Convert to opponent's perspective
                if opponent_color == chess.BLACK:
                    eval_before = -eval_before
                    eval_after = -eval_after
                
                # Check for errors and blunders
                eval_diff = eval_after - eval_before
                if eval_diff < -0.5:
                    error_count += 1
                    phase_error_counts[phase] += 1
                if eval_diff < -2.0:
                    blunder_count += 1
            
            # Extract move timing information if available
            if hasattr(move, "clock"):
                move_times.append(move.clock())
            
            # Make the move
            board.push(move)
        
        # Update profile with move statistics
        if move_times:
            profile.avg_move_time = np.mean(move_times)
            profile.move_time_variance = np.var(move_times)
            
            # Detect time pressure errors
            for i, time in enumerate(move_times):
                if time < 10 and i + 1 < len(engine_eval) and engine_eval[i + 1] - engine_eval[i] < -1.0:
                    profile.time_pressure_errors += 1
        
        # Update error statistics
        total_moves = sum(count for count in phase_move_counts.values())
        if total_moves > 0:
            profile.error_rate = error_count / total_moves
            profile.blunder_rate = blunder_count / total_moves
            
            # Update phase error distribution
            for phase in phase_move_counts:
                if phase_move_counts[phase] > 0:
                    profile.error_phase_distribution[phase] = phase_error_counts[phase] / phase_move_counts[phase]
        
        # Update phase performance based on error rates
        profile.opening_performance = 1.0 - profile.error_phase_distribution["opening"]
        profile.middlegame_performance = 1.0 - profile.error_phase_distribution["middlegame"]
        profile.endgame_performance = 1.0 - profile.error_phase_distribution["endgame"]
    
    def _get_phase_name(self, phase_value: float) -> str:
        """Convert a phase value to a phase name.

        Args:
            phase_value: Phase value between 0 and 1.

        Returns:
            Phase name: "opening", "middlegame", or "endgame".
        """
        if phase_value < 0.3:
            return "opening"
        elif phase_value < 0.7:
            return "middlegame"
        else:
            return "endgame"


class WeaknessDetector(nn.Module):
    """Neural network for detecting opponent weaknesses."""

    def __init__(self, input_size: int = 15, hidden_size: int = 128) -> None:
        """Initialize the weakness detector.

        Args:
            input_size: Size of the input feature vector.
            hidden_size: Size of hidden layers.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Output different weakness scores
        self.tactical_weakness = nn.Linear(hidden_size, 1)
        self.positional_weakness = nn.Linear(hidden_size, 1)
        self.time_pressure_weakness = nn.Linear(hidden_size, 1)
        self.opening_weakness = nn.Linear(hidden_size, 1)
        self.middlegame_weakness = nn.Linear(hidden_size, 1)
        self.endgame_weakness = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the weakness detector.

        Args:
            x: Input tensor with opponent profile features.

        Returns:
            Dictionary mapping weakness types to scores.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return {
            "tactical": torch.sigmoid(self.tactical_weakness(x)),
            "positional": torch.sigmoid(self.positional_weakness(x)),
            "time_pressure": torch.sigmoid(self.time_pressure_weakness(x)),
            "opening": torch.sigmoid(self.opening_weakness(x)),
            "middlegame": torch.sigmoid(self.middlegame_weakness(x)),
            "endgame": torch.sigmoid(self.endgame_weakness(x))
        }
    
    def detect_weaknesses(
        self, 
        profile: OpponentProfile
    ) -> Dict[str, float]:
        """Detect weaknesses in an opponent's profile.

        Args:
            profile: Opponent profile.

        Returns:
            Dictionary mapping weakness types to scores.
        """
        # Convert profile to feature vector
        features = profile.to_feature_vector()
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Run through the model
        with torch.no_grad():
            weakness_scores = self.forward(features_tensor)
        
        # Convert to dictionary of floats
        return {
            key: value.item()
            for key, value in weakness_scores.items()
        }
