"""Evaluation metrics for the chess AI."""

from typing import Dict, List
from collections import defaultdict

from adela.core.chess_shim import chess
import numpy as np
import torch

from adela.core.board import BoardRepresentation
from adela.gating.system import MixtureOfExperts
from adela.mcts.search import MCTS


class EvaluationMetrics:
    """Metrics for evaluating the chess AI."""

    @staticmethod
    def calculate_move_accuracy(
        predicted_moves: List[object],
        target_moves: List[object]
    ) -> float:
        """Calculate move accuracy.

        Args:
            predicted_moves: List of predicted moves.
            target_moves: List of target moves.

        Returns:
            Move accuracy.
        """
        if not predicted_moves or not target_moves:
            return 0.0
        
        correct = sum(1 for pred, target in zip(predicted_moves, target_moves) if pred == target)
        return correct / len(predicted_moves)
    
    @staticmethod
    def calculate_top_k_accuracy(
        predicted_moves: List[List[object]],
        target_moves: List[object],
        k: int = 3
    ) -> float:
        """Calculate top-k move accuracy.

        Args:
            predicted_moves: List of lists of predicted moves (top-k for each position).
            target_moves: List of target moves.
            k: Number of top moves to consider.

        Returns:
            Top-k move accuracy.
        """
        if not predicted_moves or not target_moves:
            return 0.0
        
        correct = 0
        for pred_moves, target in zip(predicted_moves, target_moves):
            if target in pred_moves[:k]:
                correct += 1
        
        return correct / len(predicted_moves)
    
    @staticmethod
    def calculate_value_mse(
        predicted_values: List[float],
        target_values: List[float]
    ) -> float:
        """Calculate mean squared error for position values.

        Args:
            predicted_values: List of predicted values.
            target_values: List of target values.

        Returns:
            Mean squared error.
        """
        if not predicted_values or not target_values:
            return 0.0
        
        return float(np.mean([(pred - target) ** 2 for pred, target in zip(predicted_values, target_values)]))
    
    @staticmethod
    def calculate_elo_gain(
        wins: int,
        losses: int,
        draws: int
    ) -> float:
        """Calculate Elo gain based on game results.

        Args:
            wins: Number of wins.
            losses: Number of losses.
            draws: Number of draws.

        Returns:
            Elo gain.
        """
        total_games = wins + losses + draws
        if total_games == 0:
            return 0.0
        
        score = wins + 0.5 * draws
        expected_score = 0.5 * total_games
        
        # Using K-factor of 32
        return 32 * (score - expected_score) / total_games
    
    @staticmethod
    def calculate_expert_usage(
        expert_weights: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate average usage of each expert.

        Args:
            expert_weights: List of dictionaries mapping expert names to weights.

        Returns:
            Dictionary mapping expert names to average usage.
        """
        if not expert_weights:
            return {}
        
        # Initialize expert usage
        expert_usage = defaultdict(float)
        
        # Sum weights for each expert
        for weights in expert_weights:
            for expert, weight in weights.items():
                expert_usage[expert] += weight
        
        # Calculate average
        for expert in expert_usage:
            expert_usage[expert] /= len(expert_weights)
        
        return {k: float(v) for k, v in expert_usage.items()}


class EngineMatch:
    """Match between two chess engines."""

    def __init__(
        self, 
        engine1: MCTS,
        engine2: MCTS,
        num_games: int = 100,
        time_per_move: float = 1.0
    ) -> None:
        """Initialize the match.

        Args:
            engine1: First engine.
            engine2: Second engine.
            num_games: Number of games to play.
            time_per_move: Time per move in seconds.
        """
        self.engine1 = engine1
        self.engine2 = engine2
        self.num_games = num_games
        self.time_per_move = time_per_move
        
        # Results
        self.wins1 = 0
        self.wins2 = 0
        self.draws = 0
        
    def play_match(self) -> Dict[str, Any]:
        """Play the match.

        Returns:
            Dictionary with match results.
        """
        for i in range(self.num_games):
            # Alternate colors
            if i % 2 == 0:
                white_engine = self.engine1
                black_engine = self.engine2
            else:
                white_engine = self.engine2
                black_engine = self.engine1
            
            # Play the game
            result = self._play_game(white_engine, black_engine)
            
            # Update results
            if result == "1-0":
                if i % 2 == 0:
                    self.wins1 += 1
                else:
                    self.wins2 += 1
            elif result == "0-1":
                if i % 2 == 0:
                    self.wins2 += 1
                else:
                    self.wins1 += 1
            else:
                self.draws += 1
        
        # Calculate Elo gain
        elo_gain = EvaluationMetrics.calculate_elo_gain(self.wins1, self.wins2, self.draws)
        
        # Return results
        return {
            "wins1": self.wins1,
            "wins2": self.wins2,
            "draws": self.draws,
            "elo_gain": elo_gain,
            "win_rate": (self.wins1 + 0.5 * self.draws) / self.num_games
        }
    
    def _play_game(
        self, 
        white_engine: MCTS,
        black_engine: MCTS
    ) -> str:
        """Play a single game.

        Args:
            white_engine: Engine playing white.
            black_engine: Engine playing black.

        Returns:
            Game result: "1-0", "0-1", or "1/2-1/2".
        """
        # Initialize the board
        board = BoardRepresentation()
        
        # Play the game
        while not board.is_game_over():
            # Select the engine based on the side to move
            engine = white_engine if board.board.turn == chess.WHITE else black_engine
            
            # Get the best move
            move = engine.get_best_move(board, temperature=0)
            
            # Make the move
            board.make_move(move)
        
        # Return the result
        return board.get_result()


class ModelEvaluator:
    """Evaluator for chess models."""

    def __init__(
        self, 
        model: MixtureOfExperts,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Neural network model.
            device: Device to evaluate on.
        """
        self.model = model
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Create MCTS
        self.mcts = MCTS(model=model)
        
    def evaluate_position_batch(
        self, 
        positions: List[str],
        target_moves: List[chess.Move],
        target_values: List[float]
    ) -> Dict[str, float]:
        """Evaluate a batch of positions.

        Args:
            positions: List of FEN strings.
            target_moves: List of target moves.
            target_values: List of target values.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        
        predicted_moves = []
        predicted_values = []
        expert_weights_list: List[Dict[str, float]] = []
        
        with torch.no_grad():
            for fen in positions:
                # Create board representation
                board = BoardRepresentation(fen)
                
                # Get board tensor and additional features
                board_tensor = torch.tensor(
                    board.get_board_tensor(),
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)
                
                additional_features = torch.tensor(
                    board.get_additional_features(),
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)
                
                # Forward pass
                policy_logits, value, expert_weights = self.model(board_tensor, additional_features)
                
                # Get the best move
                policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
                legal_moves = board.get_legal_moves()
                
                # Sort moves by policy probability
                move_probs = [(move, policy[board.get_move_index(move)]) for move in legal_moves]
                move_probs.sort(key=lambda x: x[1], reverse=True)
                
                # Get the top moves
                top_moves = [move for move, _ in move_probs[:3]]
                
                # Add predictions
                predicted_moves.append(top_moves[0])
                predicted_values.append(value.item())
                
                # Get expert weights
                expert_weights_dict: Dict[str, float] = {
                    name: float(weight.item())
                    for name, weight in zip(self.model.expert_names, expert_weights[0])
                }
                expert_weights_list.append(expert_weights_dict)
        
        # Calculate metrics
        metrics: Dict[str, float] = {
            "move_accuracy": EvaluationMetrics.calculate_move_accuracy(predicted_moves, target_moves),
            "top_3_accuracy": EvaluationMetrics.calculate_top_k_accuracy([top_moves for _, top_moves in zip(positions, predicted_moves)], target_moves, k=3),
            "value_mse": EvaluationMetrics.calculate_value_mse(predicted_values, target_values),
        }
        # Return separate mapping; caller can include expert usage alongside metrics if needed
        _ = EvaluationMetrics.calculate_expert_usage(expert_weights_list)
        return metrics
    
    def evaluate_against_stockfish(
        self, 
        stockfish_path: str,
        num_games: int = 10,
        time_per_move: float = 1.0
    ) -> Dict[str, Any]:
        """Evaluate against Stockfish.

        Args:
            stockfish_path: Path to Stockfish executable.
            num_games: Number of games to play.
            time_per_move: Time per move in seconds.

        Returns:
            Dictionary with evaluation metrics.
        """
        # This is a placeholder for actual Stockfish integration
        # In a real implementation, you would use the chess.engine module
        # to communicate with Stockfish
        
        # For now, we'll just return dummy results
        return {
            "wins": 0,
            "losses": 10,
            "draws": 0,
            "elo_gain": -200,
            "win_rate": 0.0
        }
