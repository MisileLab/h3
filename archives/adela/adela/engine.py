"""Main engine class for the chess AI."""

import os
from typing import Callable, final

import bulletchess as chess
import torch

from adela.core.board import BoardRepresentation
from adela.mcts.search import MCTS
from adela.opponent.analyzer import OpponentAnalyzer
from adela.training.pipeline import create_mixture_of_experts

@final
class AdelaEngine:
  """Main engine class for the chess AI."""

  def __init__(
    self, 
    model_path: str | None = None,
    mcts_simulations: int = 800,
    temperature: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # pyright: ignore[reportCallInDefaultInitializer]
  ) -> None:
    """Initialize the engine.

    Args:
      model_path: Path to the model checkpoint. If None, creates a new model.
      mcts_simulations: Number of MCTS simulations per move.
      temperature: Temperature for move selection.
      device: Device to run the model on.
    """
    self.device = device
    self.mcts_simulations = mcts_simulations
    self.temperature = temperature
    
    # Create or load model
    self.model = create_mixture_of_experts(device=device)
    if model_path and os.path.exists(model_path):
      _ = self.model.load_state_dict(torch.load(model_path, map_location=device)) # pyright: ignore[reportAny]
    
    # Create MCTS
    self.mcts = MCTS(
      model=self.model,
      num_simulations=mcts_simulations,
      temperature=temperature,
      device=device
    )
    
    # Create opponent analyzer
    self.opponent_analyzer = OpponentAnalyzer()
    
    # Current game state
    self.board = BoardRepresentation()
    self.move_history: list[chess.Move] = []
  
  def reset(self) -> None:
    """Reset the engine to the starting position."""
    self.board = BoardRepresentation()
    self.move_history.clear()
  
  def set_position(self, fen: str) -> None:
    """Set the position.

    Args:
      fen: FEN string.
    """
    self.board = BoardRepresentation(fen)
    self.move_history.clear()
  
  def make_move(self, move: chess.Move) -> None:
    """Make a move on the board.

    Args:
      move: Chess move.
    """
    self.board.make_move(move)
    self.move_history.append(move)
  
  def get_best_move(
    self,
    fen: str | None = None,
    temperature: float | None = None,
  ) -> chess.Move:
    """Get the best move for a position.

    Args:
      fen: Optional FEN string. If None, uses the current position.
      time_limit: Optional time limit in seconds.
      temperature: Temperature for move selection. If None, uses self.temperature.

    Returns:
      Best move.
    """
    # Set the position if provided
    if fen:
      self.set_position(fen)
    
    # Get the best move
    move = self.mcts.get_best_move(
      self.board,
      temperature=temperature if temperature is not None else self.temperature
    )
    
    return move
  
  def play_game(
    self,
    opponent_function: Callable[[BoardRepresentation], chess.Move] | None = None,
    max_moves: int = 200,
  ) -> str:
    """Play a game against an opponent.

    Args:
      opponent_function: Function that takes a board and returns a move.
        If None, the engine plays against itself.
      max_moves: Maximum number of moves before declaring a draw.

    Returns:
      Game result.
    """
    # Reset the engine
    self.reset()
    
    # Play the game
    move_count = 0
    while not self.board.is_game_over() and move_count < max_moves:
      # Get the current player
      is_white = self.board.board.turn == chess.WHITE
      
      # Get the move
      move = self.get_best_move() if is_white or opponent_function is None else opponent_function(self.board)
      
      # Make the move
      self.make_move(move)
      move_count += 1
    
    # Return the result
    return self.board.get_result()
  
  def get_expert_contributions(self) -> dict[str, float]:
    """Get the contribution of each expert for the current position.

    Returns:
      Dictionary mapping expert names to their weights.
    """
    # Get board tensor and additional features
    board_tensor = torch.tensor(
      self.board.get_board_tensor(),
      dtype=torch.float32,
      device=self.device
    ).unsqueeze(0)
    
    additional_features = torch.tensor(
      self.board.get_additional_features(),
      dtype=torch.float32,
      device=self.device
    ).unsqueeze(0)
    
    # Get expert contributions
    return self.model.get_expert_contributions(board_tensor, additional_features)
  
  def set_opponent_profile(self) -> None:
    """Set the opponent profile for adaptation.

    Args:
      profile: Opponent profile.
    """
    # This is a placeholder for actual opponent adaptation
    # In a real implementation, you would use the opponent profile
    # to adjust the model's behavior
    pass
  
  def save_model(self, path: str) -> None:
    """Save the model.

    Args:
      path: Path to save the model.
    """
    torch.save(self.model.state_dict(), path)
  
  def load_model(self, path: str) -> None:
    """Load the model.

    Args:
      path: Path to load the model from.
    """
    _ = self.model.load_state_dict(torch.load(path, map_location=self.device)) # pyright: ignore[reportAny]


# Example usage
if __name__ == "__main__":
  # Create engine
  engine = AdelaEngine()
  
  # Get the best move for the starting position
  best_move = engine.get_best_move()
  print(f"Best move: {best_move}")
  
  # Play a quick game against itself (for smoke test)
  result = engine.play_game()
  print(f"Game result: {result}")
  
  # Get expert contributions
  contributions = engine.get_expert_contributions()
  for expert, weight in contributions.items():
    print(f"{expert}: {weight:.4f}")
