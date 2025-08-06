"""Simple example of using the Adela chess engine."""

import chess
import chess.svg
import time
from pathlib import Path

from adela.engine import AdelaEngine
from adela.core.board import BoardRepresentation


def play_simple_game() -> None:
    """Play a simple game with the engine against itself."""
    print("Initializing Adela chess engine...")
    engine = AdelaEngine(mcts_simulations=100)  # Lower simulations for faster moves
    
    # Reset the engine to the starting position
    engine.reset()
    
    # Play 10 moves
    for i in range(10):
        # Get the current board state
        board = engine.board.board
        print(f"\nMove {i+1}")
        print(board)
        
        # Get the best move
        start_time = time.time()
        move = engine.get_best_move()
        end_time = time.time()
        
        print(f"Best move: {move} (took {end_time - start_time:.2f} seconds)")
        
        # Get expert contributions
        contributions = engine.get_expert_contributions()
        print("Expert contributions:")
        for expert, weight in sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {expert}: {weight:.4f}")
        
        # Make the move
        engine.make_move(move)
    
    # Print the final position
    print("\nFinal position:")
    print(engine.board.board)


def play_against_engine(engine=None) -> None:
    """Play against the engine.
    
    Args:
        engine: Optional pre-initialized engine. If None, creates a new engine.
    """
    if engine is None:
        print("Initializing Adela chess engine...")
        engine = AdelaEngine(mcts_simulations=100)  # Lower simulations for faster moves
        
        # Reset the engine to the starting position
        engine.reset()
    
    while not engine.board.is_game_over():
        # Get the current board state
        board = engine.board.board
        print("\nCurrent position:")
        print(board)
        
        # Player's turn if white, engine's turn if black
        if board.turn == chess.WHITE:
            # Get player move
            valid_move = False
            while not valid_move:
                try:
                    move_str = input("Enter your move (e.g., 'e2e4'): ")
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        valid_move = True
                    else:
                        print("Illegal move!")
                except ValueError:
                    print("Invalid move format! Use format like 'e2e4'.")
            
            # Make the move
            engine.make_move(move)
        else:
            # Engine's turn
            print("Engine is thinking...")
            start_time = time.time()
            move = engine.get_best_move()
            end_time = time.time()
            
            print(f"Engine plays: {move} (took {end_time - start_time:.2f} seconds)")
            
            # Get expert contributions
            contributions = engine.get_expert_contributions()
            print("Expert contributions:")
            for expert, weight in sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {expert}: {weight:.4f}")
            
            # Make the move
            engine.make_move(move)
    
    # Print the result
    print("\nGame over!")
    print(f"Result: {engine.board.get_result()}")
    print(engine.board.board)


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)
    
    # Uncomment one of these to run the example
    play_simple_game()
    # play_against_engine()
