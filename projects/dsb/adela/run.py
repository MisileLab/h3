"""Main script for running the Adela chess engine."""

import argparse
import sys
from pathlib import Path

from adela.engine import AdelaEngine


def main():
    """Run the Adela chess engine."""
    parser = argparse.ArgumentParser(description="Adela Chess Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Play a game against the engine")
    play_parser.add_argument("--fen", help="FEN string to start from")
    play_parser.add_argument("--model", help="Path to model checkpoint")
    play_parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a position")
    analyze_parser.add_argument("--fen", required=True, help="FEN string to analyze")
    analyze_parser.add_argument("--model", help="Path to model checkpoint")
    analyze_parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    
    # Self-play command
    selfplay_parser = subparsers.add_parser("selfplay", help="Engine plays against itself")
    selfplay_parser.add_argument("--model", help="Path to model checkpoint")
    selfplay_parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    selfplay_parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    
    args = parser.parse_args()
    
    if args.command == "play":
        play_game(args)
    elif args.command == "analyze":
        analyze_position(args)
    elif args.command == "selfplay":
        self_play(args)
    else:
        parser.print_help()


def play_game(args):
    """Play a game against the engine."""
    print("Initializing Adela chess engine...")
    engine = AdelaEngine(
        model_path=args.model,
        mcts_simulations=args.simulations
    )
    
    if args.fen:
        engine.set_position(args.fen)
    else:
        engine.reset()
    
    from examples.simple_game import play_against_engine
    play_against_engine(engine)


def analyze_position(args):
    """Analyze a position."""
    print("Initializing Adela chess engine...")
    engine = AdelaEngine(
        model_path=args.model,
        mcts_simulations=args.simulations
    )
    
    engine.set_position(args.fen)
    
    # Get the best move
    move = engine.get_best_move()
    print(f"Best move: {move}")
    
    # Get expert contributions
    contributions = engine.get_expert_contributions()
    print("Expert contributions:")
    for expert, weight in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {expert}: {weight:.4f}")


def self_play(args):
    """Engine plays against itself."""
    print("Initializing Adela chess engine...")
    engine = AdelaEngine(
        model_path=args.model,
        mcts_simulations=args.simulations
    )
    
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    
    for i in range(args.games):
        print(f"\nGame {i+1}/{args.games}")
        result = engine.play_game()
        print(f"Result: {result}")
        results[result] = results.get(result, 0) + 1
    
    print("\nFinal results:")
    print(f"White wins: {results['1-0']}")
    print(f"Black wins: {results['0-1']}")
    print(f"Draws: {results['1/2-1/2']}")


if __name__ == "__main__":
    main()
