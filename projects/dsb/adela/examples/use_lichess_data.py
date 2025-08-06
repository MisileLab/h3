"""Example script for using the processed Lichess data."""

import os
import sys
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adela.data.pipeline import LichessDataPipeline


def analyze_games(games_path: Path) -> None:
    """Analyze processed chess games.

    Args:
        games_path: Path to the processed games Parquet file.
    """
    print(f"Analyzing games from {games_path}")
    
    # Load the data
    df = pl.read_parquet(games_path)
    
    # Print basic statistics
    print(f"Number of games: {len(df)}")
    print(f"Average white Elo: {df['white_elo'].mean():.1f}")
    print(f"Average black Elo: {df['black_elo'].mean():.1f}")
    print(f"Average number of moves: {df['num_moves'].mean():.1f}")
    
    # Results statistics
    results = df['result'].value_counts().sort(by="count", descending=True)
    print("\nResults:")
    for row in results.iter_rows():
        result, count = row
        print(f"  {result}: {count} ({count/len(df)*100:.1f}%)")
    
    # ECO code statistics
    eco_counts = df['eco'].value_counts().sort(by="count", descending=True).head(10)
    print("\nTop 10 ECO codes:")
    for row in eco_counts.iter_rows():
        eco, count = row
        print(f"  {eco}: {count} ({count/len(df)*100:.1f}%)")
    
    # Plot number of moves distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['num_moves'].to_numpy(), bins=50)
    plt.title('Distribution of Game Lengths')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.savefig('game_lengths.png')
    print("\nSaved game length distribution to game_lengths.png")


def analyze_puzzles(puzzles_path: Path) -> None:
    """Analyze processed chess puzzles.

    Args:
        puzzles_path: Path to the processed puzzles Parquet file.
    """
    print(f"Analyzing puzzles from {puzzles_path}")
    
    # Load the data
    df = pl.read_parquet(puzzles_path)
    
    # Print basic statistics
    print(f"Number of puzzles: {len(df)}")
    print(f"Average rating: {df['rating'].mean():.1f}")
    print(f"Average popularity: {df['popularity'].mean():.1f}")
    print(f"Average number of plays: {df['nb_plays'].mean():.1f}")
    
    # Theme statistics
    theme_counts = (
        df.with_columns(pl.col("themes").str.split(" ").alias("theme_list"))
        .explode("theme_list")
        .group_by("theme_list")
        .count()
        .sort("count", descending=True)
        .head(10)
    )
    print("\nTop 10 themes:")
    for row in theme_counts.iter_rows():
        theme, count = row
        print(f"  {theme}: {count} ({count/len(df)*100:.1f}%)")
    
    # Plot rating distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['rating'].to_numpy(), bins=50)
    plt.title('Distribution of Puzzle Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.savefig('puzzle_ratings.png')
    print("\nSaved puzzle rating distribution to puzzle_ratings.png")


def analyze_evaluations(evals_path: Path) -> None:
    """Analyze processed chess position evaluations.

    Args:
        evals_path: Path to the processed evaluations Parquet file.
    """
    print(f"Analyzing evaluations from {evals_path}")
    
    # Load the data
    df = pl.read_parquet(evals_path)
    
    # Print basic statistics
    print(f"Number of evaluations: {len(df)}")
    print(f"Average depth: {df['depth'].mean():.1f}")
    print(f"Average knodes: {df['knodes'].mean():.1f}")
    print(f"Percentage with mate: {df['has_mate'].mean()*100:.1f}%")
    
    # Evaluation value statistics
    eval_stats = df.select(
        pl.col("eval_value").mean().alias("mean"),
        pl.col("eval_value").median().alias("median"),
        pl.col("eval_value").std().alias("std"),
        pl.col("eval_value").min().alias("min"),
        pl.col("eval_value").max().alias("max"),
    )
    print("\nEvaluation statistics:")
    for col in ["mean", "median", "std", "min", "max"]:
        print(f"  {col}: {eval_stats[0][col]:.1f}")
    
    # Plot evaluation distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['eval_value'].to_numpy(), bins=50, range=(-500, 500))
    plt.title('Distribution of Evaluation Values')
    plt.xlabel('Evaluation (centipawns)')
    plt.ylabel('Frequency')
    plt.savefig('eval_distribution.png')
    print("\nSaved evaluation distribution to eval_distribution.png")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    # Check if data files exist
    data_dir = Path("./data")
    
    # Find the most recent files
    games_files = list(data_dir.glob("games/lichess_games_*.parquet"))
    puzzle_files = list(data_dir.glob("puzzles/lichess_puzzles_*.parquet"))
    eval_files = list(data_dir.glob("evaluations/lichess_evaluations_*.parquet"))
    
    # If no files exist, run the pipeline to download and process data
    if not (games_files or puzzle_files or eval_files):
        print("No processed data files found. Running the pipeline to download and process data...")
        
        # Create pipeline
        pipeline = LichessDataPipeline(data_dir=data_dir)
        
        # Run pipeline with limited data for example purposes
        results = pipeline.run_full_pipeline(
            max_games=10000,
            max_puzzles=10000,
            max_evaluations=10000
        )
        
        # Update file lists
        games_files = [results.get("games")] if "games" in results else []
        puzzle_files = [results.get("puzzles")] if "puzzles" in results else []
        eval_files = [results.get("evaluations")] if "evaluations" in results else []
    
    # Analyze the most recent files
    if games_files:
        games_path = max(games_files, key=lambda p: p.stat().st_mtime)
        analyze_games(games_path)
    
    if puzzle_files:
        puzzles_path = max(puzzle_files, key=lambda p: p.stat().st_mtime)
        analyze_puzzles(puzzles_path)
    
    if eval_files:
        evals_path = max(eval_files, key=lambda p: p.stat().st_mtime)
        analyze_evaluations(evals_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
