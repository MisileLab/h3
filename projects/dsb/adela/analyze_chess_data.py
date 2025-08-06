#!/usr/bin/env python
"""Script to analyze processed chess data in Parquet format."""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    Parsed arguments.
  """
  parser = argparse.ArgumentParser(description="Analyze processed chess data in Parquet format")
  
  parser.add_argument(
    "parquet_file",
    type=str,
    help="Path to the Parquet file to analyze"
  )
  
  parser.add_argument(
    "--output-dir",
    type=str,
    default="./analysis",
    help="Directory to save analysis results"
  )
  
  parser.add_argument(
    "--sample-size",
    type=int,
    default=None,
    help="Number of games to sample for analysis (None for all)"
  )
  
  parser.add_argument(
    "--min-elo",
    type=int,
    default=1000,
    help="Minimum Elo rating for games to include"
  )
  
  parser.add_argument(
    "--no-plots",
    action="store_true",
    help="Don't generate plots"
  )
  
  return parser.parse_args()


def load_data(parquet_path: Path, sample_size: Optional[int] = None, min_elo: int = 0) -> pl.DataFrame:
  """Load and filter data from Parquet file.

  Args:
    parquet_path: Path to the Parquet file.
    sample_size: Number of games to sample. If None, uses all games.
    min_elo: Minimum Elo rating for games to include.

  Returns:
    Filtered DataFrame.
  """
  logger.info(f"Loading data from {parquet_path}")
  
  # Load the data
  df = pl.read_parquet(parquet_path)
  
  # Apply Elo filter
  if min_elo > 0:
    df = df.filter((pl.col("white_elo") >= min_elo) & (pl.col("black_elo") >= min_elo))
    logger.info(f"Filtered to {len(df)} games with minimum Elo {min_elo}")
  
  # Sample if requested
  if sample_size is not None and sample_size < len(df):
    df = df.sample(sample_size, seed=42)
    logger.info(f"Sampled {sample_size} games")
  
  return df


def analyze_basic_stats(df: pl.DataFrame) -> Dict[str, Any]:
  """Analyze basic statistics.

  Args:
    df: DataFrame with chess games.

  Returns:
    Dictionary with basic statistics.
  """
  logger.info("Analyzing basic statistics")
  
  # Game results
  results = df.group_by("result").agg(
    pl.count().alias("count"),
    (pl.count() / len(df) * 100).alias("percentage")
  ).sort("count", descending=True)
  
  # Elo statistics
  elo_stats = {
    "white_elo_avg": df["white_elo"].mean(),
    "white_elo_min": df["white_elo"].min(),
    "white_elo_max": df["white_elo"].max(),
    "black_elo_avg": df["black_elo"].mean(),
    "black_elo_min": df["black_elo"].min(),
    "black_elo_max": df["black_elo"].max(),
  }
  
  # Move statistics
  move_stats = {
    "avg_moves": df["num_moves"].mean(),
    "min_moves": df["num_moves"].min(),
    "max_moves": df["num_moves"].max(),
  }
  
  # Time control statistics
  time_controls = df.group_by("time_control").agg(
    pl.count().alias("count"),
    (pl.count() / len(df) * 100).alias("percentage")
  ).sort("count", descending=True).head(10)
  
  # Opening statistics
  eco_counts = df.group_by("eco").agg(
    pl.count().alias("count"),
    (pl.count() / len(df) * 100).alias("percentage")
  ).sort("count", descending=True).head(10)
  
  opening_counts = df.group_by("opening").agg(
    pl.count().alias("count"),
    (pl.count() / len(df) * 100).alias("percentage")
  ).sort("count", descending=True).head(10)
  
  # Termination statistics
  termination_counts = df.group_by("termination").agg(
    pl.count().alias("count"),
    (pl.count() / len(df) * 100).alias("percentage")
  ).sort("count", descending=True)
  
  return {
    "total_games": len(df),
    "results": results.to_dicts(),
    "elo_stats": elo_stats,
    "move_stats": move_stats,
    "time_controls": time_controls.to_dicts(),
    "eco_counts": eco_counts.to_dicts(),
    "opening_counts": opening_counts.to_dicts(),
    "termination_counts": termination_counts.to_dicts(),
  }


def analyze_time_usage(df: pl.DataFrame) -> Dict[str, Any]:
  """Analyze time usage statistics.

  Args:
    df: DataFrame with chess games.

  Returns:
    Dictionary with time usage statistics.
  """
  logger.info("Analyzing time usage statistics")
  
  # Filter games with clock times
  df_with_times = df.filter(pl.col("has_clock_times") == True)
  
  if len(df_with_times) == 0:
    logger.warning("No games with clock times found")
    return {"games_with_clock_times": 0}
  
  # Time usage statistics
  time_stats = {
    "games_with_clock_times": len(df_with_times),
    "avg_time_per_move": df_with_times["avg_time_per_move"].mean(),
    "min_time_per_move": df_with_times["min_time_per_move"].min(),
    "max_time_per_move": df_with_times["max_time_per_move"].max(),
  }
  
  # Time usage by Elo rating
  df_with_times = df_with_times.with_columns(
    ((pl.col("white_elo") + pl.col("black_elo")) / 2).alias("avg_elo")
  )
  
  df_with_times = df_with_times.with_columns(
    pl.when(pl.col("avg_elo") < 1200).then("< 1200")
    .when(pl.col("avg_elo") < 1400).then("1200-1399")
    .when(pl.col("avg_elo") < 1600).then("1400-1599")
    .when(pl.col("avg_elo") < 1800).then("1600-1799")
    .when(pl.col("avg_elo") < 2000).then("1800-1999")
    .when(pl.col("avg_elo") < 2200).then("2000-2199")
    .when(pl.col("avg_elo") < 2400).then("2200-2399")
    .otherwise("≥ 2400").alias("elo_group")
  )
  
  time_by_elo = df_with_times.group_by("elo_group").agg(
    pl.col("avg_time_per_move").mean().alias("avg_time"),
    pl.count().alias("count")
  ).sort("elo_group")
  
  return {
    **time_stats,
    "time_by_elo": time_by_elo.to_dicts(),
  }


def generate_plots(stats: Dict[str, Any], output_dir: Path) -> None:
  """Generate plots from analysis results.

  Args:
    stats: Dictionary with analysis results.
    output_dir: Directory to save plots.
  """
  logger.info("Generating plots")
  
  # Create output directory
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Set Seaborn style
  sns.set(style="whitegrid")
  
  # Plot game results
  if "results" in stats:
    plt.figure(figsize=(10, 6))
    results_df = pl.DataFrame(stats["results"])
    sns.barplot(x="result", y="percentage", data=results_df.to_pandas())
    plt.title("Game Results")
    plt.xlabel("Result")
    plt.ylabel("Percentage")
    plt.savefig(output_dir / "game_results.png")
    plt.close()
  
  # Plot move distribution
  if "move_stats" in stats:
    plt.figure(figsize=(10, 6))
    move_counts = pl.DataFrame(stats.get("df", [])).select("num_moves").to_pandas()
    if not move_counts.empty:
      sns.histplot(data=move_counts, x="num_moves", bins=30)
      plt.title("Distribution of Game Lengths")
      plt.xlabel("Number of Moves")
      plt.ylabel("Frequency")
      plt.savefig(output_dir / "move_distribution.png")
    plt.close()
  
  # Plot Elo distribution
  if "elo_stats" in stats:
    plt.figure(figsize=(10, 6))
    elo_df = pl.DataFrame(stats.get("df", [])).select(["white_elo", "black_elo"]).to_pandas()
    if not elo_df.empty:
      sns.histplot(data=elo_df, x="white_elo", label="White", alpha=0.5, bins=30)
      sns.histplot(data=elo_df, x="black_elo", label="Black", alpha=0.5, bins=30)
      plt.title("Distribution of Elo Ratings")
      plt.xlabel("Elo Rating")
      plt.ylabel("Frequency")
      plt.legend()
      plt.savefig(output_dir / "elo_distribution.png")
    plt.close()
  
  # Plot time usage by Elo
  if "time_by_elo" in stats:
    plt.figure(figsize=(12, 6))
    time_by_elo_df = pl.DataFrame(stats["time_by_elo"]).to_pandas()
    sns.barplot(x="elo_group", y="avg_time", data=time_by_elo_df)
    plt.title("Average Time per Move by Elo Rating")
    plt.xlabel("Elo Rating Group")
    plt.ylabel("Average Time per Move (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "time_by_elo.png")
    plt.close()
  
  # Plot top openings
  if "eco_counts" in stats:
    plt.figure(figsize=(12, 6))
    eco_df = pl.DataFrame(stats["eco_counts"]).head(10).to_pandas()
    sns.barplot(x="eco", y="percentage", data=eco_df)
    plt.title("Top 10 ECO Codes")
    plt.xlabel("ECO Code")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "top_eco_codes.png")
    plt.close()
  
  logger.info(f"Plots saved to {output_dir}")


def main() -> int:
  """Main entry point.

  Returns:
    Exit code.
  """
  args = parse_args()
  
  # Check if the Parquet file exists
  parquet_path = Path(args.parquet_file)
  if not parquet_path.exists():
    logger.error(f"Parquet file not found: {parquet_path}")
    return 1
  
  # Create output directory
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  try:
    # Load data
    df = load_data(parquet_path, args.sample_size, args.min_elo)
    
    # Analyze basic statistics
    basic_stats = analyze_basic_stats(df)
    
    # Analyze time usage
    time_stats = analyze_time_usage(df)
    
    # Combine statistics
    stats = {
      **basic_stats,
      **time_stats,
      "df": df.sample(min(1000, len(df))).to_dicts()  # Sample for plotting
    }
    
    # Save statistics as CSV
    results_df = pl.DataFrame(stats["results"])
    results_df.write_csv(output_dir / "game_results.csv")
    
    eco_df = pl.DataFrame(stats["eco_counts"])
    eco_df.write_csv(output_dir / "top_eco_codes.csv")
    
    opening_df = pl.DataFrame(stats["opening_counts"])
    opening_df.write_csv(output_dir / "top_openings.csv")
    
    termination_df = pl.DataFrame(stats["termination_counts"])
    termination_df.write_csv(output_dir / "termination_types.csv")
    
    if "time_by_elo" in stats:
      time_by_elo_df = pl.DataFrame(stats["time_by_elo"])
      time_by_elo_df.write_csv(output_dir / "time_by_elo.csv")
    
    # Generate plots
    if not args.no_plots:
      generate_plots(stats, output_dir)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"Total games analyzed: {stats['total_games']}")
    print("\nGame Results:")
    for result in stats["results"]:
      print(f"  {result['result']}: {result['count']} ({result['percentage']:.1f}%)")
    
    print("\nElo Statistics:")
    print(f"  Average White Elo: {stats['elo_stats']['white_elo_avg']:.1f}")
    print(f"  Average Black Elo: {stats['elo_stats']['black_elo_avg']:.1f}")
    
    print("\nMove Statistics:")
    print(f"  Average moves per game: {stats['move_stats']['avg_moves']:.1f}")
    print(f"  Min moves: {stats['move_stats']['min_moves']}")
    print(f"  Max moves: {stats['move_stats']['max_moves']}")
    
    print("\nTop 5 ECO Codes:")
    for eco in stats["eco_counts"][:5]:
      print(f"  {eco['eco']}: {eco['count']} ({eco['percentage']:.1f}%)")
    
    print("\nTop 5 Openings:")
    for opening in stats["opening_counts"][:5]:
      print(f"  {opening['opening']}: {opening['count']} ({opening['percentage']:.1f}%)")
    
    print("\nTop 5 Termination Types:")
    for term in stats["termination_counts"][:5]:
      print(f"  {term['termination']}: {term['count']} ({term['percentage']:.1f}%)")
    
    print(f"\nAnalysis results saved to {output_dir}")
    
    return 0
  
  except Exception as e:
    logger.exception(f"Error analyzing data: {e}")
    return 1


if __name__ == "__main__":
  sys.exit(main())
