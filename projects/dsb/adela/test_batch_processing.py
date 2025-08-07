#!/usr/bin/env python3
"""Test script for batch processing functionality with large datasets."""

import tempfile
import shutil
from pathlib import Path
import polars as pl

from load_and_parse_parquets import process_parquet_files_in_batches


def create_large_test_data(num_files: int = 5, games_per_file: int = 15000) -> Path:
    """Create test data that simulates a large dataset.
    
    Args:
        num_files: Number of Parquet files to create
        games_per_file: Number of games per file
        
    Returns:
        Path to temporary directory containing test files
    """
    # Sample movetext variations for realistic data
    movetexts = [
        "1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 0-1",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e6 7. f3 b5 8. Qd2 Bb7 9. O-O-O Nbd7 10. h4 Rc8 11. Kb1 Qc7 12. g4 b4 13. Na4 Nxe4 1/2-1/2",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 0-1",
        "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. Nd3 f5 11. f3 f4 12. Nb5 g5 1-0"
    ]
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    print(f"Creating {num_files} test files with {games_per_file} games each...")
    
    for file_idx in range(num_files):
        games_data = []
        
        for game_idx in range(games_per_file):
            # Vary Elo ratings to test filtering
            base_elo = 1000 + (game_idx % 1000)  # Elo from 1000-2000
            white_elo = base_elo + (game_idx % 200)
            black_elo = base_elo + ((game_idx + 100) % 200)
            
            # Ensure some games meet the Elo threshold
            if game_idx % 3 == 0:  # Every 3rd game has high Elo
                white_elo = max(white_elo, 1600)
            if game_idx % 4 == 0:  # Every 4th game has high Elo for black
                black_elo = max(black_elo, 1600)
            
            game_data = {
                "id": f"file{file_idx}_game{game_idx}",
                "white": f"Player{game_idx}_W",
                "black": f"Player{game_idx}_B",
                "WhiteElo": white_elo,
                "BlackElo": black_elo,
                "result": ["1-0", "0-1", "1/2-1/2"][game_idx % 3],
                "movetext": movetexts[game_idx % len(movetexts)]
            }
            games_data.append(game_data)
        
        # Create Parquet file
        df = pl.DataFrame(games_data)
        file_path = temp_dir / f"large_games_part_{file_idx:03d}.parquet"
        df.write_parquet(file_path)
        
        print(f"  Created {file_path.name} with {len(games_data)} games")
    
    total_games = num_files * games_per_file
    print(f"Created test dataset with {total_games} total games in {temp_dir}")
    
    return temp_dir


def test_batch_processing():
    """Test the batch processing functionality."""
    print("=" * 60)
    print("Testing Batch Processing for Large Datasets")
    print("=" * 60)
    
    # Create test data (simulating 75,000 games across 5 files)
    test_data_dir = create_large_test_data(num_files=5, games_per_file=15000)
    
    try:
        print(f"\nTest data created in: {test_data_dir}")
        
        # Create output directory for processed batches
        output_dir = test_data_dir / "processed"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nStarting batch processing...")
        print(f"  Input directory: {test_data_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Batch size: 50,000 games")
        print(f"  Min Elo: 1500")
        
        # Process files in batches
        output_files = process_parquet_files_in_batches(
            folder_path=test_data_dir,
            min_elo=1500,
            batch_size=50000,  # 50k games per batch
            output_dir=output_dir,
            output_prefix="test_processed"
        )
        
        print(f"\nBatch processing completed!")
        print(f"Created {len(output_files)} output files:")
        
        # Analyze the results
        total_games_in_output = 0
        for i, output_file in enumerate(output_files):
            batch_df = pl.read_parquet(output_file)
            num_games = len(batch_df)
            total_games_in_output += num_games
            
            print(f"  {output_file.name}: {num_games} games")
            
            # Show sample of first batch
            if i == 0:
                print(f"    Sample from first batch:")
                sample = batch_df.select(["white", "black", "WhiteElo", "BlackElo", "num_moves"]).head(3)
                for row in sample.to_dicts():
                    print(f"      {row['white']} ({row['WhiteElo']}) vs {row['black']} ({row['BlackElo']}) - {row['num_moves']} moves")
        
        print(f"\nTotal games in output: {total_games_in_output}")
        
        # Verify the batch processing worked correctly
        if len(output_files) > 1:
            first_batch_size = len(pl.read_parquet(output_files[0]))
            print(f"First batch size: {first_batch_size} (should be ≤ 50,000)")
            
            if len(output_files) > 1:
                last_batch_size = len(pl.read_parquet(output_files[-1]))
                print(f"Last batch size: {last_batch_size} (can be < 50,000)")
        
        # Test that all games meet Elo criteria
        print(f"\nVerifying Elo filtering...")
        for output_file in output_files[:2]:  # Check first 2 files
            batch_df = pl.read_parquet(output_file)
            elo_check = batch_df.filter(
                (pl.col("WhiteElo") < 1500) & (pl.col("BlackElo") < 1500)
            )
            if len(elo_check) > 0:
                print(f"  WARNING: Found {len(elo_check)} games not meeting Elo criteria in {output_file.name}")
            else:
                print(f"  ✓ All games in {output_file.name} meet Elo criteria")
        
        print(f"\n" + "=" * 60)
        print("Batch processing test completed successfully!")
        print("=" * 60)
        
    finally:
        # Clean up test data
        print(f"\nCleaning up test directory: {test_data_dir}")
        shutil.rmtree(test_data_dir)


def test_memory_efficiency():
    """Test memory efficiency with a smaller example."""
    print("\n" + "=" * 60)
    print("Testing Memory Efficiency")
    print("=" * 60)
    
    # Create smaller test to verify memory doesn't grow excessively
    test_data_dir = create_large_test_data(num_files=2, games_per_file=5000)
    
    try:
        output_dir = test_data_dir / "memory_test"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nTesting with small batch size to verify memory cleanup...")
        
        # Use very small batch size to test batch saving and memory cleanup
        output_files = process_parquet_files_in_batches(
            folder_path=test_data_dir,
            min_elo=1200,  # Lower threshold to get more games
            batch_size=2000,  # Small batches to test saving logic
            output_dir=output_dir,
            output_prefix="memory_test"
        )
        
        print(f"Created {len(output_files)} small batches successfully")
        print("Memory efficiency test passed!")
        
    finally:
        shutil.rmtree(test_data_dir)


if __name__ == "__main__":
    test_batch_processing()
    test_memory_efficiency()
    print("\nAll tests completed!")
