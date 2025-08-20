#!/usr/bin/env python3
"""Test script to demonstrate the Parquet processing functionality."""

import tempfile
from pathlib import Path
import polars as pl

from load_and_parse_parquets import load_parquet_files, filter_by_elo, add_parsed_moves, parse_movetext_to_moves


def create_test_data() -> Path:
    """Create test Parquet files with sample chess game data.
    
    Returns:
        Path to temporary directory containing test Parquet files
    """
    # Sample game data with various Elo ratings and movetext
    games_data = [
        {
            "id": "game1",
            "white": "PlayerA",
            "black": "PlayerB", 
            "WhiteElo": 1600,
            "BlackElo": 1450,
            "result": "1-0",
            "movetext": "1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0"
        },
        {
            "id": "game2", 
            "white": "PlayerC",
            "black": "PlayerD",
            "WhiteElo": 1200,
            "BlackElo": 1300,
            "result": "0-1",
            "movetext": "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 0-1"
        },
        {
            "id": "game3",
            "white": "PlayerE", 
            "black": "PlayerF",
            "WhiteElo": 1800,
            "BlackElo": 1750,
            "result": "1/2-1/2",
            "movetext": "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e6 7. f3 b5 8. Qd2 Bb7 9. O-O-O Nbd7 10. h4 Rc8 11. Kb1 Qc7 12. g4 b4 13. Na4 Nxe4 1/2-1/2"
        },
        {
            "id": "game4",
            "white": "PlayerG",
            "black": "PlayerH", 
            "WhiteElo": 1400,
            "BlackElo": 1600,
            "result": "0-1",
            "movetext": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 0-1"
        },
        {
            "id": "game5",
            "white": "PlayerI",
            "black": "PlayerJ",
            "WhiteElo": 900,
            "BlackElo": 1000, 
            "result": "1-0",
            "movetext": "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. Nd3 f5 11. f3 f4 12. Nb5 g5 1-0"
        }
    ]
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Split data into multiple Parquet files to test concatenation
    batch1 = games_data[:3]
    batch2 = games_data[3:]
    
    # Create first batch file
    df1 = pl.DataFrame(batch1)
    df1.write_parquet(temp_dir / "games_batch_1.parquet")
    
    # Create second batch file  
    df2 = pl.DataFrame(batch2)
    df2.write_parquet(temp_dir / "games_batch_2.parquet")
    
    print(f"Created test data with {len(games_data)} games in {temp_dir}")
    print(f"Files created: {list(temp_dir.glob('*.parquet'))}")
    
    return temp_dir


def test_movetext_parsing():
    """Test the movetext parsing function with various examples."""
    print("\n=== Testing Movetext Parsing ===")
    
    test_cases = [
        "1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 0-1",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e6 7. f3 b5 8. Qd2 Bb7 9. O-O-O Nbd7 10. h4 Rc8 11. Kb1 Qc7 12. g4 b4 13. Na4 Nxe4 1/2-1/2"
    ]
    
    for i, movetext in enumerate(test_cases):
        moves = parse_movetext_to_moves(movetext)
        print(f"\nTest {i+1}:")
        print(f"Input: {movetext[:80]}...")
        print(f"Parsed moves ({len(moves)}): {moves[:10]}...")  # Show first 10 moves
        

def test_full_pipeline():
    """Test the complete pipeline from loading to parsing."""
    print("\n=== Testing Full Pipeline ===")
    
    # Create test data
    test_dir = create_test_data()
    
    try:
        # Load Parquet files
        print("\n1. Loading Parquet files...")
        df = load_parquet_files(test_dir)
        print(f"   Loaded {len(df)} games")
        
        # Show original data
        print("\n2. Original data overview:")
        print("   Elo ranges:")
        for row in df.select(["white", "black", "white_elo", "black_elo"]).to_dicts():
            print(f"     {row['white']} ({row['white_elo']}) vs {row['black']} ({row['black_elo']})")
        
        # Filter by Elo >= 1500
        print("\n3. Filtering by Elo >= 1500...")
        filtered_df = filter_by_elo(df, min_elo=1500)
        print(f"   Filtered to {len(filtered_df)} games")
        
        if len(filtered_df) > 0:
            print("   Remaining games:")
            for row in filtered_df.select(["white", "black", "white_elo", "black_elo"]).to_dicts():
                print(f"     {row['white']} ({row['white_elo']}) vs {row['black']} ({row['black_elo']})")
        
        # Parse movetext
        print("\n4. Parsing movetext...")
        parsed_df = add_parsed_moves(filtered_df)
        print(f"   Successfully parsed {len(parsed_df)} games")
        
        # Show results
        print("\n5. Results:")
        for row in parsed_df.select(["white", "black", "parsed_moves", "num_moves"]).to_dicts():
            print(f"   {row['white']} vs {row['black']}: {row['num_moves']} moves")
            print(f"     First 8 moves: {row['parsed_moves'][:8]}")
        
        # Show statistics
        if len(parsed_df) > 0:
            stats = parsed_df.select([
                pl.col("num_moves").mean().alias("avg_moves"),
                pl.col("num_moves").min().alias("min_moves"), 
                pl.col("num_moves").max().alias("max_moves")
            ]).to_dicts()[0]
            
            print(f"\n6. Move statistics:")
            print(f"   Average moves: {stats['avg_moves']:.1f}")
            print(f"   Min moves: {stats['min_moves']}")
            print(f"   Max moves: {stats['max_moves']}")
    
    finally:
        # Clean up test data
        import shutil
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")


def main():
    """Run all tests."""
    print("Chess Game Parquet Processing Test")
    print("=" * 40)
    
    # Test movetext parsing
    test_movetext_parsing()
    
    # Test full pipeline
    test_full_pipeline()
    
    print("\n" + "=" * 40)
    print("All tests completed!")


if __name__ == "__main__":
    main()
