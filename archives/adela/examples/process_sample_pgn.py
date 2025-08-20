"""Example script for processing a sample PGN file."""

import os
import sys
from pathlib import Path
import tempfile

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adela.data.large_pgn_parser import LargePGNParser


def create_sample_pgn() -> Path:
    """Create a sample PGN file with a few games.

    Returns:
        Path to the sample PGN file.
    """
    sample_pgn = """[Event "Rated Bullet game"]
[Site "https://lichess.org/VsUqVhC2"]
[Date "2025.07.01"]
[Round "-"]
[White "my_name_jeff"]
[Black "xxxgrishaxxx"]
[Result "0-1"]
[UTCDate "2025.07.01"]
[UTCTime "00:00:31"]
[WhiteElo "1706"]
[BlackElo "1671"]
[WhiteRatingDiff "-6"]
[BlackRatingDiff "+6"]
[ECO "A43"]
[Opening "Benoni Defense: Old Benoni"]
[TimeControl "60+0"]
[Termination "Time forfeit"]

1. d4 { [%clk 0:01:00] } 1... c5 { [%clk 0:01:00] } 2. e3 { [%clk 0:01:00] } 2... e6 { [%clk 0:00:59] } 3. dxc5 { [%clk 0:00:59] } 3... Bxc5 { [%clk 0:00:58] } 4. Nf3 { [%clk 0:00:59] } 4... Nf6 { [%clk 0:00:57] } 5. c3 { [%clk 0:00:59] } 5... Nc6 { [%clk 0:00:56] } 6. Bb5 { [%clk 0:00:58] } 6... a6 { [%clk 0:00:55] } 7. Bxc6 { [%clk 0:00:57] } 7... bxc6 { [%clk 0:00:55] } 8. O-O { [%clk 0:00:57] } 8... d5 { [%clk 0:00:54] } 9. Nd4 { [%clk 0:00:56] } 9... Bd7 { [%clk 0:00:52] } 10. Qf3 { [%clk 0:00:53] } 10... O-O { [%clk 0:00:51] } 11. h4 { [%clk 0:00:51] } 11... e5 { [%clk 0:00:50] } 12. Nc2 { [%clk 0:00:50] } 12... Bd6 { [%clk 0:00:49] } 13. h5 { [%clk 0:00:47] } 13... e4 { [%clk 0:00:49] } 14. Qe2 { [%clk 0:00:46] } 14... Bg4 { [%clk 0:00:46] } 15. f3 { [%clk 0:00:43] } 15... exf3 { [%clk 0:00:46] } 16. gxf3 { [%clk 0:00:40] } 16... Bxh5 { [%clk 0:00:45] } 17. e4 { [%clk 0:00:39] } 17... dxe4 { [%clk 0:00:44] } 18. Ne1 { [%clk 0:00:34] } 18... exf3 { [%clk 0:00:42] } 19. Nxf3 { [%clk 0:00:33] } 19... Qd7 { [%clk 0:00:39] } 20. Bg5 { [%clk 0:00:31] } 20... Qg4+ { [%clk 0:00:37] } 21. Qg2 { [%clk 0:00:30] } 21... Qxg2+ { [%clk 0:00:35] } 22. Kxg2 { [%clk 0:00:29] } 22... Bxf3+ { [%clk 0:00:34] } 23. Rxf3 { [%clk 0:00:27] } 23... Ne4 { [%clk 0:00:32] } 24. Be3 { [%clk 0:00:26] } 24... h6 { [%clk 0:00:31] } 25. Nd2 { [%clk 0:00:24] } 25... Rfe8 { [%clk 0:00:30] } 26. Nxe4 { [%clk 0:00:23] } 26... Rxe4 { [%clk 0:00:30] } 27. Bd4 { [%clk 0:00:19] } 27... Rae8 { [%clk 0:00:28] } 28. Raf1 { [%clk 0:00:18] } 28... c5 { [%clk 0:00:27] } 29. Bg1 { [%clk 0:00:16] } 29... f6 { [%clk 0:00:24] } 30. a3 { [%clk 0:00:15] } 30... Re2+ { [%clk 0:00:22] } 31. Bf2 { [%clk 0:00:13] } 31... Rxb2 { [%clk 0:00:21] } 32. Kh3 { [%clk 0:00:09] } 32... Ree2 { [%clk 0:00:20] } 33. Kg2 { [%clk 0:00:07] } 33... c4 { [%clk 0:00:18] } 34. a4 { [%clk 0:00:07] } 34... Bc5 { [%clk 0:00:17] } 35. Re3 { [%clk 0:00:04] } 35... Bxe3 { [%clk 0:00:16] } 0-1

[Event "Rated Bullet game"]
[Site "https://lichess.org/AbCdEfGh"]
[Date "2025.07.01"]
[Round "-"]
[White "player1"]
[Black "player2"]
[Result "1-0"]
[UTCDate "2025.07.01"]
[UTCTime "00:05:31"]
[WhiteElo "2100"]
[BlackElo "2050"]
[WhiteRatingDiff "+6"]
[BlackRatingDiff "-6"]
[ECO "B01"]
[Opening "Scandinavian Defense"]
[TimeControl "60+0"]
[Termination "Normal"]

1. e4 { [%clk 0:01:00] } 1... d5 { [%clk 0:01:00] } 2. exd5 { [%clk 0:00:59] } 2... Qxd5 { [%clk 0:00:59] } 3. Nc3 { [%clk 0:00:58] } 3... Qa5 { [%clk 0:00:58] } 4. d4 { [%clk 0:00:57] } 4... c6 { [%clk 0:00:57] } 5. Nf3 { [%clk 0:00:56] } 5... Nf6 { [%clk 0:00:56] } 6. Bc4 { [%clk 0:00:55] } 6... Bf5 { [%clk 0:00:55] } 7. Bd2 { [%clk 0:00:54] } 7... e6 { [%clk 0:00:54] } 8. Qe2 { [%clk 0:00:53] } 8... Bb4 { [%clk 0:00:53] } 9. O-O-O { [%clk 0:00:52] } 9... O-O { [%clk 0:00:52] } 10. Ne5 { [%clk 0:00:51] } 10... Bxc3 { [%clk 0:00:51] } 11. Bxc3 { [%clk 0:00:50] } 11... Qb6 { [%clk 0:00:50] } 12. g4 { [%clk 0:00:49] } 12... Bg6 { [%clk 0:00:49] } 13. h4 { [%clk 0:00:48] } 13... h6 { [%clk 0:00:48] } 14. h5 { [%clk 0:00:47] } 14... Bh7 { [%clk 0:00:47] } 15. g5 { [%clk 0:00:46] } 15... hxg5 { [%clk 0:00:46] } 16. Bxg7 { [%clk 0:00:45] } 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/XyZaBcDe"]
[Date "2025.07.01"]
[Round "-"]
[White "grandmaster1"]
[Black "grandmaster2"]
[Result "1/2-1/2"]
[UTCDate "2025.07.01"]
[UTCTime "00:10:31"]
[WhiteElo "2450"]
[BlackElo "2480"]
[WhiteRatingDiff "+0"]
[BlackRatingDiff "+0"]
[ECO "C65"]
[Opening "Ruy Lopez: Berlin Defense"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:04:58] } 2... Nc6 { [%clk 0:04:58] } 3. Bb5 { [%clk 0:04:56] } 3... Nf6 { [%clk 0:04:56] } 4. d3 { [%clk 0:04:54] } 4... Bc5 { [%clk 0:04:54] } 5. c3 { [%clk 0:04:52] } 5... O-O { [%clk 0:04:52] } 6. O-O { [%clk 0:04:50] } 6... d6 { [%clk 0:04:50] } 7. h3 { [%clk 0:04:48] } 7... Ne7 { [%clk 0:04:48] } 8. d4 { [%clk 0:04:46] } 8... Bb6 { [%clk 0:04:46] } 9. Bd3 { [%clk 0:04:44] } 9... Ng6 { [%clk 0:04:44] } 10. Re1 { [%clk 0:04:42] } 10... c6 { [%clk 0:04:42] } 11. Nbd2 { [%clk 0:04:40] } 11... Re8 { [%clk 0:04:40] } 12. Nf1 { [%clk 0:04:38] } 12... d5 { [%clk 0:04:38] } 13. exd5 { [%clk 0:04:36] } 13... cxd5 { [%clk 0:04:36] } 14. dxe5 { [%clk 0:04:34] } 14... Nxe5 { [%clk 0:04:34] } 15. Nxe5 { [%clk 0:04:32] } 15... Rxe5 { [%clk 0:04:32] } 16. Rxe5 { [%clk 0:04:30] } 16... Qxe5 { [%clk 0:04:30] } 17. Ng3 { [%clk 0:04:28] } 17... Be6 { [%clk 0:04:28] } 18. Qf3 { [%clk 0:04:26] } 18... Rc8 { [%clk 0:04:26] } 19. Be3 { [%clk 0:04:24] } 19... Bxe3 { [%clk 0:04:24] } 20. Qxe3 { [%clk 0:04:22] } 20... Qxe3 { [%clk 0:04:22] } 21. fxe3 { [%clk 0:04:20] } 21... Rxc3 { [%clk 0:04:20] } 22. bxc3 { [%clk 0:04:18] } 22... Nxd3 { [%clk 0:04:18] } 23. Rd1 { [%clk 0:04:16] } 23... Nc5 { [%clk 0:04:16] } 24. Nf5 { [%clk 0:04:14] } 24... Bxf5 { [%clk 0:04:14] } 25. Rxd5 { [%clk 0:04:12] } 25... Ne6 { [%clk 0:04:12] } 26. Rd6 { [%clk 0:04:10] } 26... Nc5 { [%clk 0:04:10] } 27. Rd5 { [%clk 0:04:08] } 27... Ne6 { [%clk 0:04:08] } 28. Rd6 { [%clk 0:04:06] } 28... Nc5 { [%clk 0:04:06] } 29. Rd5 { [%clk 0:04:04] } 29... Ne6 { [%clk 0:04:04] } 1/2-1/2
"""
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pgn') as f:
        f.write(sample_pgn)
        temp_path = f.name
    
    return Path(temp_path)


def main():
    """Main function."""
    print("Creating sample PGN file...")
    sample_pgn_path = create_sample_pgn()
    print(f"Sample PGN file created at: {sample_pgn_path}")
    
    # Create output directory
    output_dir = Path("./data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing sample PGN file...")
    parser = LargePGNParser(
        batch_size=10,  # Small batch size for the example
        output_dir=output_dir,
        min_elo=0,  # Include all games
        output_filename_prefix="sample_games"
    )
    
    # Parse the PGN file
    games_processed = parser.parse_pgn_file(
        pgn_path=sample_pgn_path,
        show_progress=True
    )
    
    print(f"\nProcessed {games_processed} games")
    
    # Merge batch files
    print("\nMerging batch files...")
    merged_path = parser.merge_parquet_files()
    print(f"Merged file saved to: {merged_path}")
    
    # Read the merged file
    print("\nReading the merged Parquet file...")
    import polars as pl
    df = pl.read_parquet(merged_path)
    
    print("\nData summary:")
    print(f"Number of games: {len(df)}")
    print(f"Columns: {df.columns}")
    
    print("\nSample data:")
    print(df.head())
    
    # Clean up
    os.unlink(sample_pgn_path)
    print(f"\nDeleted sample PGN file: {sample_pgn_path}")
    
    print("\nDone! You can now analyze the processed data.")
    print(f"To analyze the data, run: python analyze_chess_data.py {merged_path}")


if __name__ == "__main__":
    main()
