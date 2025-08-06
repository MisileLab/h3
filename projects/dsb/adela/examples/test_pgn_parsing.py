"""Test script for parsing PGN files with the updated parser."""

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

1. d4 { [%clk 0:01:00] } 1... c5 { [%clk 0:01:00] } 2. e3 { [%clk 0:01:00] } 2... e6 { [%clk 0:00:59] } 3. dxc5 { [%clk 0:00:59] } 3... Bxc5 { [%clk 0:00:58] } 4. Nf3 { [%clk 0:00:59] } 4... Nf6 { [%clk 0:00:57] } 5. c3 { [%clk 0:00:59] } 5... Nc6 { [%clk 0:00:56] } 6. Bb5 { [%clk 0:00:58] } 6... a6 { [%clk 0:00:55] } 7. Bxc6 { [%clk 0:00:57] } 7... bxc6 { [%clk 0:00:55] } 8. O-O { [%clk 0:00:57] } 8... d5 { [%clk 0:00:54] } 9. Nd4 { [%clk 0:00:56] } 9... Bd7 { [%clk 0:00:52] } 10. Qf3 { [%clk 0:00:53] } 10... O-O { [%clk 0:00:51] } 11. h4 { [%clk 0:00:51] } 11... e5 { [%clk 0:00:50] } 12. Nc2 { [%clk 0:00:50] } 12... Bd6 { [%clk 0:00:49] } 13. h5 { [%clk 0:00:47] } 13... e4 { [%clk 0:00:49] } 14. Qe2 { [%clk 0:00:46] } 14... Bg4 { [%clk 0:00:46] } 15. f3 { [%clk 0:00:43] } 15... exf3 { [%clk 0:00:46] } 16. gxf3 { [%clk 0:00:40] } 16... Bxh5 { [%clk 0:00:45] } 17. e4 { [%clk 0:00:39] } 17... dxe4 { [%clk 0:00:44] } 18. Ne1 { [%clk 0:00:34] } 18... exf3 { [%clk 0:00:42] } 19. Nxf3 { [%clk 0:00:33] } 19... Qd7 { [%clk 0:00:39] } 20. Bg5 { [%clk 0:00:31] } 20... Qg4+ { [%clk 0:00:37] } 21. Qg2 { [%clk 0:00:30] } 21... Qxg2+ { [%clk 0:00:35] } 22. Kxg2 { [%clk 0:00:29] } 22... Bxf3+ { [%clk 0:00:34] } 23. Rxf3 { [%clk 0:00:27] } 23... Ne4 { [%clk 0:00:32] } 24. Be3 { [%clk 0:00:26] } 24... h6 { [%clk 0:00:31] } 25. Nd2 { [%clk 0:00:24] } 25... Rfe8 { [%clk 0:00:30] } 26. Nxe4 { [%clk 0:00:23] } 26... Rxe4 { [%clk 0:00:30] } 27. Bd4 { [%clk 0:00:19] } 27... Rae8 { [%clk 0:00:28] } 28. Raf1 { [%clk 0:00:18] } 28... c5 { [%clk 0:00:27] } 29. Bg1 { [%clk 0:00:16] } 29... f6 { [%clk 0:00:24] } 30. a3 { [%clk 0:00:15] } 30... Re2+ { [%clk 0:00:22] } 31. Bf2 { [%clk 0:00:13] } 31... Rxb2 { [%clk 0:00:21] } 32. Kh3 { [%clk 0:00:09] } 32... Ree2 { [%clk 0:00:20] } 33. Kg2 { [%clk 0:00:07] } 33... c4 { [%clk 0:00:18] } 34. a4 { [%clk 0:00:07] } 34... Bc5 { [%clk 0:00:17] } 35. Re3 { [%clk 0:00:04] } 35... Bxe3 { [%clk 0:00:16] } 0-1"""
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pgn') as f:
        f.write(sample_pgn)
        temp_path = f.name
    
    return Path(temp_path)


def test_move_extraction():
    """Test the move extraction functionality."""
    print("Creating sample PGN file...")
    sample_pgn_path = create_sample_pgn()
    
    print("\nTesting move extraction...")
    parser = LargePGNParser()
    
    # Read the PGN file
    with open(sample_pgn_path, 'r') as f:
        pgn_content = f.read()
    
    # Extract moves and clock times
    moves, clock_times = parser._extract_moves_and_times(pgn_content)
    
    print("\nExtracted moves:")
    print(moves)
    print(f"\nTotal moves: {len(moves)}")
    
    print("\nExtracted clock times:")
    print(clock_times[:10])  # Show first 10 clock times
    print(f"\nTotal clock times: {len(clock_times)}")
    
    # Process the game
    game_record = parser._process_game(pgn_content, pgn_content)
    
    print("\nProcessed game record:")
    for key, value in game_record.items():
        if key == 'moves':
            print(f"{key}: {value[:100]}...")  # Show first 100 characters of moves
        else:
            print(f"{key}: {value}")
    
    # Clean up
    os.unlink(sample_pgn_path)
    print(f"\nDeleted sample PGN file: {sample_pgn_path}")


if __name__ == "__main__":
    test_move_extraction()
