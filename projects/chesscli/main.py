from chess import Board, Move
from berserk import Client, TokenSession
from pathlib import Path
import os
import time

# Function to clear the console screen
def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

# Read the API token from a file
def read_token():
    token = Path("token.txt").read_text()
    return token.strip()

# Create a Berserk client with the token
def create_client(token):
    session = TokenSession(token)
    return Client(session=session)

# Print the chess board
def print_board(board, username, opponent_id):
    if username != opponent_id:
        board = board[::-1]
    board_display = "\n".join(
        [f"{8 - i} {row}" for i, row in enumerate(board.split("\n"))]
    )
    print(f"{board_display}\n  A B C D E F G H")

# Play the chess game
def play_game(client, username): # sourcery skip
    chess_board = Board()
    is_my_turn = False

    for event in client.board.stream_incoming_events():
        if event["type"] == "gameStart":
            game_id = event["game"]["gameId"]
            print(f"Game ID: {game_id}")

            while not chess_board.is_game_over():
                game_state = client.board.stream_game_state(game_id)
                is_my_turn = True

                for state in game_state:
                    if state["type"] == "gameFull":
                        opponent_id = state["white"]["id"]
                        is_my_turn = opponent_id == username
                    elif state["type"] == "gameState" and state["moves"] != "":
                        last_move = Move.from_uci(state["moves"].split(" ")[-1])
                        chess_board.push(last_move)

                    clear_screen()
                    print_board(chess_board, username, opponent_id)

                    if is_my_turn:
                        while True:
                            move_input = input("Enter your move (e.g., 'e2e4'): ")
                            move = Move.from_uci(move_input)
                            if move in chess_board.legal_moves:
                                client.board.make_move(game_id, move_input)
                                break
                            print("Invalid move")

                    is_my_turn = not is_my_turn

                clear_screen()
                print_board(chess_board, username, opponent_id)

        time.sleep(0.1)

def main():
    token = read_token()
    client = create_client(token)
    username = client.account.get()["username"]
    play_game(client, username)

if __name__ == "__main__":
    main()
