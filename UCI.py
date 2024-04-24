from ChessEngine import GameState

class UCI:
    ENGINENAME = "CozminCrimaEngine"
    game_state = GameState()

    @staticmethod
    def uci_communication():
        while True:
            input_string = input()
            if input_string == "uci":
                UCI.input_uci()
            elif input_string == "isready":
                UCI.input_is_ready()
            elif input_string == "setoption":
                # Add implementation for setoption command
                pass
            elif input_string == "ucinewgame":
                UCI.input_uci_new_game()
            elif input_string.startswith("position"):
                UCI.input_position(input_string)
            elif input_string.startswith("go"):
                UCI.input_go(input_string)
            elif input_string == "stop":
                # Add implementation for stop command
                pass
            elif input_string == "ponderhit":
                # Add implementation for ponderhit command
                pass
            elif input_string == "start":
                # Add implementation for start command
                pass
            elif input_string == "quit":
                # Add implementation for quit command
                pass
            elif input_string == "fen":
                # Add implementation for fen command
                pass
            elif input_string == "xyzzy":
                UCI.input_xyzzy()
            else:
                print("Unknown command:", input_string)

    @staticmethod
    def input_uci():
        try:
            print("id name", UCI.ENGINENAME)
            print("id author CozminCrima")
            print("uciok")
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_is_ready():
        try:
            print("readyok")
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_uci_new_game():
        try:
            # Initialize new game state
            UCI.game_state = GameState()
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_position(input_string):
        try:
            # Parse input_string to set up the board position
            tokens = input_string.split()
            if "startpos" in tokens:
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            else:
                fen_index = tokens.index("fen") + 1
                fen = " ".join(tokens[fen_index:])

            moves_index = tokens.index("moves") + 1 if "moves" in tokens else len(tokens)
            moves = tokens[moves_index:]

            # Set up the position on the board and play the moves
            UCI.game_state.board = fen
            for move in moves:
                UCI.game_state.board.push_uci(move)
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_go(input_string):
        try:
            print("UCI board: ", UCI.game_state.board)
            best_move,  best_actual_move, _ = GameState.minimax_uci(UCI.game_state.board, 3)
            print("bestactualmove:", best_actual_move)

            # Print the best move
            print("bestmove", best_move)
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_stop():
        try:
            # Add implementation for stop command
            pass
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_ponderhit():
        try:
            # Add implementation for ponderhit command
            pass
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_start():
        try:
            # Add implementation for start command
            pass
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_quit():
        try:
            # Add implementation for quit command
            pass
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_fen():
        try:
            # Add implementation for fen command
            pass
        except Exception as e:
            print("Error:", e)

    @staticmethod
    def input_xyzzy():
        try:
            print("Nothing happens.")
        except Exception as e:
            print("Error:", e)

# Add implementations for setoption, go, stop, ponderhit, start, quit, and fen commands

# Start the UCI communication loop
UCI.uci_communication()
