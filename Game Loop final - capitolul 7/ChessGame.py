import chess

import ChessEngine
from fentoboardimage import loadPiecesFolder, fenToImage
from PIL import Image, ImageTk


class ChessGame:
    def __init__(self):
        self.label = None
        self.gs = ChessEngine.GameState()
        self.boardImage = None

    def create_ui(self):
        self.boardImage = fenToImage(
            fen=self.gs.board.fen(),
            squarelength=50,
            pieceSet=loadPiecesFolder("./images"),
            darkColor="#D18B47",
            lightColor="#FFCE9E"
        )

    def get_board_image(self):
        return self.boardImage

    def play_black_turn(self, model):

        if not self.gs.white_to_move:
            old_position = self.gs.board.fen()
            _, best_move, _ = ChessEngine.GameState.model_alpha_beta(
                old_position, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, model=model, is_playing_Black=True)


            board_copy = chess.Board(old_position)
            board_copy.push(best_move)
            new_position = board_copy.fen()

            print("old_pos:", old_position)
            print("new_pos:", new_position)
            print("best_move:", best_move)

            self.gs.board.set_fen(new_position)
            self.gs.white_to_move = not self.gs.white_to_move
            self.update_board(best_move)
            return best_move


    def update_board(self, move):

        start_square = move.from_square
        end_square = move.to_square


        start_pos = chess.square_name(start_square)
        end_pos = chess.square_name(end_square)

        self.last_move_highlighting = {"yellow": [start_pos, end_pos]}


        self.boardImage = fenToImage(
            fen=self.gs.board.fen(),
            squarelength=50,
            pieceSet=loadPiecesFolder("./images"),
            darkColor="#D18B47",
            lightColor="#FFCE9E",
            highlighting=self.last_move_highlighting
        )
        #self.image_tk = ImageTk.PhotoImage(self.boardImage)
        #self.label.configure(image=self.image_tk)