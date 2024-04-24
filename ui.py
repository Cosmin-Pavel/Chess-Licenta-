import random
import time
import torch
import numpy as np
import chess
import ChessEngine
from fentoboardimage import loadPiecesFolder, fenToImage
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk

from model import Net


class ChessGame:
    def __init__(self):
        self.label = None
        self.image_tk = None
        self.last_move_highlighting = None
        self.boardImage = None
        self.root = tk.Tk()
        self.gs = ChessEngine.GameState()
        self.drag_data = {}
        self.create_ui()

    def create_ui(self):

        self.boardImage = fenToImage(
            fen=self.gs.board.fen(),
            squarelength=80,
            pieceSet=loadPiecesFolder("./images"),
            darkColor="#D18B47",
            lightColor="#FFCE9E",
            highlighting=self.last_move_highlighting
        )
        self.image_tk = ImageTk.PhotoImage(self.boardImage)
        self.label = tk.Label(self.root, image=self.image_tk)
        self.label.pack()
        self.label.bind("<ButtonPress-1>", self.on_drag_start)
        self.label.bind("<ButtonRelease-1>", self.on_drag_end)
        self.game_loop()

    def on_drag_start(self, event):

        col, row = event.x // 80, event.y // 80
        piece = self.gs.board.piece_at(chess.square(col, 7 - row))
        if piece and piece.color == chess.WHITE and self.gs.white_to_move:
            self.drag_data = {'piece': piece,
                              'start_col': col,
                              'start_row': row}

    def on_drag_end(self, event):

        if self.drag_data:
            col, row = event.x // 80, event.y // 80
            start_square = chess.square(self.drag_data['start_col'], 7 - self.drag_data['start_row'])
            end_square = chess.square(col, 7 - row)
            move = chess.Move(start_square, end_square)
            if move in self.gs.board.legal_moves:
                self.gs.board.push(move)
                self.gs.white_to_move = not self.gs.white_to_move
                self.update_board(move)
                print("Move made:", move)
            self.drag_data = {}

    def game_loop(self):
        model = torch.load('chessModel.pth')
        model.eval()  # Set the model to evaluation mode

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        while not self.gs.board.is_game_over():
            if self.gs.white_to_move:
                self.root.update()
                self.root.after(100)
            else:
                self.play_black_turn(model)
                self.root.update()
                self.root.after(100)
        print("Joc încheiat")

    def play_black_turn(self, model):

        if not self.gs.white_to_move:
            old_position = self.gs.board.fen()
            _, best_move, _ = ChessEngine.GameState.model_alpha_beta(
                old_position, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, model=model, is_playing_Black=True)

            # Apply the best move to the old position
            board_copy = chess.Board(old_position)
            board_copy.push(best_move)
            new_position = board_copy.fen()

            print("old_pos:", old_position)
            print("new_pos:", new_position)
            print("best_move:", best_move)

            self.gs.board.set_fen(new_position)
            self.gs.white_to_move = not self.gs.white_to_move
            self.update_board(best_move)

    def update_board(self, move):
        # Get the start and end squares from the move object
        start_square = move.from_square
        end_square = move.to_square

        # Convert squares to algebraic notation
        start_pos = chess.square_name(start_square)
        end_pos = chess.square_name(end_square)

        # Assign the values to self.last_move_highlighting
        self.last_move_highlighting = {"yellow": [start_pos, end_pos]}

        # Update the board image
        self.boardImage = fenToImage(
            fen=self.gs.board.fen(),
            squarelength=80,
            pieceSet=loadPiecesFolder("./images"),
            darkColor="#D18B47",
            lightColor="#FFCE9E",
            highlighting=self.last_move_highlighting
        )
        self.image_tk = ImageTk.PhotoImage(self.boardImage)
        self.label.configure(image=self.image_tk)

    def run(self):
        self.root.mainloop()

    def run_experiment(self, num_moves=50):
        with open("minimax_times.txt", "w") as minimax_file, open("alpha_beta_times.txt", "w") as alpha_beta_file:
            for move_num in range(num_moves):
                print(f"\nMutare {move_num + 1}: ")
                self.make_random_white_move()
                self.saved_board_state = self.gs.board.fen()

                print("\nCalculare miscare cu Minimax")
                start_time_minimax = time.time()
                self.make_black_move_minimax()
                end_time_minimax = time.time()
                print(f"Timp pentru Minimax: {end_time_minimax - start_time_minimax:.4f} secunde")
                minimax_file.write(f"{end_time_minimax - start_time_minimax:.4f}\n")

                self.gs.board.set_fen(self.saved_board_state)
                self.gs.white_to_move = not self.gs.white_to_move

                print("\nCalculare miscare cu Alpha-Beta Pruning...")
                start_time_alpha_beta = time.time()
                self.make_black_move_alpha_beta()
                end_time_alpha_beta = time.time()
                print(f"Timp pentru Alpha-Beta Pruning: {end_time_alpha_beta - start_time_alpha_beta:.4f} secunde")
                alpha_beta_file.write(f"{end_time_alpha_beta - start_time_alpha_beta:.4f}\n")

    def make_black_move_minimax(self):

        if not self.gs.white_to_move:
            best_move, _ = ChessEngine.GameState.minimax(self.gs.board.fen(), depth=3, maximizing_player=True)
            self.gs.board.set_fen(best_move)
            self.gs.white_to_move = not self.gs.white_to_move

    def make_black_move_alpha_beta(self):

        if not self.gs.white_to_move:
            best_move, _ = ChessEngine.GameState.minimax_alpha_beta(self.gs.board.fen(), depth=5, alpha=float('-inf'),
                                                                    beta=float('inf'), maximizing_player=True)
            self.gs.board.set_fen(best_move)
            self.gs.white_to_move = not self.gs.white_to_move


if __name__ == "__main__":
    game = ChessGame()
    game.run()
    # game.run_experiment(num_moves=30)
