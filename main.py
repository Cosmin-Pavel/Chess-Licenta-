import random
import time

import chess
import ChessEngine
from fentoboardimage import fenToImage, loadPiecesFolder
from PIL import Image, ImageTk
import tkinter as tk

class ChessGame:
    def __init__(self):
        self.root = tk.Tk()
        self.gs = ChessEngine.GameState()
        self.gs.board = chess.Board(self.gs.board)
        self.drag_data = {}
        self.create_ui()

    def create_ui(self):

        self.boardImage = fenToImage(
            fen=self.gs.board.fen(),
            squarelength=80,
            pieceSet=loadPiecesFolder("./images"),
            darkColor="#D18B47",
            lightColor="#FFCE9E"
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


                self.update_board()

            self.drag_data = {}

    def game_loop(self):

        while not self.gs.board.is_game_over():
            if self.gs.white_to_move:

                self.root.update()
                self.root.after(100)
            else:

                self.play_black_turn()
                self.root.update()
                self.root.after(100)

        print("Joc încheiat")

    def play_black_turn(self):

        if not self.gs.white_to_move:
            start_time = time.time()
            best_move, _ = ChessEngine.GameState.minimax_alpha_beta(
                self.gs.board.fen(), depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
            end_time = time.time()


            self.gs.board.set_fen(best_move)
            self.gs.white_to_move = not self.gs.white_to_move
            self.update_board()

    def update_board(self, drag_data=None):

        if drag_data:
            piece = drag_data['piece']
            start_col, start_row = drag_data['start_col'], drag_data['start_row']
            current_col, current_row = drag_data['current_col'], drag_data['current_row']


            move = chess.Move.from_uci(f"{start_col}{7 - start_row}{current_col}{7 - current_row}")
            self.gs.board.push(move)


        self.boardImage = fenToImage(
            fen=self.gs.board.fen(),
            squarelength=80,
            pieceSet=loadPiecesFolder("./images"),
            darkColor="#D18B47",
            lightColor="#FFCE9E"
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

    def make_random_white_move(self):

        legal_moves = list(self.gs.board.legal_moves)
        random_move = random.choice(legal_moves)
        self.gs.board.push(random_move)
        self.gs.white_to_move = not self.gs.white_to_move

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
    #game.run_experiment(num_moves=30)
