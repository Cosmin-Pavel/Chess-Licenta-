
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
        self.gs.board = chess.Board(self.gs.board)  # Convert the FEN string to a chess.Board object
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

        # Bind mouse events for drag-and-drop
        self.label.bind("<ButtonPress-1>", self.on_drag_start)

        self.label.bind("<ButtonRelease-1>", self.on_drag_end)

        # Start the game loop
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
                self.gs.white_to_move = not self.gs.white_to_move  # Switch turn
                self.update_board()

            self.drag_data = {}  # Clear drag data

    def game_loop(self):
        while not self.gs.board.is_game_over():
            if self.gs.white_to_move:
                # White's turn (human player)
                self.root.update()
                self.root.after(100)  # Small delay to allow the GUI to update
            else:
                # Black's turn (computer player)
                self.play_black_turn()
                self.root.update()
                self.root.after(100)  # Small delay to allow the GUI to update

        print("Game Over")

    def play_black_turn(self):
        if not self.gs.white_to_move:
            start_time = time.time()  # Record the start time
            best_move, _ = ChessEngine.GameState.minimax(self.gs.board.fen(), depth=3, maximizing_player=True)
            end_time = time.time()  # Record the end time

            print(f"Time taken for minimax: {end_time - start_time} seconds")

            self.gs.board.set_fen(best_move)
            self.gs.white_to_move = not self.gs.white_to_move
            self.update_board()

    def update_board(self, drag_data=None):
        if drag_data:
            piece = drag_data['piece']
            start_col, start_row = drag_data['start_col'], drag_data['start_row']
            current_col, current_row = drag_data['current_col'], drag_data['current_row']

            # Perform the move on the board
            move = chess.Move.from_uci(f"{start_col}{7 - start_row}{current_col}{7 - current_row}")
            self.gs.board.push(move)

        # Update the displayed chessboard
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


if __name__ == "__main__":
    game = ChessGame()
    game.run()