import random
import chess
import ChessEngine
from fentoboardimage import fenToImage, loadPiecesFolder
from PIL import Image, ImageTk
import tkinter as tk

class ChessGame:
    def __init__(self):
        self.root = tk.Tk()
        self.gs = ChessEngine.GameState()
        self.create_ui()

    def create_ui(self):
        self.boardImage = fenToImage(
            fen=self.gs.board,
            squarelength=80,
            pieceSet=loadPiecesFolder("./images"),
            darkColor="#D18B47",
            lightColor="#FFCE9E"
        )


        self.image_tk = ImageTk.PhotoImage(self.boardImage)


        self.label = tk.Label(self.root, image=self.image_tk)
        self.label.pack()


        self.button_white = tk.Button(self.root, text="White Turn", command=self.play_white_turn)
        self.button_black = tk.Button(self.root, text="Black Turn", command=self.play_black_turn)
        self.button_white.pack()
        self.button_black.pack()

    def play_white_turn(self):
        if self.gs.white_to_move:
            legal_moves = ChessEngine.GameState.generate_future_positions(self.gs.board)
            random_move = random.choice(legal_moves)
            self.gs.board = random_move
            self.gs.white_to_move=False
            self.update_board()

    def play_black_turn(self):
        if not self.gs.white_to_move:
            best_move, _ = ChessEngine.GameState.minimax(self.gs.board, depth=3, maximizing_player=True)
            self.gs.board = best_move
            self.gs.white_to_move=True
            self.update_board()

    def update_board(self):
        self.boardImage = fenToImage(
            fen=self.gs.board,
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
