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
        self.gs.board = chess.Board(self.gs.board)  # Conversia stringului FEN la obiect chess.Board
        self.drag_data = {}
        self.create_ui()

    def create_ui(self):
        # Inițializarea interfeței grafice
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

        # Asocierea evenimentelor de mouse pentru funcționalitatea drag-and-drop
        self.label.bind("<ButtonPress-1>", self.on_drag_start)
        self.label.bind("<ButtonRelease-1>", self.on_drag_end)

        # Lansarea buclei principale a jocului
        self.game_loop()

    def on_drag_start(self, event):
        # Funcția apelată la începerea tragerii unei piese
        col, row = event.x // 80, event.y // 80
        piece = self.gs.board.piece_at(chess.square(col, 7 - row))
        if piece and piece.color == chess.WHITE and self.gs.white_to_move:
            self.drag_data = {'piece': piece,
                              'start_col': col,
                              'start_row': row}

    def on_drag_end(self, event):
        # Funcția apelată la sfârșitul tragerii piesei
        if self.drag_data:
            col, row = event.x // 80, event.y // 80
            start_square = chess.square(self.drag_data['start_col'], 7 - self.drag_data['start_row'])
            end_square = chess.square(col, 7 - row)
            move = chess.Move(start_square, end_square)
            if move in self.gs.board.legal_moves:
                self.gs.board.push(move)
                self.gs.white_to_move = not self.gs.white_to_move  # Schimbarea jucătorului

                # Actualizarea interfeței
                self.update_board()

            self.drag_data = {}  # Ștergerea datelor de trageri

    def game_loop(self):
        # Buclează până când jocul se încheie
        while not self.gs.board.is_game_over():
            if self.gs.white_to_move:
                # Rândul jucătorului alb (umanoizid)
                self.root.update()
                self.root.after(100)  # Mică întârziere pentru a permite actualizarea GUI-ului
            else:
                # Rândul jucătorului negru (computer)
                self.play_black_turn()
                self.root.update()
                self.root.after(100)  # Mică întârziere pentru a permite actualizarea GUI-ului

        print("Joc încheiat")

    def play_black_turn(self):
        # Funcția pentru rândul jucătorului negru
        if not self.gs.white_to_move:
            start_time = time.time()  # Înregistrarea momentului de început
            best_move, _ = ChessEngine.GameState.minimax_alpha_beta(
                self.gs.board.fen(), depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
            end_time = time.time()  # Înregistrarea momentului de sfârșit

            # Actualizarea stării și a interfeței după mișcarea calculată
            self.gs.board.set_fen(best_move)
            self.gs.white_to_move = not self.gs.white_to_move
            self.update_board()

    def update_board(self, drag_data=None):
        # Actualizarea tablei de șah afișate
        if drag_data:
            piece = drag_data['piece']
            start_col, start_row = drag_data['start_col'], drag_data['start_row']
            current_col, current_row = drag_data['current_col'], drag_data['current_row']

            # Efectuarea mișcării pe tablă
            move = chess.Move.from_uci(f"{start_col}{7 - start_row}{current_col}{7 - current_row}")
            self.gs.board.push(move)

        # Actualizarea imaginii afișate
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
        # Lansarea buclei principale a aplicației
        self.root.mainloop()

    def run_experiment(self, num_moves=50):
        # Deschiderea și scrierea în fișiere separate pentru Minimax și Alpha-Beta Pruning
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
        # Funcție pentru efectuarea unei mutări aleatoare pentru jucătorul alb
        legal_moves = list(self.gs.board.legal_moves)
        random_move = random.choice(legal_moves)
        self.gs.board.push(random_move)
        self.gs.white_to_move = not self.gs.white_to_move

    def make_black_move_minimax(self):
        # Funcție pentru efectuarea mișcării jucătorului negru cu Minimax
        if not self.gs.white_to_move:
            best_move, _ = ChessEngine.GameState.minimax(self.gs.board.fen(), depth=3, maximizing_player=True)
            self.gs.board.set_fen(best_move)
            self.gs.white_to_move = not self.gs.white_to_move

    def make_black_move_alpha_beta(self):
        # Funcție pentru efectuarea mișcării jucătorului negru cu Alpha-Beta Pruning
        if not self.gs.white_to_move:
            best_move, _ = ChessEngine.GameState.minimax_alpha_beta(self.gs.board.fen(), depth=3, alpha=float('-inf'),
                                                                    beta=float('inf'), maximizing_player=True)
            self.gs.board.set_fen(best_move)
            self.gs.white_to_move = not self.gs.white_to_move

if __name__ == "__main__":
    game = ChessGame()
    game.run()
    #game.run_experiment(num_moves=30)
