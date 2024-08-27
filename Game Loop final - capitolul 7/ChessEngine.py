import chess
import torch
from model import fen_to_bit_vector


class GameState:
    def __init__(self):
        self.board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.board = chess.Board(self.board_fen)
        self.white_to_move = True
        self.is_playing_Black = True;

    @staticmethod
    def evaluate_fen(fen):
        piece_values = {
            'P': 10, 'N': 30, 'B': 30, 'R': 50, 'Q': 90, 'K': 0,
            'p': -10, 'n': -30, 'b': -30, 'r': -50, 'q': -90, 'k': 0
        }

        parts = fen.split()

        if len(parts) != 6:
            return 0  # Invalid FEN format

        board_layout, turn, castling, en_passant, half_moves, full_moves = parts

        board_score = 0

        for char in board_layout:
            piece_value = piece_values.get(char, 0)
            board_score += piece_value

        if turn == 'w':
            board_score -= 20
        else:
            board_score += 20

        if 'K' in castling:
            board_score -= 10
        if 'k' in castling:
            board_score += 10


        min_score = -540
        max_score = 540

        normalized_score = (board_score - min_score) / (max_score - min_score)
        return normalized_score

    @staticmethod
    def generate_future_positions(fen):

        board = chess.Board(fen)
        future_positions = []

        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            future_positions.append((board_copy.fen(), move))

        return future_positions

    @staticmethod
    def minimax(fen, depth, maximizing_player=True):
        if depth == 0:
            return fen, GameState.evaluate_fen(fen)

        if maximizing_player:
            best_move = None
            best_value = float('-inf')

            for future_fen in GameState.generate_future_positions(fen):
                _, value = GameState.minimax(future_fen, depth - 1, False)
                if value > best_value:
                    best_value = value
                    best_move = future_fen

            return best_move, best_value
        else:
            best_move = None
            best_value = float('inf')

            for future_fen in GameState.generate_future_positions(fen):
                _, value = GameState.minimax(future_fen, depth - 1, True)
                if value < best_value:
                    best_value = value
                    best_move = future_fen

            return best_move, best_value

    @staticmethod
    def minimax_alpha_beta(fen, depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True,
                           is_playing_Black=True):
        if depth == 0 or GameState.generate_future_positions(fen) == []:
            score = GameState.evaluate_fen(fen)
            if is_playing_Black:
                score = -score
            return fen, None, score

        if maximizing_player:
            best_move = None
            best_value = float('-inf')

            for future_fen, move in GameState.generate_future_positions(fen):
                _, _, value = GameState.minimax_alpha_beta(future_fen, depth - 1, alpha, beta, False, is_playing_Black)
                if value > best_value:
                    best_value = value
                    best_move = move

                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Pruning
            return fen, best_move, best_value
        else:
            best_move = None
            best_value = float('inf')

            for future_fen, move in GameState.generate_future_positions(fen):
                _, _, value = GameState.minimax_alpha_beta(future_fen, depth - 1, alpha, beta, True, is_playing_Black)
                if value < best_value:
                    best_value = value
                    best_move = move

                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Pruning
            return fen, best_move, best_value


    @staticmethod
    def model_alpha_beta(fen, depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, model=None,
                            is_playing_Black=True):
        if depth == 0 or GameState.generate_future_positions(fen) == []:
            score_board = GameState.evaluate_board(fen, model)
            score_fen = GameState.evaluate_fen(fen)
            score = 0.5 * score_board + 0.5 * score_fen


            if is_playing_Black:
                score = -score
            return fen, None, score

        if maximizing_player:
            best_move = None
            best_value = float('-inf')

            for future_fen, move in GameState.generate_future_positions(fen):
                _, _, value = GameState.model_alpha_beta(future_fen, depth - 1, alpha, beta, False, model,
                                                          is_playing_Black)
                if value > best_value:
                    best_value = value
                    best_move = move

                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Pruning
            return fen, best_move, best_value
        else:
            best_move = None
            best_value = float('inf')

            for future_fen, move in GameState.generate_future_positions(fen):
                _, _, value = GameState.model_alpha_beta(future_fen, depth - 1, alpha, beta, True, model,
                                                             is_playing_Black)
                if value < best_value:
                    best_value = value
                    best_move = move

                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Pruning
            return fen, best_move, best_value

    def evaluate_board(fen, model):

        input_tensor = torch.tensor(fen_to_bit_vector(fen), dtype=torch.float32)
        input_tensor = input_tensor.to(next(model.parameters()).device)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        value = output.item()

        return value