import chess


class GameState():
    def __init__(self):
        self.board = "nrbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.white_to_move = True



#Functie provizorie de evaluare
    @staticmethod
    def evaluate_fen(fen):
        piece_values = {
            'P': -10, 'N': -30, 'B': -30, 'R': -50, 'Q': -90, 'K': 0,
            'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 0
        }

        # Split the FEN string into its components
        parts = fen.split()

        if len(parts) != 6:
            return 0  # Invalid FEN format

        board_layout, turn, castling, en_passant, half_moves, full_moves = parts

        board_score = 0


        for char in board_layout:
            piece_value = piece_values.get(char, 0)
            board_score += piece_value




        if turn == 'w':
            board_score += 20
        else:
            board_score -= 20


        if 'K' in castling:
            board_score += 10
        if 'k' in castling:
            board_score -= 10

        return board_score

    @staticmethod
    def generate_future_positions(fen):
        board = chess.Board(fen)
        future_positions = []

        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            future_positions.append(board_copy.fen())

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
    def minimax_alpha_beta(fen, depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        if depth == 0:
            return fen, GameState.evaluate_fen(fen)

        if maximizing_player:
            best_move = None
            best_value = float('-inf')

            for future_fen in GameState.generate_future_positions(fen):
                _, value =  GameState.minimax_alpha_beta(future_fen, depth - 1, alpha, beta, False)
                if value > best_value:
                    best_value = value
                    best_move = future_fen

                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Pruning
            return best_move, best_value
        else:
            best_move = None
            best_value = float('inf')

            for future_fen in GameState.generate_future_positions(fen):
                _, value =  GameState.minimax_alpha_beta(future_fen, depth - 1, alpha, beta, True)
                if value < best_value:
                    best_value = value
                    best_move = future_fen

                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Pruning
            return best_move, best_value