def parse_fen(fen):
    rows = fen.split(' ')[0].split('/')
    board = []
    for row in rows:
        parsed_row = []
        for char in row:
            if char.isdigit():
                parsed_row.extend([''] * int(char))
            else:
                parsed_row.append(char)
        board.append(parsed_row)
    return board

def get_move(old_fen, new_fen):
    old_board = parse_fen(old_fen)
    new_board = parse_fen(new_fen)
    start = None
    end = None

    print("Old Board:")
    for row in old_board:
        print(row)
    print("New Board:")
    for row in new_board:
        print(row)

    for r in range(8):
        for c in range(8):
            if old_board[r][c] != new_board[r][c]:
                if old_board[r][c] != '' and new_board[r][c] == '':
                    start = (r, c)
                elif old_board[r][c] == '' and new_board[r][c] != '':
                    end = (r, c)
                elif old_board[r][c] != '' and new_board[r][c] != '' and old_board[r][c] != new_board[r][c]:
                        end = (r, c)

    # Ensure both start and end are found
    if start is None or end is None:
        print(f"Start or end is None: start={start}, end={end}")
        return None, None

    print(f"Start: {start}, End: {end}")

    def index_to_square(index):
        files = 'abcdefgh'
        ranks = '87654321'
        return files[index[1]] + ranks[index[0]]

    start_square = index_to_square(start)
    end_square = index_to_square(end)

    # Castling
    if old_board[start[0]][start[1]].lower() == 'k' and abs(end[1] - start[1]) == 2:
        if end_square == 'g1' or end_square == 'g8':
            return ("e1", "g1") if start_square == "e1" else ("e8", "g8")
        elif end_square == 'c1' or end_square == 'c8':
            return ("e1", "c1") if start_square == "e1" else ("e8", "c8")

    return start_square, end_square


# Test cases
old_fen_1 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
new_fen_1 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
print(get_move(old_fen_1, new_fen_1))

old_fen_2 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
new_fen_2 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
print(get_move(old_fen_2, new_fen_2)) 

old_fen_3 = 'rnbqk1nr/pppp1ppp/8/2b1p3/4P1Q1/8/PPPP1PPP/RNB1KBNR w Qkq c6 0 1'
new_fen_3 = 'rnbqk1nr/pppp1pQp/8/2b1p3/4P3/8/PPPP1PPP/RNB1KBNR b Qkq - 0 1'
print(get_move(old_fen_3, new_fen_3))
