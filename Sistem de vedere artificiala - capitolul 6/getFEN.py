import cv2
import numpy as np
from ultralytics import YOLO


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


def draw_chessboard(fen, square_size=50, dark_color="#D18B47", light_color="#FFCE9E"):
    dark_color = hex_to_bgr(dark_color)
    light_color = hex_to_bgr(light_color)

    board_size = square_size * 8
    chessboard_image = np.zeros((board_size, board_size, 3), dtype=np.uint8)

    piece_paths = {
        'P': 'images/white/Pawn.png', 'R': 'images/white/Rook.png', 'N': 'images/white/Knight.png', 'B': 'images/white/Bishop.png', 'Q': 'images/white/Queen.png', 'K': 'images/white/King.png',
        'p': 'images/black/Pawn.png', 'r': 'images/black/Rook.png', 'n': 'images/black/Knight.png', 'b': 'images/black/Bishop.png', 'q': 'images/black/Queen.png', 'k': 'images/black/King.png'
    }

    for i in range(8):
        for j in range(8):
            y, x = i * square_size, j * square_size
            color = light_color if (i + j) % 2 == 0 else dark_color
            cv2.rectangle(chessboard_image, (x, y), (x + square_size, y + square_size), color, -1)

    rows = fen.split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                piece_path = piece_paths[char]
                piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
                if piece_img is None:
                    continue
                piece_img = cv2.resize(piece_img, (square_size, square_size))
                y, x = i * square_size, col * square_size
                for c in range(3):
                    chessboard_image[y:y + square_size, x:x + square_size, c] = \
                        piece_img[:, :, c] * (piece_img[:, :, 3] / 255.0) + \
                        chessboard_image[y:y + square_size, x:x + square_size, c] * (1.0 - piece_img[:, :, 3] / 255.0)
                col += 1

    return chessboard_image


def construct_fen(chessboard_matrix):
    piece_map = {
        "WP": "P", "WR": "R", "WN": "N", "WB": "B", "WQ": "Q", "WK": "K",
        "BP": "p", "BR": "r", "BN": "n", "BB": "b", "BQ": "q", "BK": "k"
    }

    fen_rows = []
    for row in chessboard_matrix:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == "0":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece_map.get(cell, cell)  # Map piece or use cell value if not found
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    fen = "/".join(fen_rows)
    return fen

def order_points_tuples(pts):
    pts_np = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_np.sum(axis=1)
    rect[0] = pts_np[np.argmin(s)]
    rect[2] = pts_np[np.argmax(s)]
    diff = np.diff(pts_np, axis=1)
    rect[1] = pts_np[np.argmin(diff)]
    rect[3] = pts_np[np.argmax(diff)]
    ordered_points = [tuple(point) for point in rect]
    return ordered_points


model_corners = YOLO('runs/detect/train3/weights/best.pt')
model_pieces = YOLO('runs_pieces_last/detect/train/weights/best.pt')
model_corners.to('cuda')
model_pieces.to('cuda')


image_path = "96.jpg"
frame = cv2.imread(image_path)


width, height = 300, 400
dst_points = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")


results_corners = model_corners(frame)

centers = []


for result in results_corners:
    for box in result.boxes:
        if len(centers) >= 4:
            break


        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        centers.append((center_x, center_y))

if len(centers) == 4:
    corners = order_points_tuples(centers)

    src_points = np.array(corners, dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    if np.linalg.cond(M) < 1 / np.finfo(M.dtype).eps:
        M_inv = np.linalg.inv(M)

        warped_image = cv2.warpPerspective(frame, M, (width, height))

        for i in range(len(dst_points)):
            corner = dst_points[i]
            corner_int = tuple(map(int, corner))
            cv2.circle(warped_image, corner_int, 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(map(int, corners[i])), 5, (0, 255, 0), -1)
            next_corner = dst_points[(i + 1) % len(dst_points)]
            next_corner_int = tuple(map(int, next_corner))
            cv2.line(warped_image, corner_int, next_corner_int, (0, 255, 0), 2)
            cv2.line(frame, tuple(map(int, corners[i])), tuple(map(int, corners[(i + 1) % len(corners)])), (0, 255, 0), 2)

        grid_rows = 8
        grid_cols = 8
        chessboard_grid = [[None] * grid_cols for _ in range(grid_rows)]

        for i in range(grid_rows):
            for j in range(grid_cols):
                top_left = (int(width * j / grid_cols), int(height * i / grid_rows))
                top_right = (int(width * (j + 1) / grid_cols), int(height * i / grid_rows))
                bottom_right = (int(width * (j + 1) / grid_cols), int(height * (i + 1) / grid_rows))
                bottom_left = (int(width * j / grid_cols), int(height * (i + 1) / grid_rows))
                top_left_original = cv2.perspectiveTransform(np.array([[top_left]], dtype='float32'), M_inv)[0][0]
                top_right_original = cv2.perspectiveTransform(np.array([[top_right]], dtype='float32'), M_inv)[0][0]
                bottom_right_original = cv2.perspectiveTransform(np.array([[bottom_right]], dtype='float32'), M_inv)[0][0]
                bottom_left_original = cv2.perspectiveTransform(np.array([[bottom_left]], dtype='float32'), M_inv)[0][0]
                chessboard_grid[i][j] = (top_left_original, top_right_original, bottom_right_original, bottom_left_original)


        for i in range(grid_rows):
            for j in range(grid_cols):
                square_coords = np.array(chessboard_grid[i][j], np.int32)
                square_coords = square_coords.reshape((-1, 1, 2))
                cv2.polylines(frame, [square_coords], True, (0, 255, 0), 2)


chessboard_matrix = [["0" for _ in range(8)] for _ in range(8)]



results_pieces = model_pieces(frame)


for result in results_pieces:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = model_pieces.names[int(class_id)]

        if confidence > 0.8:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            for i in range(grid_rows):
                for j in range(grid_cols):
                    tl, tr, br, bl = chessboard_grid[i][j]
                    if cv2.pointPolygonTest(np.array([tl, tr, br, bl]), (center_x, center_y), False) >= 0:
                        chessboard_matrix[i][j] = class_name


print("Chessboard Matrix:")

for row in chessboard_matrix:
    print(row)

fen_part = construct_fen(chessboard_matrix)
print(fen_part)


chessboard_image = draw_chessboard(fen_part)


cv2.imshow('Chessboard', chessboard_image)
cv2.waitKey(0)
cv2.destroyAllWindows()