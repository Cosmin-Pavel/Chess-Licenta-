import cv2
import numpy as np
import os
from ultralytics import YOLO
import os
import cv2
import itertools


squares_dir = "squares"
if not os.path.exists(squares_dir):
    os.makedirs(squares_dir)


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))


def draw_chessboard(fen, square_size=50, dark_color="#D18B47", light_color="#FFCE9E"):
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

    dark_color = hex_to_bgr(dark_color)
    light_color = hex_to_bgr(light_color)
    board_size = square_size * 8
    chessboard_image = np.zeros((board_size, board_size, 3), dtype=np.uint8)

    piece_paths = {
        'P': 'images/white/Pawn.png', 'R': 'images/white/Rook.png', 'N': 'images/white/Knight.png',
        'B': 'images/white/Bishop.png', 'Q': 'images/white/Queen.png', 'K': 'images/white/King.png',
        'p': 'images/black/Pawn.png', 'r': 'images/black/Rook.png', 'n': 'images/black/Knight.png',
        'b': 'images/black/Bishop.png', 'q': 'images/black/Queen.png', 'k': 'images/black/King.png'
    }


    for i in range(8):
        for j in range(8):
            y, x = i * square_size, j * square_size
            color = light_color if (i + j) % 2 == 0 else dark_color
            cv2.rectangle(chessboard_image, (x, y), (x + square_size, y + square_size), color, -1)

    print(f"FEN: {fen}")

    rows = fen.split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                piece_path = piece_paths.get(char, None)
                if piece_path:
                    piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
                    if piece_img is None:
                        print(f"Could not load image for piece: {char} at {i},{col}")
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
        "BP": "p", "BR": "r", "BN": "n", "BB": "b", "BQ": "q", "BK": "k",
        "Background": "0"
    }

    fen_rows = []
    for row in chessboard_matrix:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == "Background":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece_map.get(cell, cell)
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


model_corners = YOLO('runs/detect/train10/weights/best.pt')
model_pieces = YOLO('runs/classify/train30/weights/best.pt')
model_turn = YOLO('runs/detect/train9/weights/best.pt')
model_corners.to('cuda')
model_pieces.to('cuda')
model_turn.to('cuda')


squares_dir = "squares"
if not os.path.exists(squares_dir):
    os.makedirs(squares_dir)

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


width, height = 400, 400
dst_points = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

padding = 15

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    results_corners = model_corners(gray_frame_3ch)
    centers = []


    boxes_with_confidence = [(box.xyxy[0], box.conf) for result in results_corners for box in result.boxes]

    sorted_boxes = sorted(boxes_with_confidence, key=lambda x: x[1], reverse=True)

    for i in range(min(4, len(sorted_boxes))):
        x1, y1, x2, y2 = map(int, sorted_boxes[i][0])
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
                    bottom_right_original = \
                    cv2.perspectiveTransform(np.array([[bottom_right]], dtype='float32'), M_inv)[0][0]
                    bottom_left_original = \
                    cv2.perspectiveTransform(np.array([[bottom_left]], dtype='float32'), M_inv)[0][0]

                    chessboard_grid[i][j] = (
                    top_left_original, top_right_original, bottom_right_original, bottom_left_original)

            all_squares = np.zeros((height + 7 * padding, width + 7 * padding, 3), dtype=np.uint8)

            chessboard_matrix = [["0" for _ in range(8)] for _ in range(8)]

            for i in range(grid_rows):
                for j in range(grid_cols):
                    square_coords = np.array(chessboard_grid[i][j], np.int32)
                    square_coords = square_coords.reshape((-1, 1, 2))

                    top_left, top_right, bottom_right, bottom_left = chessboard_grid[i][j]
                    src_points_square = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
                    dst_points_square = np.array([
                        [0, 0],
                        [top_right[0] - top_left[0], 0],
                        [top_right[0] - top_left[0], bottom_left[1] - top_left[1]],
                        [0, bottom_left[1] - top_left[1]]
                    ], dtype="float32")
                    M_square = cv2.getPerspectiveTransform(src_points_square, dst_points_square)
                    square_img = cv2.warpPerspective(frame, M_square,
                                                     (int(dst_points_square[2][0]), int(dst_points_square[2][1])))


                    resized_square_img = cv2.resize(square_img, (int(width / grid_cols), int(height / grid_rows)))

                    counter = itertools.count()

                    count = next(counter)
                    square_img_path = os.path.join(squares_dir, f"square_{i}_{j}_{count}.png")
                    cv2.imwrite(square_img_path, square_img)

                    all_squares[i * (int(height / grid_rows) + padding):(i + 1) * int(height / grid_rows) + i * padding,
                    j * (int(width / grid_cols) + padding):(j + 1) * int(
                        width / grid_cols) + j * padding] = resized_square_img

                    results_pieces = model_pieces(square_img)

                    results_pieces = model_pieces(square_img)

                    if isinstance(results_pieces, list):
                        results = results_pieces[0]

                    class_names = results.names
                    top_class_index = results.probs.top1
                    top_class_confidence = results.probs.top1conf.item()

                    max_class_name = class_names[top_class_index]

                    chessboard_matrix[i][j] = max_class_name
            print(chessboard_matrix)

            fen_part = construct_fen(chessboard_matrix)
            chessboard_image = draw_chessboard(fen_part)
            cv2.imshow('Chessboard', chessboard_image)

            cv2.imshow('All Squares', all_squares)

    results_turn = model_turn(gray_frame_3ch)
    for result in results_turn:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = model_turn.names[int(class_id)]

            if confidence > 0.8:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                text = f"{class_name}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if frame.shape[0] > 500:
        scale_factor = 500 / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), 500))

    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
