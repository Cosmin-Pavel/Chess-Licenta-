import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk


class CameraProcessor:
    def __init__(self, camera_index=2, model_paths=None):
        self.padding = 10
        self.fen_string = ""
        self.robot_detected = False
        self.robot_count = 0
        self.robot_message = ""
        self.camera_index = camera_index
        self.model_corners = YOLO(model_paths['corners'])
        self.model_pieces = YOLO(model_paths['pieces'])
        self.model_turn = YOLO(model_paths['turn'])
        self.chessboard_image = None
        self.annotated_frame = None  # Initialize annotated_frame here
        self.model_corners.to('cuda')
        self.model_pieces.to('cuda')
        self.model_turn.to('cuda')
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.width, self.height = 300, 400
        self.dst_points = np.array([
            [0, 0],
            [self.width - 1, 0],
            [self.width - 1, self.height - 1],
            [0, self.height - 1]
        ], dtype="float32")

    def get_camera_content(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None

    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

    def draw_chessboard(self, fen, square_size=50, dark_color="#D18B47", light_color="#FFCE9E"):
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

        # Draw the chessboard squares
        for i in range(8):
            for j in range(8):
                y, x = i * square_size, j * square_size
                color = light_color if (i + j) % 2 == 0 else dark_color
                cv2.rectangle(chessboard_image, (x, y), (x + square_size, y + square_size), color, -1)

        # Print the FEN for debugging
        print(f"FEN: {fen}")

        # Draw the pieces
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
                                chessboard_image[y:y + square_size, x:x + square_size, c] * (
                                            1.0 - piece_img[:, :, 3] / 255.0)
                        col += 1

        return chessboard_image

    def construct_fen(self, chessboard_matrix):
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

    def order_points_tuples(self, pts):
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

    def process_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        results_corners = self.model_corners(gray_frame_3ch)
        centers = []

        boxes_with_confidence = [(box.xyxy[0], box.conf) for result in results_corners for box in result.boxes]
        sorted_boxes = sorted(boxes_with_confidence, key=lambda x: x[1], reverse=True)

        for i in range(min(4, len(sorted_boxes))):
            x1, y1, x2, y2 = map(int, sorted_boxes[i][0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            centers.append((center_x, center_y))

        if len(centers) == 4:
            corners = self.order_points_tuples(centers)
            src_points = np.array(corners, dtype="float32")
            M = cv2.getPerspectiveTransform(src_points, self.dst_points)

            if np.linalg.cond(M) < 1 / np.finfo(M.dtype).eps:
                M_inv = np.linalg.inv(M)
                warped_image = cv2.warpPerspective(frame, M, (self.width, self.height))

                grid_rows = 8
                grid_cols = 8
                chessboard_grid = [[None] * grid_cols for _ in range(grid_rows)]

                for i in range(grid_rows):
                    for j in range(grid_cols):
                        top_left = (int(self.width * j / grid_cols), int(self.height * i / grid_rows))
                        top_right = (int(self.width * (j + 1) / grid_cols), int(self.height * i / grid_rows))
                        bottom_right = (int(self.width * (j + 1) / grid_cols), int(self.height * (i + 1) / grid_rows))
                        bottom_left = (int(self.width * j / grid_cols), int(self.height * (i + 1) / grid_rows))

                        top_left_original = cv2.perspectiveTransform(np.array([[top_left]], dtype='float32'), M_inv)[0][
                            0]
                        top_right_original = \
                        cv2.perspectiveTransform(np.array([[top_right]], dtype='float32'), M_inv)[0][0]
                        bottom_right_original = \
                        cv2.perspectiveTransform(np.array([[bottom_right]], dtype='float32'), M_inv)[0][0]
                        bottom_left_original = \
                        cv2.perspectiveTransform(np.array([[bottom_left]], dtype='float32'), M_inv)[0][0]

                        chessboard_grid[i][j] = (
                        top_left_original, top_right_original, bottom_right_original, bottom_left_original)

                all_squares = np.zeros((self.height + 7 * self.padding, self.width + 7 * self.padding, 3),
                                       dtype=np.uint8)

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

                        resized_square_img = cv2.resize(square_img,
                                                        (int(self.width / grid_cols), int(self.height / grid_rows)))


                        all_squares[i * (int(self.height / grid_rows) + self.padding):(i + 1) * int(
                            self.height / grid_rows) + i * self.padding,
                        j * (int(self.width / grid_cols) + self.padding):(j + 1) * int(
                            self.width / grid_cols) + j * self.padding] = resized_square_img

                        results_pieces = self.model_pieces(resized_square_img)

                        if isinstance(results_pieces, list):
                            results = results_pieces[0]

                        class_names = results.names
                        top_class_index = results.probs.top1
                        top_class_confidence = results.probs.top1conf.item()

                        max_class_name = class_names[top_class_index]

                        chessboard_matrix[i][j] = max_class_name

                fen_part = self.construct_fen(chessboard_matrix)
                self.chessboard_image = self.draw_chessboard(fen_part, dark_color="#8B4513", light_color="#DEB887")
                self.fen_string = fen_part

        results_turn = self.model_turn(gray_frame_3ch)
        robot_found = False
        for result in results_turn:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = box.cls[0]
                class_name = self.model_turn.names[int(class_id)]

                if confidence > 0.0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    if class_name == "Robot":
                        robot_found = True

        if robot_found and not self.robot_detected:
            self.robot_count += 1
            self.robot_message += f"Robot found {self.robot_count} times\n"
            self.robot_detected = True
        elif not robot_found:
            self.robot_detected = False

        self.annotated_frame = frame

    def get_annotated_frame(self):
        return self.annotated_frame

    def get_robot_message(self):
        return self.robot_message

    def get_fen_string(self):
        return self.fen_string

    def run(self, label):
        def update_label():
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                label.imgtk = imgtk
                label.config(image=imgtk)
            label.after(10, update_label)

        update_label()
