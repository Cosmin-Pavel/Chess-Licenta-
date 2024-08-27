import chess
import cv2
import torch
from model_arch import Net
import ChessGame
import tkinter as tk
from PIL import Image, ImageTk
import threading
from camera import CameraProcessor
import paho.mqtt.client as mqtt
from queue import Queue


# MQTT settings
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC1 = "test/tuiasi/robot/feedback"
TOPIC2 = "test/tuiasi/robot/DoTaskCmd"
client_is_connected = False
move_in_progress = False
move_queue = Queue()


def on_connect(client, userdata, flags, rc):
    global client_is_connected
    print("Connected to the Broker")
    client.subscribe(TOPIC1)
    client_is_connected = True


def on_message(client, userdata, msg):
    global move_in_progress
    print(f"Received msg: {msg.payload.decode()} from {msg.topic}")
    if msg.payload.decode() == "true":
        move_in_progress = False
        if not move_queue.empty():
            next_move = move_queue.get()
            client.publish(TOPIC2, next_move)
            move_in_progress = True


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_start()


def calculeaza_coordonate(pozitii: str):
    litere = {'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e': 3, 'f': 2, 'g': 1, 'h': 0}
    cifre = {'8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7}

    if(pozitii == "turn"):
        coord2_x = -2.46
        coord2_y = 2.4
        coord1_x = -2.46
        coord1_y = 4.4
    else:
        prima_pozitie = pozitii[:2]
        a_doua_pozitie = pozitii[2:]
        coord1_x = litere[prima_pozitie[0].lower()]
        coord1_y = cifre[prima_pozitie[1]]

        if a_doua_pozitie == "outside":
            coord2_x = 9
            coord2_y = 3
        else:
            coord2_x = litere[a_doua_pozitie[0].lower()]
            coord2_y = cifre[a_doua_pozitie[1]]

    rezultat = f"{coord1_x} {coord1_y} {coord2_x} {coord2_y}"
    return rezultat


def parse_fen(fen):
    rows = fen.split(' ')[0].split('/')
    board = []
    for row in rows:
        parsed_row = []
        for char in row:
            if (char.isdigit()):
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


def setup_ui(root, board_image, camera_processor, chessGame):
    # Configure the grid layout for 4 equal sections
    for i in range(2):
        root.grid_rowconfigure(i, weight=1, minsize=400)
        root.grid_columnconfigure(i, weight=1, minsize=400)

    photo = ImageTk.PhotoImage(board_image)

    label_section1 = tk.Label(root, image=photo)
    label_section1.image = photo
    label_section1.grid(row=0, column=0, sticky="nsew")

    section2_frame = tk.Frame(root, bg="lightgray")
    section2_frame.grid(row=0, column=1, sticky="nsew")

    section2_label = tk.Label(section2_frame)
    section2_label.pack(expand=True, fill="both")

    section3_label = tk.Label(root)
    section3_label.grid(row=1, column=0, sticky="nsew")

    section4 = tk.Frame(root, bg="lightgray")
    section4.grid(row=1, column=1, sticky="nsew")

    section4_text = tk.Text(section4, bg="lightgray")
    section4_text.pack(expand=True, fill="both")

    def update_camera_content():
        annotated_frame = camera_processor.get_annotated_frame()
        if annotated_frame is not None:
            camera_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            label_width = section2_label.winfo_width()
            label_height = section2_label.winfo_height()
            camera_image.thumbnail((label_width, label_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(camera_image)

            section2_label.config(image=photo)
            section2_label.image = photo

        chessboard_image = camera_processor.chessboard_image
        if chessboard_image is not None:
            chessboard_image_pil = Image.fromarray(chessboard_image)
            chessboard_photo = ImageTk.PhotoImage(chessboard_image_pil)
            section3_label.config(image=chessboard_photo)
            section3_label.image = chessboard_photo

        # Update the board image in Section 1
        current_board_image = chessGame.get_board_image()
        current_board_photo = ImageTk.PhotoImage(current_board_image)
        label_section1.config(image=current_board_photo)
        label_section1.image = current_board_photo

    def resize_handler(event):
        update_camera_content()

    section2_label.bind("<Configure>", resize_handler)

    return update_camera_content, section4_text, label_section1


def process_frames(camera_processor, root, update_camera_content, section4_text, chessGame):
    global move_in_progress
    previous_robot_count = camera_processor.robot_count

    while True:
        ret, frame = camera_processor.cap.read()
        if ret:
            camera_processor.process_frame(frame)
            root.after(10, update_camera_content)
            previous_fen = chessGame.gs.board.fen()

            if camera_processor.robot_count > previous_robot_count:
                previous_robot_count = camera_processor.robot_count
                fen_string = camera_processor.get_fen_string()
                start_square, end_square = get_move(previous_fen, fen_string)

                if start_square and end_square:
                    print("Start: ", start_square, "end: ", end_square)

                    move_notation = (start_square + end_square).lower()
                    section4_text.insert(tk.END, f"White move: {move_notation}\n")
                    section4_text.see(tk.END)

                    move = chess.Move.from_uci(move_notation)

                    if move in chessGame.gs.board.legal_moves:
                        chessGame.gs.board.push(move)
                        chessGame.update_board(move)
                        chessGame.gs.white_to_move = False

                        previous_fen = chessGame.gs.board.fen()
                        previous_board = chessGame.gs.board

                        # Play black's turn
                        chessGame.play_black_turn(model)

                        fen_string_after_black = chessGame.gs.board.fen()
                        black_start_square, black_end_square = get_move(previous_fen, fen_string_after_black)

                        if black_start_square and black_end_square:
                            move_coords = calculeaza_coordonate(black_start_square + black_end_square)
                            print("previous_fen:", previous_fen)
                            print("move:", chess.Move.from_uci(black_start_square + black_end_square))
                            if chess.Board(previous_fen).is_capture(chess.Move.from_uci(black_start_square + black_end_square)):
                                section4_text.insert(tk.END, f"Black move: {black_end_square} outside\n")
                                move_queue.put(calculeaza_coordonate(black_end_square + "outside"))

                            section4_text.insert(tk.END, f"Black move: {black_start_square}{black_end_square}\n")
                            move_queue.put(move_coords)
                            section4_text.insert(tk.END, f"Black move: turn \n")
                            move_queue.put(calculeaza_coordonate("turn"))
                            section4_text.see(tk.END)

                            if client_is_connected:
                                if not move_in_progress:
                                    next_move = move_queue.get()
                                    client.publish(TOPIC2, next_move)
                                    move_in_progress = True

model = Net()

print(model)

# Step 3: Load the state dictionary
state_dict = torch.load('D:\\PycharmProjects\\Game Loop final - capitolul 7\\runs\\position_evaluation\\train9\\chessModel.pth')
model.load_state_dict(state_dict)

print(model)

# Step 4: Set the model to evaluation mode
model.eval()


def main():
    chessGame = ChessGame.ChessGame()
    print(chessGame.gs.board.fen())
    chessGame.create_ui()

    board_image = chessGame.get_board_image()

    root = tk.Tk()
    root.title("Chess Board")
    root.geometry("800x800")

    model_paths = {
        'corners': 'runs/detect/train10/weights/best.pt',
        'pieces': 'runs/classify/train30/weights/best.pt',
        'turn': 'runs/detect/train47/weights/best.pt'
    }
    camera_processor = CameraProcessor(model_paths=model_paths)

    update_camera_content, section4_text, label_section1 = setup_ui(root, board_image, camera_processor, chessGame)

    # Start the frame processing in a separate thread
    camera_thread = threading.Thread(target=process_frames,
                                     args=(camera_processor, root, update_camera_content, section4_text, chessGame))
    camera_thread.daemon = True
    camera_thread.start()

    root.mainloop()


if __name__ == "__main__":
    main()
