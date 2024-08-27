import cv2
import numpy as np
from ultralytics import YOLO


model_corners = YOLO('runs/detect/train10/weights/best.pt')
model_corners.to('cuda')

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

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


width, height = 800, 800
dst_points = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

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


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(centers) == 4:
        corners = order_points_tuples(centers)
        src_points = np.array(corners, dtype="float32")
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        if np.linalg.cond(M) < 1 / np.finfo(M.dtype).eps:
            M_inv = np.linalg.inv(M)
            warped_image = cv2.warpPerspective(frame, M, (width, height))


            for i in range(9):
                x = i * (width // 8)
                y = i * (height // 8)
                cv2.line(warped_image, (x, 0), (x, height), (255, 255, 255), 1)
                cv2.line(warped_image, (0, y), (width, y), (255, 255, 255), 1)


            grid_points = []
            for i in range(9):
                for j in range(9):
                    grid_points.append((i * (width // 8), j * (height // 8)))


            for point in grid_points:
                cv2.circle(warped_image, point, 3, (0, 0, 255), -1)


            grid_points_original = cv2.perspectiveTransform(np.array([grid_points], dtype='float32'), M_inv)[0]


            for i in range(9):
                pt1 = tuple(map(int, grid_points_original[i * 9]))
                pt2 = tuple(map(int, grid_points_original[(i + 1) * 9 - 1]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)
            for i in range(9):
                pt1 = tuple(map(int, grid_points_original[i]))
                pt2 = tuple(map(int, grid_points_original[i + 72]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

            cv2.imshow('Warped Chessboard', warped_image)


    height, width = frame.shape[:2]
    if height > width:
        new_height = 800
        new_width = int((width / height) * 800)
    else:
        new_width = 800
        new_height = int((height / width) * 800)
    frame_resized = cv2.resize(frame, (new_width, new_height))

    cv2.imshow('YOLOv8 Corner Detection', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
