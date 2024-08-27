import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO('runs/detect/train42/weights/best.pt')
model.to('cuda')


cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    filtered_frame_stacked = np.stack((blurred_frame,) * 3, axis=-1)

    results = model(filtered_frame_stacked)

    filtered_frame_display = cv2.cvtColor(blurred_frame, cv2.COLOR_GRAY2BGR)

    centers = []


    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = box.cls[0]
                class_name = model.names[int(class_id)]

                cv2.rectangle(filtered_frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f'{class_name}: {confidence:.2f}'
                cv2.putText(filtered_frame_display, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centers.append((center_x, center_y))

    cv2.imshow('YOLOv8 Detection', filtered_frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
