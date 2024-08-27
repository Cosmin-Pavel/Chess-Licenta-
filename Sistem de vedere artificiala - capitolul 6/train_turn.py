import multiprocessing
from multiprocessing import freeze_support

from ultralytics import YOLO


def main():
    # Load a YOLOv8 model (choose a pretrained model to start with)
    model = YOLO('yolov8n.pt')

    # Train the model on your custom data
    model.train(data='logo-detect.v5'
                     '/data.yaml', epochs=200, imgsz=640, device='cuda', amp=False, patience=5)


if __name__ == '__main__':
    main()