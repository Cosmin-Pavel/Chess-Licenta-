import torch
from ultralytics import YOLO


def main():
    # Load a YOLOv8 model (choose a pretrained model to start with)
    model = YOLO('yolov8n-cls.pt')
    torch.cuda.empty_cache()

    data_path = "D:\PycharmProjects\CameraSetup\chess-piece-classification-v6"
    print(f"Training data path: {data_path}")

    # Train the model on your custom data
    model.train(data=data_path, epochs=10, imgsz=96, device='cuda', amp=False, patience=5, dropout=0.5, weight_decay=0.001, cos_lr=True)


if __name__ == '__main__':
    main()
