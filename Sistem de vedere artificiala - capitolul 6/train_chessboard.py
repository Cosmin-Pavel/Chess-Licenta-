from ultralytics import YOLO
import torch

def main():
    # Load a YOLOv8 model (choose a pretrained model to start with)
    model = YOLO('yolov8n.pt')
    torch.cuda.empty_cache()
    # Train the model on your custom data
    model.train(data='corner-dataset/data.yaml', epochs=100, imgsz=640, device='cuda', amp= False, patience=5)


if __name__ == '__main__':
    main()
