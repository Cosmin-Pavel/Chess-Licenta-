from ultralytics import YOLO

def main():
    model = YOLO("runs/classify/train29/weights/best.pt")


    metrics = model.val()
    metrics.top1
    metrics.top5


if __name__ == '__main__':
    main()