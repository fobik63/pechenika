from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',
        epochs=100,       # количество эпох
        imgsz=640,        # размер изображения
        batch=16,
        workers=4,        # отключаем многопоточку (много проблем с этой фигнёй)
        name='cookie_detector',
        patience=50       # остановка, если нет улучшений 50 эпох подряд
    )