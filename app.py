import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time
import mss
import mss.tools

# Загрузка обученной модели
model = YOLO('runs/detect/cookie_detector/weights/best.pt')

def find_and_click():
    # Скриншот экрана через mss
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # весь экран (все мониторы)
        # monitor = sct.monitors[1]  # или только основной монитор
        screenshot = sct.grab(monitor)
        screenshot_np = np.array(screenshot)
        screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_BGRA2BGR)

    # Детекция
    results = model(screenshot_np)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]

            if confidence > 0.7:  # порог уверенности
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 10 Кликов за секунду
                for i in range(10):
                    pyautogui.click(center_x, center_y)
                    time.sleep(0.1)
                print(f"Клик по координатам: ({center_x}, {center_y})")
                return True

    return False

while True:
    if find_and_click():
        time.sleep(1)
    else:
        time.sleep(0.5)