"""Three-class ONNX cookie clicker. Space pauses and resumes the bot."""

from __future__ import annotations

import argparse
import ast
import ctypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np
import onnxruntime as ort
import pyautogui


CLASSES = ("game", "ads_close", "upgrade")
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "best.onnx"


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    box: tuple[int, int, int, int]  # Screen x1, y1, x2, y2.


def click_point(detection: Detection) -> tuple[int, int]:
    x1, y1, x2, y2 = detection.box
    if detection.name == "ads_close" and (x2 - x1 > 80 or y2 - y1 > 80):
        return max(x1, x2 - 10), min(y2, y1 + 10)
    return (x1 + x2) // 2, (y1 + y2) // 2


def choose_target(detections: list[Detection], last_upgrade: float,
                  now: float, upgrade_cooldown: float) -> Detection | None:
    for name in ("ads_close", "upgrade", "game"):
        if name == "upgrade" and now - last_upgrade < upgrade_cooldown:
            continue
        target = max((item for item in detections if item.name == name),
                     key=lambda item: item.confidence, default=None)
        if target is not None:
            return target
    return None


def space_down() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000)


def status(message: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(message)


def toggle_pause(paused: bool, was_down: bool, down: bool) -> tuple[bool, bool]:
    if down and not was_down:
        paused = not paused
        status("⏸ Пауза" if paused else "▶ Работаем...")
    return paused, down


class Detector:
    def __init__(self, model: Path, confidence: float):
        if not model.is_file():
            raise FileNotFoundError(f"Модель не найдена: {model}")
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4 or shape[1] != 3 or shape[2] != shape[3] or not isinstance(shape[2], int):
            raise ValueError(f"Ожидался квадратный вход ONNX NCHW: {shape}")
        self.size = shape[2]
        self.confidence = confidence
        names = self.session.get_modelmeta().custom_metadata_map.get("names", "")
        try:
            parsed = ast.literal_eval(names)
            actual = tuple(parsed[i] for i in range(len(CLASSES)))
        except (ValueError, SyntaxError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"В ONNX нет корректных имён классов: {names}") from exc
        if actual != CLASSES or len(parsed) != len(CLASSES):
            raise ValueError(f"Классы ONNX {parsed} не совпадают с {CLASSES}")

    def detect(self, image: np.ndarray, monitor: dict) -> list[Detection]:
        height, width = image.shape[:2]
        scale = min(self.size / width, self.size / height)
        new_width, new_height = round(width * scale), round(height * scale)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[top:top + new_height, left:left + new_width] = cv2.resize(image, (new_width, new_height))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(f"Неожиданный выход ONNX: {output.shape}")
        rows = output[0]
        if rows.shape[0] == 4 + len(CLASSES):
            rows = rows.T
        elif rows.shape[1] != 4 + len(CLASSES):
            raise ValueError(f"ONNX должен выдавать {len(CLASSES)} класса: {output.shape}")

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in rows:
            class_id = int(np.argmax(row[4:]))
            score = float(row[4 + class_id])
            if score < self.confidence:
                continue
            cx, cy, box_width, box_height = map(float, row[:4])
            x1 = max(0, min(width - 1, round((cx - box_width / 2 - left) / scale)))
            y1 = max(0, min(height - 1, round((cy - box_height / 2 - top) / scale)))
            x2 = max(0, min(width - 1, round((cx + box_width / 2 - left) / scale)))
            y2 = max(0, min(height - 1, round((cy + box_height / 2 - top) / scale)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        detections = []
        for class_id in set(class_ids):
            indices = [i for i, value in enumerate(class_ids) if value == class_id]
            kept = cv2.dnn.NMSBoxes([boxes[i] for i in indices], [scores[i] for i in indices],
                                    self.confidence, 0.45)
            for local_index in np.array(kept).reshape(-1):
                index = indices[int(local_index)]
                x, y, box_width, box_height = boxes[index]
                screen_x, screen_y = monitor["left"] + x, monitor["top"] + y
                detections.append(Detection(CLASSES[class_id], scores[index],
                                            (screen_x, screen_y, screen_x + box_width, screen_y + box_height)))
        return detections


def run(model: Path, confidence: float, monitor_index: int, upgrade_cooldown: float) -> None:
    detector = Detector(model, confidence)
    pyautogui.FAILSAFE = True
    paused = False
    was_down = False
    last_upgrade = -float("inf")
    status("▶ Работаем... Пробел — пауза/продолжение; Ctrl+C — выход.")
    with mss.mss() as capture:
        if not 0 <= monitor_index < len(capture.monitors):
            raise ValueError(f"Недоступный монитор {monitor_index}; выберите 0..{len(capture.monitors) - 1}")
        monitor = capture.monitors[monitor_index]
        while True:
            paused, was_down = toggle_pause(paused, was_down, space_down())
            if paused:
                time.sleep(0.08)
                continue
            frame = np.asarray(capture.grab(monitor))[:, :, :3]
            target = choose_target(detector.detect(frame, monitor), last_upgrade,
                                   time.monotonic(), upgrade_cooldown)
            paused, was_down = toggle_pause(paused, was_down, space_down())
            if target is not None and not paused:
                pyautogui.click(*click_point(target))
                if target.name == "upgrade":
                    last_upgrade = time.monotonic()
                if target.name == "ads_close":
                    deadline = time.monotonic() + 1.5
                    while time.monotonic() < deadline:
                        paused, was_down = toggle_pause(paused, was_down, space_down())
                        time.sleep(0.05)
            time.sleep(0.08)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--upgrade-cooldown", type=float, default=4.0)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or args.upgrade_cooldown < 0:
        parser.error("confidence must be in (0, 1]; upgrade cooldown must be nonnegative")
    try:
        run(args.model, args.confidence, args.monitor, args.upgrade_cooldown)
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        status("Остановлено.")


if __name__ == "__main__":
    main()
