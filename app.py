"""Three-class ONNX cookie clicker. Space pauses and resumes the bot."""

from __future__ import annotations

import argparse
import ast
import ctypes
import random
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


def is_close_button_box(box: tuple[int, int, int, int],
                        frame_width: int, frame_height: int) -> bool:
    x1, y1, x2, y2 = box
    return 0 < x2 - x1 <= frame_width * 0.1 and 0 < y2 - y1 <= frame_height * 0.1


def find_overlay_close(image: np.ndarray, monitor: dict) -> Detection | None:
    """Find the white X in Yandex's grey circular button beside a modal ad."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pale = cv2.inRange(hsv, (0, 0, 200), (179, 75, 255))
    _, _, cards, _ = cv2.connectedComponentsWithStats(pale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for x, y, card_width, card_height, area in sorted(cards[1:], key=lambda row: row[4], reverse=True):
        if (card_width < width * 0.15 or card_height < height * 0.12
                or area < card_width * card_height * 0.12):
            continue
        rx1 = max(0, round(x + card_width * 0.92))
        ry1 = max(0, round(y - card_height * 0.18))
        rx2 = min(width, round(x + card_width * 1.10))
        ry2 = min(height, round(y + card_height * 0.06))
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        white = (gray[ry1:ry2, rx1:rx2] > 210).astype(np.uint8)
        _, _, marks, _ = cv2.connectedComponentsWithStats(white)
        for mx, my, mark_width, mark_height, mark_area in marks[1:]:
            if not (8 <= mark_width <= card_height * 0.10
                    and 8 <= mark_height <= card_height * 0.10
                    and 0.2 <= mark_area / (mark_width * mark_height) <= 0.7):
                continue
            cx = int(rx1 + mx + mark_width // 2)
            cy = int(ry1 + my + mark_height // 2)
            step = max(3, round(min(mark_width, mark_height) * 0.3))
            ring = max(6, round(max(mark_width, mark_height) * 0.8))
            if not (ring <= cx < width - ring and ring <= cy < height - ring):
                continue
            diagonal = [gray[cy + dy, cx + dx] for dx, dy in
                        ((-step, -step), (step, -step), (-step, step), (step, step))]
            axes = [gray[cy + dy, cx + dx] for dx, dy in
                    ((-step, 0), (step, 0), (0, -step), (0, step))]
            circle = [gray[cy + dy, cx + dx] for dx, dy in
                      ((-ring, 0), (ring, 0), (0, -ring), (0, ring))]
            if (np.mean(diagonal) > 185 and np.mean(diagonal) - np.mean(axes) > 55
                    and all(65 < value < 190 for value in circle)):
                sx = monitor["left"] + cx
                sy = monitor["top"] + cy
                return Detection("ads_close", 1.0, (sx - 2, sy - 2, sx + 2, sy + 2))
    return None


def choose_target(detections: list[Detection], affordable: list[Detection],
                  last_upgrade: float, now: float, upgrade_cooldown: float) -> Detection | None:
    ads = [item for item in detections if item.name == "ads_close"]
    if ads:
        return max(ads, key=lambda item: item.confidence)
    # In the existing model, the small X on some ad overlays is tagged as upgrade.
    # Only treat it as a close button when the overlay has hidden the game.
    if not any(item.name == "game" for item in detections):
        crosses = [item for item in detections if item.name == "upgrade"
                   and 10 <= item.box[2] - item.box[0] <= 80
                   and 10 <= item.box[3] - item.box[1] <= 80]
        if crosses:
            cross = max(crosses, key=lambda item: item.confidence)
            return Detection("ads_close", cross.confidence, cross.box)
    if affordable and now - last_upgrade >= upgrade_cooldown:
        return random.choice(affordable)
    return max((item for item in detections if item.name == "game"),
               key=lambda item: item.confidence, default=None)


def price_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    # Join adjacent digits into one price without joining separate shop rows.
    joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    boxes = []
    for x, y, width, height, _ in stats[1:]:
        if (5 <= width <= 100 and 7 <= height <= 35
                and np.count_nonzero(mask[y:y + height, x:x + width]) >= 15):
            boxes.append((int(x), int(y), int(width), int(height)))
    return boxes


def affordable_upgrades(image: np.ndarray, monitor: dict) -> list[Detection]:
    """Use the game's green price state; red prices mean insufficient cookies."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 90, 80), (85, 255, 255))
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 70), (10, 255, 255)),
                         cv2.inRange(hsv, (170, 90, 70), (179, 255, 255)))
    green_boxes = price_boxes(green)
    all_prices = green_boxes + price_boxes(red)
    found = []
    for x, y, _, glyph_height in green_boxes:
        scale = glyph_height / 12
        click_x = round(x + 110 * scale)
        click_y = round(y - 20 * scale)
        if not (5 <= click_x < width - 5 and 5 <= click_y < height - 5):
            continue
        # Shop prices occur in stacked rows on pale stone; tooltip/ad text does not.
        nearby_row = any(
            abs(other_x - x) <= 30 * scale
            and 55 * scale <= abs(other_y - y) <= 90 * scale
            for other_x, other_y, _, _ in all_prices
        )
        if not nearby_row:
            continue
        background = hsv[click_y - 5:click_y + 5, click_x - 5:click_x + 5].mean(axis=(0, 1))
        if not (10 <= background[0] <= 45 and background[1] < 65
                and 100 < background[2] < 220):
            continue
        screen_x = monitor["left"] + click_x
        screen_y = monitor["top"] + click_y
        if all(abs(item.box[0] - screen_x) > 25 or abs(item.box[1] - screen_y) > 25 for item in found):
            found.append(Detection("upgrade", 1.0,
                                   (screen_x, screen_y, screen_x, screen_y)))
    return found


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
                if class_id == 1 and not is_close_button_box(
                    (x, y, x + box_width, y + box_height), width, height
                ):
                    continue  # The model also calls whole side banners ads_close.
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
            overlay_close = find_overlay_close(frame, monitor)
            if overlay_close is not None:
                target = overlay_close
            else:
                detections = detector.detect(frame, monitor)
                if any(item.name == "ads_close" for item in detections):
                    target = choose_target(detections, [], last_upgrade,
                                       time.monotonic(), upgrade_cooldown)
                else:
                    affordable = affordable_upgrades(frame, monitor)
                    target = choose_target(detections, affordable, last_upgrade,
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
