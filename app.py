"""Clicker driven by a four-class YOLO ONNX model. Press Space to stop."""

from __future__ import annotations

import argparse
import ctypes
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np
import onnxruntime as ort
import pyautogui


ROOT = Path(__file__).resolve().parent
CLASSES = ("game", "ads_close", "upgrade", "bonus")
PRIORITY = ("ads_close", "bonus", "upgrade", "game")


@dataclass(frozen=True)
class Config:
    model: Path = ROOT / "weights" / "best.onnx"
    confidence: float = 0.45
    nms_threshold: float = 0.45
    image_size: int = 640
    poll_interval: float = 0.08
    game_cooldown: float = 0.12
    upgrade_cooldown: float = 2.0
    ads_close_cooldown: float = 0.7
    bonus_cooldown: float = 0.5


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    x: int
    y: int


def space_pressed() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    scaled_width, scaled_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (scaled_width, scaled_height))
    left = (size - scaled_width) // 2
    top = (size - scaled_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + scaled_height, left:left + scaled_width] = resized
    return canvas, ratio, left, top


class Detector:
    def __init__(self, config: Config):
        if not config.model.is_file():
            raise FileNotFoundError(f"ONNX model missing: {config.model}")
        self.config = config
        self.session = ort.InferenceSession(str(config.model), providers=["CPUExecutionProvider"])
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        shape = input_info.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Expected NCHW input, got {shape}")
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            if shape[2] != shape[3]:
                raise ValueError(f"Expected square input, got {shape}")
            self.image_size = shape[2]
        else:
            self.image_size = config.image_size
        metadata = self.session.get_modelmeta().custom_metadata_map
        names = metadata.get("names", "")
        if names and any(name not in names for name in CLASSES):
            raise ValueError(
                f"Model classes {names} do not contain {CLASSES}. "
                "Label missing classes, retrain, and export weights/best.onnx."
            )

    def detect(self, image: np.ndarray, monitor: dict) -> list[Detection]:
        padded, ratio, left, top = letterbox(image, self.image_size)
        tensor = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        ) / 255.0
        raw = self.session.run(None, {self.input_name: tensor})[0]
        if raw.ndim != 3 or raw.shape[0] != 1:
            raise ValueError(f"Unsupported ONNX output shape: {raw.shape}")
        rows = raw[0]
        if rows.shape[0] == 4 + len(CLASSES):
            rows = rows.T
        elif rows.shape[1] != 4 + len(CLASSES):
            raise ValueError(
                f"ONNX output {raw.shape} does not have {len(CLASSES)} classes. "
                "Retrain with data.yaml and re-export weights/best.onnx."
            )
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        height, width = image.shape[:2]
        for row in rows:
            class_id = int(np.argmax(row[4:]))
            confidence = float(row[4 + class_id])
            if confidence < self.config.confidence:
                continue
            cx, cy, w, h = map(float, row[:4])
            x1 = max(0, min(width - 1, round((cx - w / 2 - left) / ratio)))
            y1 = max(0, min(height - 1, round((cy - h / 2 - top) / ratio)))
            x2 = max(0, min(width - 1, round((cx + w / 2 - left) / ratio)))
            y2 = max(0, min(height - 1, round((cy + h / 2 - top) / ratio)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)
        detections = []
        for class_id in set(class_ids):
            indices = [i for i, value in enumerate(class_ids) if value == class_id]
            kept = cv2.dnn.NMSBoxes(
                [boxes[i] for i in indices], [scores[i] for i in indices],
                self.config.confidence, self.config.nms_threshold,
            )
            for local_index in np.array(kept).reshape(-1):
                index = indices[int(local_index)]
                x, y, w, h = boxes[index]
                detections.append(Detection(
                    CLASSES[class_id], scores[index],
                    int(monitor["left"] + x + w / 2),
                    int(monitor["top"] + y + h / 2),
                ))
        return detections


def choose_target(detections: list[Detection], last_click: dict[str, float],
                  now: float, config: Config) -> Detection | None:
    cooldowns = {
        "ads_close": config.ads_close_cooldown,
        "bonus": config.bonus_cooldown,
        "upgrade": config.upgrade_cooldown,
        "game": config.game_cooldown,
    }
    for name in PRIORITY:
        matches = [item for item in detections if item.name == name]
        if matches:
            if now - last_click.get(name, -float("inf")) < cooldowns[name]:
                return None
            return max(matches, key=lambda item: item.confidence)
    return None


def run(config: Config) -> None:
    detector = Detector(config)
    last_click: dict[str, float] = {}
    pyautogui.FAILSAFE = True
    print("Running. Press Space to stop; move mouse to a corner for emergency stop.")
    with mss.mss() as capture:
        monitor = capture.monitors[0]
        while not space_pressed():
            frame = np.asarray(capture.grab(monitor))[:, :, :3]
            target = choose_target(
                detector.detect(frame, monitor), last_click, time.monotonic(), config
            )
            if target is not None and not space_pressed():
                pyautogui.click(target.x, target.y)
                last_click[target.name] = time.monotonic()
            time.sleep(config.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Config.model)
    parser.add_argument("--confidence", type=float, default=Config.confidence)
    parser.add_argument("--upgrade-cooldown", type=float, default=Config.upgrade_cooldown)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or args.upgrade_cooldown < 0:
        parser.error("confidence must be in (0, 1]; cooldown must be nonnegative")
    config = Config(
        model=args.model, confidence=args.confidence,
        upgrade_cooldown=args.upgrade_cooldown,
    )
    try:
        run(config)
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        print("Stopped.")


if __name__ == "__main__":
    main()
