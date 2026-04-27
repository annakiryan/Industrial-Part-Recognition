from typing import Optional

import cv2
import numpy as np


def remove_black_borders(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return image[y_min : y_max + 1, x_min : x_max + 1]


class CameraSource:
    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.camera_index}")

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera is not opened")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")

        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
