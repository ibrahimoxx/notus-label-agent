from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image(path: str | Path) -> np.ndarray:
    """Read image as BGR numpy array. Falls back to PIL for exotic formats."""
    img = cv2.imread(str(path))
    if img is None:
        pil = Image.open(str(path)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
