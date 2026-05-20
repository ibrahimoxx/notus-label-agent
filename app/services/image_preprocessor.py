from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessedVariants:
    """All preprocessing variants — OCR tries each in cascade."""

    original_gray: np.ndarray
    thresholded: np.ndarray
    inverted: np.ndarray
    clahe_enhanced: np.ndarray
    deskewed: np.ndarray


class ImagePreprocessor:
    """OpenCV pipeline ordered for pharmaceutical label OCR accuracy."""

    def process(self, image_bgr: np.ndarray) -> PreprocessedVariants:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        gray = self._deskew(gray)

        # Adaptive threshold — NEVER cv2.threshold simple
        thresholded = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # CLAHE — critical for embossed / white-on-white text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(denoised)

        # Light dilation reconnects fragmented characters
        kernel = np.ones((2, 2), np.uint8)
        clahe_enhanced = cv2.dilate(clahe_enhanced, kernel, iterations=1)

        # Inverted variant for white text on dark background
        inverted = cv2.bitwise_not(thresholded)

        if self._is_blurry(gray):
            gray = self._unsharp_mask(gray)

        return PreprocessedVariants(
            original_gray=gray,
            thresholded=thresholded,
            inverted=inverted,
            clahe_enhanced=clahe_enhanced,
            deskewed=gray,
        )

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Straighten image if skew angle > 1° (via moments)."""
        coords = np.column_stack(np.where(gray < 200))
        if len(coords) == 0:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 1.0:
            return gray
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            gray,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _is_blurry(self, gray: np.ndarray, threshold: float = 100.0) -> bool:
        return bool(cv2.Laplacian(gray, cv2.CV_64F).var() < threshold)

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
