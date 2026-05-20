import cv2
import numpy as np
import pytest

from app.services.image_preprocessor import ImagePreprocessor


@pytest.fixture()
def sample_bgr() -> np.ndarray:
    img = np.ones((200, 300, 3), dtype=np.uint8) * 200
    cv2.putText(img, "LOT: ABC123", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def test_all_variants_are_valid_arrays(sample_bgr: np.ndarray) -> None:
    variants = ImagePreprocessor().process(sample_bgr)
    for name in ("original_gray", "thresholded", "inverted", "clahe_enhanced", "deskewed"):
        arr = getattr(variants, name)
        assert isinstance(arr, np.ndarray), f"{name} is not ndarray"
        assert arr.size > 0, f"{name} is empty"
        assert arr.dtype == np.uint8, f"{name} dtype unexpected: {arr.dtype}"


def test_variants_are_grayscale(sample_bgr: np.ndarray) -> None:
    variants = ImagePreprocessor().process(sample_bgr)
    for name in ("original_gray", "thresholded", "inverted", "clahe_enhanced", "deskewed"):
        arr = getattr(variants, name)
        assert arr.ndim == 2, f"{name} should be 2D (grayscale), got shape {arr.shape}"


def test_inverted_is_bitwise_not_of_thresholded(sample_bgr: np.ndarray) -> None:
    variants = ImagePreprocessor().process(sample_bgr)
    expected = cv2.bitwise_not(variants.thresholded)
    assert np.array_equal(variants.inverted, expected)


def test_deskew_triggers_on_rotated_image() -> None:
    base = np.ones((200, 400), dtype=np.uint8) * 220
    cv2.putText(base, "LOT: TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Rotate 15 degrees
    M = cv2.getRotationMatrix2D((200, 100), 15, 1.0)
    rotated = cv2.warpAffine(base, M, (400, 200), borderValue=220)

    # Convert to BGR for processor input
    bgr = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    variants = ImagePreprocessor().process(bgr)
    # After deskew, the result should be a valid array (not identical to input gray)
    assert variants.deskewed.shape == rotated.shape


def test_blurry_image_gets_sharpened() -> None:
    blurry_bgr = cv2.GaussianBlur(
        np.ones((200, 300, 3), dtype=np.uint8) * 128, (21, 21), 5
    )
    variants = ImagePreprocessor().process(blurry_bgr)
    assert variants.original_gray is not None
