import cv2
import numpy as np


def dither(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Add light Gaussian texture so smoothed regions do not look plastic."""
    noise = rng.normal(0, strength, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def air_branch(img_input: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bilateral filter for air/background (edge-preserving denoising)."""
    out = cv2.bilateralFilter(img_input, d=9, sigmaColor=25, sigmaSpace=5)
    return dither(out, 3.0, rng)


def bone_branch(img_input: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Non-Local Means for bone (denoising at full resolution)."""
    out = cv2.fastNlMeansDenoising(
        img_input, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    return dither(out, 2.0, rng)


def mask_guide(img_input: np.ndarray) -> np.ndarray:
    """Denoised copy of the *degraded input*, used to build masks without
    access to the ground truth."""
    return cv2.fastNlMeansDenoising(
        img_input, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
