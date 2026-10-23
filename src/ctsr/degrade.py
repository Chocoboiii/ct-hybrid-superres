import cv2
import numpy as np

from .params import Params


def degrade(img: np.ndarray, p: Params, rng: np.random.Generator) -> np.ndarray:
    """Blur + additive Gaussian noise at a target SNR (uint8 in, uint8 out).

    `rng` is explicit so that every evaluation run is reproducible.
    """
    x = img.astype(np.float32)
    ksize = max(int(6 * p.gauss_sigma) // 2 * 2 + 1, 1)
    blurred = cv2.GaussianBlur(x, (ksize, ksize), p.gauss_sigma)

    signal_power = float(np.var(x))
    if signal_power == 0:
        return img
    noise_power = signal_power / (10 ** (p.snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), img.shape)
    return np.clip(blurred + noise, 0, 255).astype(np.uint8)
