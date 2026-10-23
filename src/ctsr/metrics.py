import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, test))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    return float(structural_similarity(ref, test))


def fsim(ref: np.ndarray, test: np.ndarray, device="cpu") -> float:
    """Feature Similarity Index (grayscale) via the `piq` package."""
    import piq
    import torch

    def to_t(a):
        return (torch.from_numpy(a).float() / 255.0)[None, None].to(device)

    with torch.no_grad():
        return float(piq.fsim(to_t(test), to_t(ref), data_range=1.0, chromatic=False).item())


def all_metrics(ref, test, device="cpu", use_fsim=True) -> dict:
    out = {"PSNR": psnr(ref, test), "SSIM": ssim(ref, test)}
    if use_fsim:
        out["FSIM"] = fsim(ref, test, device)
    return out
