"""SwinIR wrapper. Requires PyTorch and a local clone of KAIR
(https://github.com/cszn/KAIR) for the network definition."""

import sys

import cv2
import numpy as np


def load_model(checkpoint: str, kair_path: str, device):
    import torch

    sys.path.insert(0, kair_path)
    from models.network_swinir import SwinIR  # noqa: E402  (from KAIR)

    # Must match configs/train_swinir_ct.json
    model = SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
        depths=[6] * 9, embed_dim=240, num_heads=[8] * 9, mlp_ratio=2,
        upsampler="nearest+conv", resi_connection="1conv",
    )
    state = torch.load(checkpoint, map_location="cpu")
    key = "params" if "params" in state else ("params_ema" if "params_ema" in state else None)
    model.load_state_dict(state[key] if key else state, strict=True)
    model.eval()
    return model.to(device)


def make_tissue_fn(model, device, scale: int = 4):
    """Return f(img_input_uint8) -> uint8 image at the input resolution.

    The degraded image is bicubically downsampled by `scale`, then SwinIR
    upsamples it back (x`scale`).
    """
    import torch

    def tissue_fn(img_input: np.ndarray) -> np.ndarray:
        h, w = img_input.shape[:2]
        lr = cv2.resize(img_input, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(lr.astype(np.float32) / 255.0)[None, None].to(device)
        t = t.repeat(1, 3, 1, 1)
        with torch.no_grad():
            out = model(t).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # Output is 3-channel; channels are near-identical for grayscale CT.
        # Channel 2 matches the original thesis evaluation code.
        out = out[2] if out.ndim == 3 else out
        sr = (out * 255.0).astype(np.uint8)
        if sr.shape != (h, w):
            sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_CUBIC)
        return sr

    return tissue_fn
