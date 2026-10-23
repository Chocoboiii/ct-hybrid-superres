from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Experiment constants. Defaults reproduce the thesis settings.

    Images are 8-bit grayscale CT slices (HU window mapped to 0-255).
    """

    # Synthetic degradation (Gaussian blur + additive Gaussian noise)
    gauss_sigma: float = 0.6
    snr_db: float = 25.0

    # Intensity thresholds on the 8-bit image
    thresh_air: int = 60
    thresh_bone: int = 160

    # SwinIR upscale factor
    scale: int = 4
