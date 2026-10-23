from typing import Callable, Dict

import numpy as np

from .degrade import degrade
from .filters import air_branch, bone_branch, mask_guide
from .masks import smart_masks
from .params import Params


def blend(air, tissue, bone, masks) -> np.ndarray:
    m_air, m_soft, m_bone = masks
    out = air * m_air + tissue * m_soft + bone * m_bone
    return np.clip(out, 0, 255).astype(np.uint8)


def run(
    gt: np.ndarray,
    tissue_fn: Callable[[np.ndarray], np.ndarray],
    p: Params,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Degrade a ground-truth slice and reconstruct it with every method.

    All methods see the SAME degraded input and the SAME branch outputs, so
    the two hybrids differ only in where their blend masks come from:

      hybrid_oracle      masks from the clean ground truth (upper bound; not
                         available at inference time)
      hybrid_deployable  masks from a denoised copy of the degraded input
                         (what you could actually run on a real low-quality scan)
    """
    inp = degrade(gt, p, rng)
    air = air_branch(inp, rng)
    bone = bone_branch(inp, rng)
    tissue = tissue_fn(inp)

    return {
        "input": inp,
        "air": air,
        "bone": bone,
        "swinir": tissue,
        "hybrid_oracle": blend(air, tissue, bone, smart_masks(gt, p)),
        "hybrid_deployable": blend(air, tissue, bone, smart_masks(mask_guide(inp), p)),
    }
