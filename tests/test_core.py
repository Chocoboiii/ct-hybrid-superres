"""Sanity tests that need only numpy/scipy/opencv (no torch, no data).

Run:  python tests/test_core.py        (or: pytest)
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ctsr.degrade import degrade  # noqa: E402
from ctsr.masks import body_mask, smart_masks  # noqa: E402
from ctsr.params import Params  # noqa: E402
from ctsr.pipeline import run  # noqa: E402

P = Params()


def synthetic_slice(seed=0):
    """Fake axial slice: dark background, soft-tissue body, two lungs, a bone ring."""
    rng = np.random.default_rng(seed)
    img = np.full((512, 512), 5, np.uint8)
    cv2.ellipse(img, (256, 256), (200, 150), 0, 0, 360, 110, -1)      # soft tissue
    cv2.ellipse(img, (180, 256), (60, 90), 0, 0, 360, 20, -1)         # lung L
    cv2.ellipse(img, (332, 256), (60, 90), 0, 0, 360, 20, -1)         # lung R
    cv2.ellipse(img, (256, 256), (200, 150), 0, 0, 360, 200, 6)       # bone ring
    for _ in range(40):                                               # "vessels" inside lungs
        x, y = rng.integers(140, 370), rng.integers(190, 330)
        cv2.circle(img, (int(x), int(y)), 2, 90, -1)
    return img


def stub_tissue_fn(img_input):
    """Stand-in for SwinIR: down x4 then bicubic up (same I/O contract)."""
    h, w = img_input.shape
    lr = cv2.resize(img_input, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)


def test_degrade_is_reproducible_and_noisy():
    gt = synthetic_slice()
    a = degrade(gt, P, np.random.default_rng(1))
    b = degrade(gt, P, np.random.default_rng(1))
    c = degrade(gt, P, np.random.default_rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.dtype == np.uint8 and a.shape == gt.shape


def test_masks_partition_unity():
    m_air, m_soft, m_bone = smart_masks(synthetic_slice(), P)
    total = m_air + m_soft + m_bone
    assert total.min() >= 0.999 and total.max() <= 1.001
    for m in (m_air, m_soft, m_bone):
        assert 0.0 <= m.min() and m.max() <= 1.0


def test_vessels_not_classified_as_air():
    gt = synthetic_slice()
    m_air, _, _ = smart_masks(gt, P)
    lung_plain = m_air[256, 150]  # flat dark lung
    assert lung_plain > 0.5


def test_body_mask_covers_body_not_background():
    gt = synthetic_slice()
    roi = body_mask(gt, P)
    assert roi[256, 256] == 1 and roi[5, 5] == 0


def test_pipeline_outputs_and_mask_sources_differ():
    gt = synthetic_slice()
    outs = run(gt, stub_tissue_fn, P, np.random.default_rng(0))
    for k in ("input", "air", "bone", "swinir", "hybrid_oracle", "hybrid_deployable"):
        assert outs[k].shape == gt.shape and outs[k].dtype == np.uint8
    # same branches, different masks -> hybrids must not be identical
    assert not np.array_equal(outs["hybrid_oracle"], outs["hybrid_deployable"])


def test_pipeline_is_deterministic_given_seed():
    gt = synthetic_slice()
    a = run(gt, stub_tissue_fn, P, np.random.default_rng(7))
    b = run(gt, stub_tissue_fn, P, np.random.default_rng(7))
    assert all(np.array_equal(a[k], b[k]) for k in a)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} tests passed")
