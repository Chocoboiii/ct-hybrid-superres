import cv2
import numpy as np
from scipy import ndimage

from .params import Params


def smart_masks(img: np.ndarray, p: Params):
    """Soft region masks (air, soft tissue, bone) from an 8-bit image.

    Rule-based, not learned:
      * bone  : bright pixels (> thresh_bone), soft ramp
      * air   : dark (< thresh_air) AND locally flat (low gradient), so that
                lung vessels, which are dark but textured, are NOT treated as air
      * soft  : whatever is left
    Masks are feathered with a Gaussian blur. The three masks sum to ~1.
    """
    x = img.astype(np.float32)

    m_bone = np.clip((x - p.thresh_bone) / 80.0, 0, 1)
    m_air_base = np.clip((p.thresh_air - x) / 60.0, 0, 1)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    m_flat = np.clip((15 - mag) / 10.0, 0, 1)

    m_air = m_air_base * m_flat
    m_bone = cv2.GaussianBlur(m_bone, (15, 15), 5.0)
    m_air = cv2.GaussianBlur(m_air, (15, 15), 5.0)
    m_soft = np.clip(1.0 - m_bone - m_air, 0, 1)
    return m_air, m_soft, m_bone


def body_mask(img_gt: np.ndarray, p: Params) -> np.ndarray:
    """Binary body ROI from the ground-truth slice (evaluation only).

    Walls = soft tissue + bone; fill holes (captures lungs); keep the largest
    component (drops table/scanner ring). The same ROI is applied to every
    method, so it does not favour any of them.
    """
    _, walls = cv2.threshold(img_gt, p.thresh_air, 255, cv2.THRESH_BINARY)
    filled = ndimage.binary_fill_holes(walls).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(filled)
    if n < 2:
        return filled // 255

    areas = [(labels == i).sum() for i in range(1, n)]
    biggest = 1 + int(np.argmax(areas))
    mask = np.zeros_like(img_gt)
    mask[labels == biggest] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask // 255
