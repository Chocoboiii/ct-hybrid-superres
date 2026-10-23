"""Build the train / validation / test sets from raw LIDC-IDRI DICOMs.

Differences from the original thesis script:
  * the split is by PATIENT (all slices of a patient go to exactly one split),
    so test/validation slices never come from a patient seen in training;
  * a seed makes the split reproducible;
  * paths are arguments, not hard-coded.

LIDC-IDRI layout expected:  <root>/LIDC-IDRI-0001/<study>/<series>/*.dcm

Outputs (under --out):
    train_HR/ train_LR/   128x128 patches (x4 downsampled LR)
    valid_HR/ valid_LR/   full 512x512 slices
    test_HR/              full 512x512 slices (ground truth for scripts/evaluate.py)
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import pydicom

PATCH, STRIDE, SCALE = 128, 64, 4
RATIOS = {"train": 0.85, "valid": 0.10, "test": 0.05}


def manufacturer(ds) -> str:
    name = str(getattr(ds, "Manufacturer", "")).upper().strip()
    for key in ("TOSHIBA", "PHILIPS", "SIEMENS"):
        if name.startswith(key):
            return key
    return "GE" if name.startswith("GE") else "OTHER"


def to_uint8(path, window):
    """DICOM -> 8-bit 512x512, or None if the slice has too little anatomy."""
    try:
        ds = pydicom.dcmread(path)
        hu = ds.pixel_array * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    except Exception:
        return None
    if np.sum(hu > -700) / hu.size < 0.15:
        return None
    hu = np.clip(hu, -1024, 1024)
    if window == "fixed":
        img = ((hu + 1024.0) / 2048.0 * 255.0).astype(np.uint8)
    else:  # "minmax": per-slice stretch, matches the released checkpoint
        img = cv2.normalize(hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img


def save_patches(img, hr_dir, lr_dir, prefix):
    k = 0
    for y in range(0, 512 - PATCH + 1, STRIDE):
        for x in range(0, 512 - PATCH + 1, STRIDE):
            hr = img[y:y + PATCH, x:x + PATCH]
            lr = cv2.resize(hr, (PATCH // SCALE, PATCH // SCALE), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(hr_dir, f"{prefix}_{k}.png"), hr)
            cv2.imwrite(os.path.join(lr_dir, f"{prefix}_{k}.png"), lr)
            k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidc-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-manufacturer", type=int, default=300,
                    help="slice budget per manufacturer across all splits (balances classes)")
    ap.add_argument("--window", choices=["minmax", "fixed"], default="minmax",
                    help="HU->8-bit mapping. 'minmax' reproduces the thesis; 'fixed' (-1024..1024 HU) "
                         "gives comparable intensities across slices but needs retraining.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    dirs = {n: os.path.join(args.out, n) for n in
            ("train_HR", "train_LR", "valid_HR", "valid_LR", "test_HR")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1) group patients by scanner manufacturer (read one header per patient)
    patients = {}
    for pdir in sorted(glob.glob(os.path.join(args.lidc_root, "LIDC-IDRI-*"))):
        files = glob.glob(os.path.join(pdir, "**", "*.dcm"), recursive=True)
        if not files:
            continue
        try:
            m = manufacturer(pydicom.dcmread(files[0], stop_before_pixels=True))
        except Exception:
            continue
        patients.setdefault(m, []).append((os.path.basename(pdir), files))

    # 2) per manufacturer: split PATIENTS, then sample slices within each split
    for m, plist in patients.items():
        rnd.shuffle(plist)
        n = len(plist)
        n_tr, n_va = int(n * RATIOS["train"]), int(n * RATIOS["valid"])
        split_patients = {"train": plist[:n_tr], "valid": plist[n_tr:n_tr + n_va], "test": plist[n_tr + n_va:]}
        print(f"{m}: {n} patients -> " + ", ".join(f"{k}={len(v)}" for k, v in split_patients.items()))

        for split, plist_s in split_patients.items():
            slices = [f for _, files in plist_s for f in files]
            rnd.shuffle(slices)
            budget = int(args.max_per_manufacturer * RATIOS[split])
            kept = 0
            for f in slices:
                if kept >= budget:
                    break
                img = to_uint8(f, args.window)
                if img is None:
                    continue
                name = f"{m}_{os.path.basename(f)[:-4]}"
                if split == "train":
                    save_patches(img, dirs["train_HR"], dirs["train_LR"], name)
                elif split == "valid":
                    cv2.imwrite(os.path.join(dirs["valid_HR"], name + ".png"), img)
                    lr = cv2.resize(img, (512 // SCALE, 512 // SCALE), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(dirs["valid_LR"], name + ".png"), lr)
                else:
                    cv2.imwrite(os.path.join(dirs["test_HR"], name + ".png"), img)
                kept += 1
            print(f"   {split}: {kept} slices")

    print(f"\nDone. Output in {args.out}")


if __name__ == "__main__":
    main()
