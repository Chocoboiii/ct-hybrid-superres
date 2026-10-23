"""Evaluate SwinIR, the single-method branches and BOTH hybrids (oracle and
deployable masks) on a folder of 8-bit CT slices.

Example:
    python scripts/evaluate.py --data-dir D:/data/test --checkpoint weights/swinir_ct_50000_G.pth \
        --kair-path D:/code/KAIR --out results/metrics.csv
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ctsr.masks import body_mask  # noqa: E402
from ctsr.metrics import all_metrics  # noqa: E402
from ctsr.params import Params  # noqa: E402
from ctsr.pipeline import run  # noqa: E402

METHODS = {  # column suffix -> key in pipeline.run() output
    "Air": "air",
    "Bone": "bone",
    "SwinIR": "swinir",
    "Hybrid_Oracle": "hybrid_oracle",
    "Hybrid_Deployable": "hybrid_deployable",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder of 8-bit grayscale PNG slices (ground truth)")
    ap.add_argument("--checkpoint", required=True, help="SwinIR generator weights (.pth)")
    ap.add_argument("--kair-path", required=True, help="path to a local clone of cszn/KAIR")
    ap.add_argument("--out", default="results/metrics.csv")
    ap.add_argument("--seed", type=int, default=0, help="base seed; image i uses seed+i")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    ap.add_argument("--no-fsim", action="store_true", help="skip FSIM (no piq needed)")
    args = ap.parse_args()

    import torch
    from ctsr.swinir import load_model, make_tissue_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.kair_path, device)
    tissue_fn = make_tissue_fn(model, device)
    p = Params()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"No PNG files found in {args.data_dir}")

    rows = []
    for i, path in enumerate(files):
        gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        rng = np.random.default_rng(args.seed + i)
        outs = run(gt, tissue_fn, p, rng)

        roi = body_mask(gt, p)
        gt_roi = cv2.bitwise_and(gt, gt, mask=roi)

        row = {"Filename": os.path.basename(path)}
        for suffix, key in METHODS.items():
            pred = cv2.bitwise_and(outs[key], outs[key], mask=roi)
            for metric, value in all_metrics(gt_roi, pred, device, not args.no_fsim).items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"[{i + 1}/{len(files)}] {os.path.basename(path)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nMean over {len(rows)} slices (ROI-masked):")
    metrics = [m for m in ("PSNR", "SSIM", "FSIM") if f"{m}_Air" in rows[0]]
    print(f"{'method':<20}" + "".join(f"{m:>10}" for m in metrics))
    for suffix in METHODS:
        vals = [np.mean([r[f"{m}_{suffix}"] for r in rows]) for m in metrics]
        print(f"{suffix:<20}" + "".join(f"{v:>10.4f}" for v in vals))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
