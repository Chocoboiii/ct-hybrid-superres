# Hybrid region-adaptive super-resolution for CT

Master's thesis project (Maestría en Ciencias en Bioingeniería y Cómputo Inteligente, CUCEI, Universidad de Guadalajara).

A CT slice is not one kind of image: air, soft tissue and bone have different noise and texture. Instead of
running one network over everything, this pipeline splits each slice into regions and treats each with the
method that suits it, then blends the results.

| Region | Method | Notes |
|---|---|---|
| Air / background | Bilateral filter | edge-preserving denoising |
| Soft tissue | SwinIR (x4), fine-tuned on LIDC-IDRI patches from the official pretrained weights | the only learned component |
| Bone | Non-Local Means | denoising at full resolution |

Regions come from a rule-based soft segmentation: intensity thresholds on the 8-bit slice plus a gradient
"flatness" term, so dark but textured structures such as lung vessels are not mistaken for air. Masks are
feathered and alpha-blended.

## Results

Synthetic degradation (Gaussian blur σ = 0.6 + noise at 25 dB SNR) of 117 validation slices, scored inside the
body ROI. Mean per metric from `scripts/evaluate.py` (seeded; the oracle row reproduces the thesis numbers):

| Method | PSNR (dB) ↑ | SSIM ↑ | FSIM ↑ |
|---|---|---|---|
| Bilateral only | 33.65 | 0.8382 | **0.9627** |
| NLM only | 34.25 | 0.8493 | 0.9455 |
| SwinIR only | 33.93 | **0.8693** | 0.9595 |
| Hybrid, oracle masks (thesis setting) | **34.33** | **0.8706** | 0.9604 |
| Hybrid, deployable masks | 34.31 | 0.8691 | 0.9606 |

Paired Wilcoxon tests, deployable hybrid vs each baseline (full output in `results/stats_deployable.csv`,
Holm-adjusted):

| | PSNR | SSIM | FSIM |
|---|---|---|---|
| vs SwinIR | **+0.38 dB** (p < 1e-19) | no difference (p = 0.96) | +0.0011 (p < 1e-18) |
| vs NLM only | no difference (p = 1.0) | +0.020 (p < 1e-13) | +0.015 (p < 1e-16) |
| vs Bilateral only | +0.65 dB (p < 1e-8) | +0.031 (p < 1e-13) | no difference (p = 1.0) |

What this says, plainly: the hybrid reliably improves on SwinIR alone in PSNR and FSIM, but the gains are small,
it matches SwinIR on SSIM, and a plain Non-Local Means denoiser is statistically tied with it on PSNR. Each
single method wins somewhere; the hybrid is the best compromise across the three metrics rather than a winner
on all of them.

### Oracle vs. deployable masks

The thesis blend masks were computed from the *clean ground-truth* slice, which is not available when
restoring a real degraded scan. `scripts/evaluate.py` therefore reports two hybrids built from identical branch
outputs:

* **Hybrid_Oracle**: masks from the ground truth (upper bound).
* **Hybrid_Deployable**: masks from a denoised copy of the degraded input only.

Report the deployable number as the realistic one. Result: using ground-truth masks costs almost nothing in PSNR and
FSIM (34.33 → 34.31 dB, 0.9604 → 0.9606), but the small SSIM edge over SwinIR (0.8706 vs 0.8693) disappears with
deployable masks (0.8691).

## Reproduce

```bash
conda env create -f environment.yml && conda activate ctsr
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the build for your system (CPU: drop --index-url)
pip install piq timm
git clone https://github.com/cszn/KAIR                     # network definition
python tests/test_core.py                                   # sanity tests, no GPU or data needed

# 1) data (LIDC-IDRI from TCIA, not redistributed here)
python scripts/prepare_data.py --lidc-root PATH/TO/LIDC-IDRI --out data/

# 2) train (see training/README.md) or download the weights from
#    https://github.com/Chocoboiii/ct-hybrid-superres/releases/latest
#    and put swinir_ct_50000_G.pth in weights/

# 3) evaluate, then test statistically
python scripts/evaluate.py --data-dir data/test_HR --checkpoint weights/swinir_ct_50000_G.pth \
       --kair-path KAIR --out results/metrics.csv
python scripts/stats.py results/metrics.csv --hybrid Hybrid_Deployable
```

## Limitations

* **Synthetic degradation only.** Low-quality inputs are simulated from clean slices; no real low-dose scans.
* **Original split was by slice, not patient.** The thesis experiments sampled slices at random, so slices of
  one patient could appear in both training and evaluation. `scripts/prepare_data.py` here splits by patient.
  Numbers from the thesis run are not guaranteed to hold under a patient-level split.
* **Per-slice min-max intensity scaling** was used to map HU to 8-bit, so intensity thresholds do not mean the
  same HU on every slice. A fixed window (`--window fixed`) is cleaner but requires retraining.
* **8-bit processing** and ROI-masked metrics (background pixels are zero in both images, which slightly raises
  PSNR/SSIM for all methods equally).
* Segmentation thresholds were tuned by hand on this dataset.

## License

MIT (see `LICENSE`). `training/main_train_psnr.py` derives from KAIR (MIT, notice in `training/LICENSE-KAIR`).
Model weights are fine-tuned from the official SwinIR weights; check the upstream SwinIR license for their terms.

## Data and credits

* Dataset: LIDC-IDRI, The Cancer Imaging Archive (Armato et al., Medical Physics 2011). Follow TCIA's terms; no
  images are included in this repository.
* SwinIR: Liang et al., *SwinIR: Image Restoration Using Swin Transformer*, ICCVW 2021. Training code and network
  from [KAIR](https://github.com/cszn/KAIR).
