# Training

`main_train_psnr.py` is a lightly modified copy of the training entry point from
[KAIR](https://github.com/cszn/KAIR) (additions: per-iteration CSV logging of loss / train PSNR and a
detector that dumps unusually bad batches). KAIR is MIT licensed; its copyright notice is kept in
`LICENSE-KAIR`.

```bash
git clone https://github.com/cszn/KAIR
cp training/main_train_psnr.py KAIR/
# edit the dataroot_* paths in configs/train_swinir_ct.json, then from inside KAIR/:
python main_train_psnr.py --opt ../configs/train_swinir_ct.json
```

Fine-tuning starts from the official pretrained SwinIR weights: set `path.pretrained_netG` in the config to
the downloaded `.pth` (it is `null` in the shared config). The network settings must match that checkpoint.
