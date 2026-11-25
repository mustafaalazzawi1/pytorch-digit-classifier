## Security maintenance — 25 November 2026

The supported requirements now pin PyTorch 2.15.0. This includes the fix for
[GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p),
which affected the former 2.9.1 dependency. The project uses validated JSON model exports,
not `torch.load` or pickle checkpoints; no exploitable checkpoint-loading path was found here.

Validation on Linux with Python 3.12.14 and PyTorch 2.15.0+cpu passed all 11 Python tests,
artifact verification, and browser parity over all 1,797 test images. A fresh 40-epoch
fixed-seed training run selected epoch 11 and reproduced the published confusion matrix:
1,717 correct predictions, 80 errors, and 95.55% accuracy. The original recorded artifacts
and their original environment metadata remain unchanged. New training runs record their
own environment. CI reads the CPU Torch requirement directly from `requirements.txt`.
