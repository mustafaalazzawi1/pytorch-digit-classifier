# Dataset provenance

**Optical Recognition of Handwritten Digits** — E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C50P49

- Dataset page: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Original archive: https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip
- License: **Creative Commons Attribution 4.0 International**, https://creativecommons.org/licenses/by/4.0/
- Retrieved: 25 November 2026.

`optdigits.tra` (3,823 rows) and `optdigits.tes` (1,797 rows) are unmodified copies from the archive. `checksums.json` records their SHA-256 hashes; `dataset.py` checks those hashes before loading. Every row contains 64 prepared image pixels followed by its digit label. Pixel values range from 0 to 16.

UCI describes 30 writers in the training set and 13 different writers in the test set. This repository preserves those file boundaries. It does not independently identify individual writers from image rows.

Project transformations: 80/20 stratified image-level training/validation split of the training file, fixed seed 2026; pixel scaling by 1/16; test images reshaped to 8×8 for display. No augmentation or test-set fitting is used. Derived galleries, plots and `predictions.json` retain this attribution. No endorsement by the dataset creators is implied.
