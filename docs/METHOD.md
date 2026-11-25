# Method and limits

## A small model with an auditable comparison

A 64-hidden-unit MLP is large enough to model nonlinear combinations of the supplied pixels while remaining small enough to export as four JSON tensors. It is not a CNN. Its parameter count is 64×64 + 64 + 64×10 + 10 = 4,810.

Logistic regression is the linear reference, not a deliberately random or undertrained baseline. Both models see the same training rows and fixed scaling. C=1.0, max_iter=2000 and the MLP hyperparameters were set before examining test metrics. No parameter sweep was run.

The original UCI training file has 3,823 images. A deterministic stratified split allocates 3,058 to optimization and 765 to validation. The model trains for 40 epochs; the state with the lowest validation cross-entropy is retained. Only after this selection are predictions made on the original 1,797-image test file. The resulting checkpoint is epoch 11. Training continues beyond that epoch for the recorded learning curve, but later checkpoints do not replace it unless their validation loss improves.

Training loss decreases while validation loss eventually rises. That pattern illustrates why selecting the last epoch would be a poor default here. There is no learned scaler fitted across all rows: division by 16 follows the dataset’s fixed value range.

## Interpret the evidence

The test accuracy is 95.5481% (1,717/1,797), and macro F1 is 0.9554. Macro F1 weights each digit class equally. The baseline has 1,700 correct predictions. A 17-image advantage in one run does not establish general superiority.

The confusion matrix preserves every true/predicted pair. The static error gallery deliberately chooses the 12 highest softmax-score mistakes to demonstrate overconfidence; it is not a random sample. The interactive demo can browse all 80 errors and all 1,717 correct cases. IDs are zero-based rows in the original test file.

The exported browser network is checked against every saved PyTorch prediction. JavaScript uses double-precision arithmetic while PyTorch uses float32; small score differences are expected. The automated parity tolerance is 0.00001, and class labels must match exactly.

## Limits and useful next steps

- This is a small prepared-image dataset, not photographs or unrestricted handwriting. The demo deliberately does not accept uploads.
- The training/validation split is image-level; per-row writer IDs are unavailable. The official train/test writer separation is documented by UCI, not independently inferred here.
- One seed and dataset are reported. No significance test, multi-seed uncertainty estimate or external dataset evaluation was performed.
- Scores are uncalibrated. An apparently confident prediction may be wrong.
- The test set is now public and inspected. Future model decisions informed by these errors need a new untouched evaluation set.
- A worthwhile next experiment is a predeclared multi-seed comparison with a small CNN and an independent dataset, not a claim that more layers must improve accuracy.

## Files to explain in an interview

`dataset.py` fixes the data boundaries. `train.py` handles optimization and validation-based checkpoint selection. `model.py` defines the network and safe JSON tensor serialization. `verify_artifacts.py` recomputes recorded evidence. `web/inference.mjs` performs the same forward pass in a browser. Tests cover failure cases as well as the valid path.
