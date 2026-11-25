"""Train on CPU, select using validation loss, then evaluate the untouched UCI test set."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch import nn
from dataset import SHA256, load_data, split_training
from model import DigitClassifier, export_model, normalize, probabilities


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def metrics(labels, predictions):
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, average='macro')),
            'correct': int(np.sum(labels == predictions)), 'total': len(labels),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=range(10)).tolist()}


def run(output, epochs=40, seed=2026):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output folder to preserve earlier results.')
    if not 1 <= epochs <= 200:
        raise ValueError('Epochs must be between 1 and 200.')
    output.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    (pixels, labels), (test_pixels, test_labels) = load_data()
    train_ids, validation_ids = split_training(labels, seed)
    x = normalize(pixels)
    y = torch.tensor(labels, dtype=torch.long)
    model = DigitClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state = float('inf'), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.tensor(train_ids)[torch.randperm(len(train_ids))]
        total_loss = 0.0
        for batch in order.split(64):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            logits = model(x[validation_ids])
            validation_loss = float(loss_fn(logits, y[validation_ids]))
            validation_accuracy = float((logits.argmax(1) == y[validation_ids]).float().mean())
        history.append({'epoch': epoch, 'train_loss': total_loss / len(train_ids),
                        'validation_loss': validation_loss, 'validation_accuracy': validation_accuracy})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    # Baseline receives the exact same training rows and fixed normalization.
    baseline = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    baseline.fit(pixels[train_ids] / 16.0, labels[train_ids])
    # No test metric is inspected during fitting or epoch selection.
    scores = probabilities(model, test_pixels)
    predictions = scores.argmax(1)
    baseline_predictions = baseline.predict(test_pixels / 16.0)
    export_model(model, output / 'model.json')
    write_json(output / 'baseline.json', {'weights': baseline.coef_.tolist(), 'bias': baseline.intercept_.tolist()})
    result = {'seed': seed, 'epochs': epochs, 'selected_epoch': best_epoch,
              'selection': 'minimum validation cross-entropy; test used only after selection',
              'split': {'training': len(train_ids), 'validation': len(validation_ids), 'test': len(test_labels)},
              'dataset': 'UCI Optical Recognition of Handwritten Digits, official train/test files',
              'dataset_sha256': SHA256, 'model_sha256': hashlib.sha256((output/'model.json').read_bytes()).hexdigest(),
              'parameters': sum(p.numel() for p in model.parameters()),
              'mlp': metrics(test_labels, predictions), 'logistic_regression': metrics(test_labels, baseline_predictions),
              'training_seconds': training_seconds,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'device': 'cpu'},
              'executed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(output / 'metrics.json', result)
    write_json(output / 'history.json', history)
    write_json(output / 'split.json', {'train_indices': train_ids.tolist(), 'validation_indices': validation_ids.tolist()})
    examples = [{'index': i, 'pixels': row.tolist(), 'actual': int(test_labels[i]),
                 'predicted': int(predictions[i]), 'baseline_predicted': int(baseline_predictions[i]),
                 'score': float(scores[i].max())} for i, row in enumerate(test_pixels)]
    write_json(output / 'predictions.json', examples)
    print(json.dumps({k: result[k] for k in ['split','selected_epoch','parameters','training_seconds']}, indent=2))
    for name in ['mlp','logistic_regression']:
        print(name, {k: result[name][k] for k in ['accuracy','macro_f1','correct','total']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    run(args.output, args.epochs, args.seed)
