"""Recompute the published results from the exported weights and original test images."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from dataset import SHA256, load_data, split_training
from model import load_model, probabilities
from train import metrics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(folder):
    folder = Path(folder)
    read = lambda name: json.loads((folder / name).read_text())
    report, split, history, rows = (read(name) for name in ['metrics.json', 'split.json', 'history.json', 'predictions.json'])
    (_, train_labels), (pixels, labels) = load_data()
    train_ids, validation_ids = split_training(train_labels, report['seed'])
    require(split == {'train_indices': train_ids.tolist(), 'validation_indices': validation_ids.tolist()}, 'Training split mismatch')
    require(report['split'] == {'training': len(train_ids), 'validation': len(validation_ids), 'test': len(labels)}, 'Split counts mismatch')
    require(report['dataset_sha256'] == SHA256, 'Dataset checksums mismatch')
    require(len(history) == report['epochs'] and [r['epoch'] for r in history] == list(range(1, report['epochs'] + 1)), 'History mismatch')
    require(all(np.isfinite(r[k]) and r[k] >= 0 for r in history for k in ['train_loss', 'validation_loss']), 'Invalid history loss')
    require(min(history, key=lambda r: r['validation_loss'])['epoch'] == report['selected_epoch'], 'Checkpoint selection mismatch')
    require(hashlib.sha256((folder / 'model.json').read_bytes()).hexdigest() == report['model_sha256'], 'Model checksum mismatch')
    model = load_model(folder / 'model.json')
    require(sum(p.numel() for p in model.parameters()) == report['parameters'], 'Parameter count mismatch')
    scores = probabilities(model, pixels)
    predicted = scores.argmax(1)
    baseline = read('baseline.json')
    weights, bias = np.asarray(baseline['weights']), np.asarray(baseline['bias'])
    require(weights.shape == (10, 64) and bias.shape == (10,) and np.isfinite(weights).all() and np.isfinite(bias).all(), 'Invalid baseline export')
    baseline_predicted = (pixels / 16.0 @ weights.T + bias).argmax(1)
    for name, pred in [('mlp', predicted), ('logistic_regression', baseline_predicted)]:
        require(metrics(labels, pred) == report[name], f'{name} metrics mismatch')
    require(len(rows) == len(labels), 'Missing test predictions')
    for i, row in enumerate(rows):
        require(row['index'] == i and row['pixels'] == pixels[i].tolist() and row['actual'] == int(labels[i]), f'Test sample mismatch: {i}')
        require(row['predicted'] == int(predicted[i]) and row['baseline_predicted'] == int(baseline_predicted[i]), f'Prediction mismatch: {i}')
        require(abs(row['score'] - float(scores[i].max())) < 1e-6, f'Score mismatch: {i}')
    return {'verified_test_images': len(labels), 'errors': int(np.sum(predicted != labels)), 'accuracy': report['mlp']['accuracy']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
    print(json.dumps(verify(parser.parse_args().artifacts), indent=2))
