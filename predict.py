"""Classify one original 8×8 UCI test image using the saved JSON weights."""
import argparse
import json
from pathlib import Path
from dataset import load_data
from model import load_model, probabilities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=int, default=0, help='Zero-based test index, 0–1796')
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'artifacts/model.json')
    args = parser.parse_args()
    _, (pixels, labels) = load_data()
    if not 0 <= args.sample < len(labels):
        parser.error('--sample must be between 0 and 1796')
    scores = probabilities(load_model(args.model), pixels[args.sample])[0]
    top = scores.argsort()[::-1][:3]
    print(json.dumps({'sample': args.sample, 'actual': int(labels[args.sample]),
                      'top_predictions': [{'digit': int(i), 'model_score': float(scores[i])} for i in top]}, indent=2))
