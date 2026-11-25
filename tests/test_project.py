import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from dataset import load_data, split_training
from model import DigitClassifier, export_model, load_model, normalize, probabilities
from train import run
from verify_artifacts import verify

ROOT = Path(__file__).resolve().parents[1]

class DataTests(unittest.TestCase):
    def test_original_split_is_preserved(self):
        (x, y), (tx, ty) = load_data()
        self.assertEqual((x.shape, tx.shape), ((3823, 64), (1797, 64)))
        self.assertEqual(set(y), set(range(10)))
        self.assertEqual(set(ty), set(range(10)))
    def test_validation_is_disjoint_complete_and_reproducible(self):
        (_, y), _ = load_data()
        a, b = split_training(y)
        self.assertEqual((len(a), len(b)), (3058, 765))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(3823)))
        np.testing.assert_array_equal(a, split_training(y)[0])
    def test_damaged_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'optdigits.tra').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_data(folder)

class ModelTests(unittest.TestCase):
    def test_normalization_bounds(self):
        x = normalize([[0] * 32 + [16] * 32])
        self.assertEqual(tuple(x.shape), (1, 64))
        self.assertEqual((float(x.min()), float(x.max())), (0, 1))
    def test_invalid_inputs_are_rejected(self):
        for pixels in [[], [1] * 63, [[0] * 64, [0]], [17] * 64, [-1] * 64, [float('nan')] * 64, [float('inf')] * 64]:
            with self.subTest(pixels=str(pixels)[:30]), self.assertRaises(ValueError):
                normalize(pixels)
    def test_scores_sum_to_one_and_batch_matches_single(self):
        model = DigitClassifier()
        x = [[0] * 64, [16] * 64]
        scores = probabilities(model, x)
        np.testing.assert_allclose(scores.sum(1), [1, 1], atol=1e-6)
        np.testing.assert_allclose(scores[0], probabilities(model, x[0])[0], atol=1e-6)
    def test_export_roundtrip_preserves_predictions(self):
        model = DigitClassifier()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            export_model(model, path)
            np.testing.assert_array_equal(probabilities(model, [8] * 64), probabilities(load_model(path), [8] * 64))
    def test_corrupt_model_is_rejected(self):
        base = json.loads((ROOT / 'artifacts/model.json').read_text())
        changes = [lambda m: m.update(pixel_max=255), lambda m: m['state'].pop('layers.0.bias'),
                   lambda m: m['state'].update({'layers.0.bias': [0]}),
                   lambda m: m['state']['layers.0.bias'].__setitem__(0, float('inf'))]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            for change in changes:
                data = copy.deepcopy(base)
                change(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_model(path)
    def test_training_does_not_overwrite_results(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'preserve'):
            run(folder)

class EvidenceTests(unittest.TestCase):
    def test_committed_results_recompute_from_export(self):
        result = verify(ROOT / 'artifacts')
        self.assertEqual(result['verified_test_images'], 1797)
    def test_modified_prediction_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            for path in (ROOT / 'artifacts').glob('*.json'):
                shutil.copy(path, folder)
            path = Path(folder) / 'predictions.json'
            rows = json.loads(path.read_text())
            rows[0]['actual'] = (rows[0]['actual'] + 1) % 10
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'sample mismatch'):
                verify(folder)

if __name__ == '__main__':
    unittest.main()
