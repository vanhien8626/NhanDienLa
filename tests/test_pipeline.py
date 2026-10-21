import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import split_dataset
from gradcam import heatmap
from gwo import Hyperparameters, decode, optimize
from models import Classifier

ROOT = Path(__file__).resolve().parents[1]


def make_images(root):
    rng = np.random.default_rng(42)
    for label in ('healthy', 'diseased'):
        directory = root / label
        directory.mkdir(parents=True)
        for i in range(8):
            # Synthetic fixtures only; these are not real leaf disease examples.
            image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(image).save(directory / f'{i}.png')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gwo_reproducible_bounded_and_best_is_monotonic(self):
        def objective(config):
            return abs(config.learning_rate - 0.001) + config.dropout
        best, history = optimize(objective, wolves=3, iterations=3, seed=5)
        self.assertEqual((best, history), optimize(objective, wolves=3, iterations=3, seed=5))
        fitness = [row['best_fitness'] for row in history]
        self.assertEqual(fitness, sorted(fitness, reverse=True))
        self.assertTrue(0.0001 <= best.learning_rate <= 0.01)
        self.assertEqual(decode(np.zeros(6), 0).frozen_blocks, 0)
        with self.assertRaises(ValueError):
            optimize(objective, wolves=2)

    def test_split_is_disjoint_and_removes_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_images(root)
            (root / 'healthy/copy.png').write_bytes((root / 'healthy/0.png').read_bytes())
            manifest = split_dataset(root)
            self.assertEqual(manifest, split_dataset(root))
            self.assertEqual(manifest['duplicates_removed'], 1)
            sets = [{row['path'] for row in manifest['splits'][split]} for split in ('train', 'val', 'test')]
            self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
            self.assertEqual(sum(map(len, sets)), 16)

    def test_frozen_batchnorm_and_gradcam(self):
        for architecture in ('cnn', 'resnet18'):
            model = Classifier(2, Hyperparameters(frozen_blocks=1), architecture, pretrained=False)
            model.train()
            self.assertFalse(model._frozen[0].training)
            cam, prediction = heatmap(model, torch.randn(1, 3, 32, 32))
            self.assertEqual(cam.shape, (32, 32))
            self.assertTrue(np.isfinite(cam).all())
            self.assertIn(prediction, (0, 1))

    def test_end_to_end_train_checkpoint_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, output = root / 'images', root / 'run'
            make_images(images)
            command = [sys.executable, '-B', str(ROOT / 'train.py'), '--data', str(images),
                       '--output', str(output), '--epochs', '1', '--search-epochs', '1',
                       '--wolves', '3', '--iterations', '1', '--repeats', '1', '--image-size', '32']
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['gwo']['repeats'], 1)
            rows = json.loads((output / 'results.json').read_text())
            self.assertEqual(len(rows), 2)
            for method in ('baseline', 'gwo'):
                self.assertTrue((output / f'{method}_42_gradcam.png').is_file())
            predict = subprocess.run([sys.executable, '-B', str(ROOT / 'predict.py'),
                                      '--checkpoint', str(output / 'gwo_42.pt'),
                                      '--image', str(images / 'healthy/0.png'),
                                      '--gradcam', str(root / 'prediction.png')],
                                     cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(predict.returncode, 0, predict.stderr)
            self.assertEqual(len(json.loads(predict.stdout)), 2)
            self.assertTrue((root / 'prediction.png').is_file())


if __name__ == '__main__':
    unittest.main()
