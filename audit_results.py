"""Independently reconstruct metrics from saved predictions and confusion counts."""
import sys
sys.dont_write_bytecode = True
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

root = Path(__file__).resolve().parent
p = json.loads((root / 'results/metrics.json').read_text())


def read(name):
    with (root / 'results' / name).open(encoding='utf-8') as file:
        return list(csv.DictReader(file))


combined = read('results.csv')
assert len(combined) == 7
history = read('training_history.csv')
assert len(history) == p['config']['epochs']
assert max(float(r['test_accuracy']) for r in history) == p['best_test_accuracy']
cm = np.array([[int(r[f'predicted_{j}']) for j in range(10)] for r in read('confusion_matrix.csv')])
assert cm.sum() == 10000
assert abs(cm.trace() / 10000 - p['deterministic']['accuracy']) < 1e-12
y = np.repeat(np.arange(10), cm.sum(1))
yp = np.concatenate([np.repeat(np.arange(10), cm[i]) for i in range(10)])
vals = precision_recall_fscore_support(y, yp, labels=np.arange(10), average='macro', zero_division=0)[:3]
assert np.allclose(vals, [p['deterministic'][k] for k in ['precision', 'recall', 'f1']])
assert abs(float(combined[0]['accuracy']) - p['deterministic']['accuracy']) < 1e-12
for position, noise in enumerate(p['noise_results'], 1):
    rows = read(f"uncertainty_noise_{noise['noise_level']:.1f}.csv")
    assert len(rows) == 10000
    assert [int(r['index']) for r in rows] == list(range(10000))
    y = [int(r['true_label']) for r in rows]
    yp = [int(r['predicted_label']) for r in rows]
    assert abs(accuracy_score(y, yp) - noise['accuracy']) < 1e-12
    assert abs(float(combined[position]['accuracy']) - noise['accuracy']) < 1e-12
    for metric, column in [('average_predictive_entropy', 'predictive_entropy'), ('average_confidence', 'confidence'), ('average_predictive_variance', 'predictive_variance')]:
        assert abs(np.mean([float(r[column]) for r in rows]) - noise[metric]) < 1e-12
        assert abs(float(combined[position][metric]) - noise[metric]) < 1e-12
    probabilities = np.array([[float(r[f'probability_{k}']) for k in range(10)] for r in rows])
    variance = np.array([[float(r[f'variance_{k}']) for k in range(10)] for r in rows])
    assert np.allclose(probabilities.sum(1), 1, atol=1e-5)
    assert np.all(variance >= 0)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(1)
    assert np.allclose(entropy, [float(r['predictive_entropy']) for r in rows], atol=1e-6)
assert (root / 'results/uncertainty_noise_0.0.csv').read_bytes() == (root / 'results/uncertainty_samples.csv').read_bytes()
for name in ['training_curves.png', 'confusion_matrix.png', 'uncertainty_examples.png', 'noisy_uncertainty.png']:
    assert (root / 'results' / name).stat().st_size > 1000
print('PASS: six complete test scans, CSV/JSON consistency, probability sums, reconstructed entropy, variance, confusion counts, independent macro metrics and complete training history.')
