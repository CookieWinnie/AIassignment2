from contextlib import contextmanager
import numpy as np
import torch
from src.evaluate import classification
from src.utils import seed_all


@contextmanager
def active_dropout(model):
    states = {layer: layer.training for layer in model.modules()}
    model.eval()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.dropout._DropoutNd):
            layer.train()
    try:
        yield
    finally:
        for layer, state in states.items():
            layer.training = state


@torch.inference_mode()
def mc_predict(model, x, samples=30):
    if samples < 2:
        raise ValueError('MC samples must be at least 2')
    with active_dropout(model):
        probabilities = torch.stack([model(x).softmax(1) for _ in range(samples)])
    mean = probabilities.mean(0)
    variance = probabilities.var(0, unbiased=False)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(1)
    confidence, pred = mean.max(1)
    return mean, variance, entropy, confidence, pred


@torch.inference_mode()
def scan(model, loader, device, samples, noise, seed):
    seed_all(seed)
    # An independent CPU generator makes noise independent of dropout sampling.
    noise_rng = torch.Generator().manual_seed(seed + 1000)
    rows, labels, predictions = [], [], []
    index = 0
    for x, y in loader:
        if noise:
            x = (x + noise * torch.randn(x.shape, generator=noise_rng)).clamp(0, 1)
        mean, variance, entropy, confidence, pred = mc_predict(model, x.to(device), samples)
        mean, variance = mean.cpu().numpy(), variance.cpu().numpy()
        entropy, confidence, pred = entropy.cpu().numpy(), confidence.cpu().numpy(), pred.cpu().numpy()
        assert np.allclose(mean.sum(1), 1, atol=1e-5)
        assert np.all(variance >= 0) and np.all(entropy >= -1e-6) and np.all(entropy <= np.log(10) + 1e-5)
        for j in range(len(y)):
            row = dict(index=index, true_label=int(y[j]), predicted_label=int(pred[j]),
                       confidence=float(confidence[j]), predictive_entropy=float(entropy[j]),
                       predictive_variance=float(variance[j].mean()))
            row.update({f'probability_{k}': float(mean[j, k]) for k in range(10)})
            row.update({f'variance_{k}': float(variance[j, k]) for k in range(10)})
            rows.append(row)
            index += 1
        labels.extend(y.tolist())
        predictions.extend(pred.tolist())
    metrics = classification(labels, predictions)
    metrics.update(average_confidence=float(np.mean([r['confidence'] for r in rows])),
                   average_predictive_entropy=float(np.mean([r['predictive_entropy'] for r in rows])),
                   average_predictive_variance=float(np.mean([r['predictive_variance'] for r in rows])))
    return metrics, rows
