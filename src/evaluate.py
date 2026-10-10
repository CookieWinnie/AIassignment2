import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def classification(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(10)), average='macro', zero_division=0)
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(p), recall=float(r), f1=float(f))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    loss_sum = 0.0
    for x, y in loader:
        logits = model(x.to(device))
        loss_sum += torch.nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
        labels.extend(y.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    metrics = classification(labels, predictions)
    metrics['loss'] = loss_sum / len(labels)
    return metrics, confusion_matrix(labels, predictions, labels=np.arange(10))
