import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import ROOT

plt.rcParams.update({'font.size': 11, 'savefig.dpi': 180})


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / 'results' / name)
    plt.close(fig)


def training_curves(history):
    epochs = [r['epoch'] for r in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for key, label in [('train_loss', 'Training'), ('test_loss', 'Test')]:
        ax[0].plot(epochs, [r[key] for r in history], marker='o', label=label)
    for key, label in [('train_accuracy', 'Training with dropout'), ('test_accuracy', 'Test without dropout')]:
        ax[1].plot(epochs, [100*r[key] for r in history], marker='o', label=label)
    for a, title, ylabel in zip(ax, ['Cross entropy loss', 'Classification accuracy'], ['Loss', 'Accuracy (%)']):
        a.set(title=title, xlabel='Epoch', ylabel=ylabel)
        a.legend()
        a.grid(alpha=0.25)
    finish(fig, 'training_curves.png')


def confusion(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel='Predicted digit', ylabel='True digit', title='Deterministic CNN confusion matrix')
    fig.colorbar(im, ax=ax, label='Number of test images')
    finish(fig, 'confusion_matrix.png')


def examples(dataset, rows):
    ordered = sorted(rows, key=lambda r: (r['predictive_entropy'], r['index']))
    selected = ordered[:5] + ordered[-5:][::-1]
    fig, axes = plt.subplots(2, 5, figsize=(12, 6.6))
    for ax, row in zip(axes.flat, selected):
        ax.imshow(dataset[row['index']][0].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"#{row['index']} True {row['true_label']} Pred {row['predicted_label']}\nConf {row['confidence']:.3f} H {row['predictive_entropy']:.3g}", fontsize=10, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Lowest entropy (top) and highest entropy (bottom) test examples')
    fig.subplots_adjust(top=0.84, bottom=0.05, hspace=0.6, wspace=0.18)
    fig.savefig(ROOT / 'results/uncertainty_examples.png')
    plt.close(fig)
    return selected


def noise_curve(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    levels = [r['noise_level'] for r in rows]
    axes[0].plot(levels, [100*r['accuracy'] for r in rows], marker='o')
    axes[1].plot(levels, [r['average_predictive_entropy'] for r in rows], marker='o', color='darkorange')
    for ax, title, ylabel in zip(axes, ['MC classification under Gaussian noise', 'MC predictive uncertainty under noise'], ['Accuracy (%)', 'Mean entropy (nats)']):
        ax.set(xlabel='Gaussian noise standard deviation', ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
    finish(fig, 'noisy_uncertainty.png')
