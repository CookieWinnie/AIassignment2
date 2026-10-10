"""Run the complete MNIST MC Dropout experiment using only project output paths."""
import sys
sys.dont_write_bytecode = True
from src.utils import ROOT, configure_paths, seed_all, write_csv, write_json, environment
configure_paths()
import argparse
import csv
import time
import os
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from src.model import BayesianCNN
from src.train import train
from src.evaluate import evaluate
from src.uncertainty import scan
from src import plots
from src.report import generate_material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--mc-samples', type=int, default=30)
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.mc_samples < 2 or args.threads < 1:
        parser.error('epochs/batch-size must be positive; mc-samples must be >= 2')
    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    config = dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.001,
                  optimizer='Adam', loss='CrossEntropyLoss', seed=42, mc_samples=args.mc_samples,
                  dropout_convolution=0.25, dropout_fc=0.5, preprocessing='ToTensor only; pixels in [0,1]')
    config['cpu_threads'] = args.threads
    seed_all(config['seed'])
    device = torch.device('cpu')  # The student explicitly selected the existing CPU environment.
    print(f'Device: {device} | Project: {ROOT}', flush=True)
    training = MNIST(str(ROOT / 'data'), train=True, download=True, transform=ToTensor())
    testing = MNIST(str(ROOT / 'data'), train=False, download=True, transform=ToTensor())
    assert len(training) == 60000 and len(testing) == 10000
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(42), pin_memory=device.type == 'cuda')
    test_loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             pin_memory=device.type == 'cuda')
    model = BayesianCNN().to(device)
    if args.skip_training:
        if not (ROOT / 'models/best_model.pth').exists() or not (ROOT / 'results/training_history.csv').exists():
            raise FileNotFoundError('--skip-training requires best_model.pth and training_history.csv')
        checkpoint = torch.load(ROOT / 'models/best_model.pth', map_location=device, weights_only=True)
        saved = checkpoint['config']
        for key in ['batch_size', 'learning_rate', 'seed', 'dropout_convolution', 'dropout_fc', 'preprocessing']:
            if saved[key] != config[key]:
                raise ValueError(f'Incompatible checkpoint setting: {key}')
        config['epochs'] = saved['epochs']
        with (ROOT / 'results/training_history.csv').open(encoding='utf-8') as f:
            history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        if len(history) != config['epochs']:
            raise ValueError('Incomplete training history; rerun training')
    else:
        history = train(model, train_loader, test_loader, device, config)
        checkpoint = torch.load(ROOT / 'models/best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    deterministic, cm = evaluate(model, test_loader, device)
    plots.training_curves(history)
    plots.confusion(cm)
    write_csv(ROOT / 'results/confusion_matrix.csv', [dict(true_label=i, **{f'predicted_{j}': int(cm[i, j]) for j in range(10)}) for i in range(10)])
    noise_results, clean_rows = [], None
    for noise in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        metrics, rows = scan(model, test_loader, device, args.mc_samples, noise, 42)
        noise_results.append(dict(noise_level=noise, **metrics))
        print(f"Noise {noise:.1f}: accuracy {metrics['accuracy']:.4%}, confidence {metrics['average_confidence']:.4f}, entropy {metrics['average_predictive_entropy']:.4f}", flush=True)
        write_csv(ROOT / f'results/uncertainty_noise_{noise:.1f}.csv', rows)
        if noise == 0:
            clean_rows = rows
            write_csv(ROOT / 'results/uncertainty_samples.csv', rows)
    selected = plots.examples(testing, clean_rows)
    plots.noise_curve(noise_results)
    fields = ['accuracy', 'precision', 'recall', 'f1', 'average_confidence', 'average_predictive_entropy', 'average_predictive_variance']
    combined = [dict(method='deterministic', noise_level=0, **{k: deterministic.get(k, '') for k in fields})]
    combined += [dict(method='mc_dropout', **r) for r in noise_results]
    write_csv(ROOT / 'results/results.csv', combined)
    payload = dict(config=config, environment=environment(device), train_samples=len(training), test_samples=len(testing),
                   best_epoch=checkpoint['best_epoch'], best_test_accuracy=checkpoint['best_test_accuracy'],
                   final_epoch_test_loss=history[-1]['test_loss'], deterministic=deterministic,
                   noise_results=noise_results, selected_examples=selected, elapsed_seconds=time.perf_counter()-start)
    write_json(ROOT / 'results/metrics.json', payload)
    generate_material(payload)
    print('Completed experiment. Outputs: results/ and report_material/', flush=True)


if __name__ == '__main__':
    main()
