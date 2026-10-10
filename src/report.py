"""Generate English report material directly from measured experiment outputs."""
import json
from src.utils import ROOT

REFERENCE = ('Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: '
             'Representing Model Uncertainty in Deep Learning. Proceedings of ICML, PMLR 48, 1050-1059. '
             'https://proceedings.mlr.press/v48/gal16.html')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def generate_material(p, visual_notes='', output_root=None, public=False):
    target = output_root or ROOT
    out = target / 'report_material'
    out.mkdir(parents=True, exist_ok=True)
    identity_path = ROOT / '.private/student.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8')) if identity_path.exists() and not public else {}
    student_name = identity.get('name', 'Withheld in public version')
    student_number = identity.get('number', 'Withheld in public version')
    info_path = ROOT / 'project_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
    webpage = info.get('repository_url', 'To be added after GitHub upload.')
    d, clean = p['deterministic'], p['noise_results'][0]
    algorithm = '''## Bayes theorem and neural networks

Bayes theorem updates a belief after observing data: P(w|D) = P(D|w)P(w) / P(D).
Here w denotes the neural network parameters and D is the training data. The prior P(w) describes beliefs before observing data. The likelihood P(D|w) describes how well the parameters explain the data. The posterior P(w|D) describes beliefs after observing the data.

A conventional neural network uses one fitted set of parameters. A Bayesian neural network instead represents uncertainty through a distribution over parameters. Its predictive distribution is P(y|x,D) = integral P(y|x,w)P(w|D) dw. This integral is usually too difficult to compute exactly for a deep network.

## Monte Carlo Dropout

Gal and Ghahramani (2016) connect dropout training to approximate Bayesian inference under specific modelling assumptions. This project uses a CNN with dropout as a practical approximation, rather than computing an exact posterior over all weights. Dropout remains active during inference, creating a different random mask on each forward pass. The predictive probability is approximated by p_bar = (1/T) sum_t p_t with T = 30.

The predicted class is argmax(p_bar), and confidence is max(p_bar). The class-wise predictive variance is (1/T) sum_t (p_t - p_bar)^2. A scalar variance summary averages these ten class-wise variances. Predictive entropy is H(p_bar) = -sum_k p_bar_k log(p_bar_k), using natural logarithms, so values are in nats and range from 0 to ln(10).

Low entropy indicates concentrated class probabilities; high entropy indicates a more ambiguous predictive distribution. Entropy reflects total predictive uncertainty and is not a pure measure of epistemic uncertainty. The variance measures disagreement across dropout predictions. Neither quantity guarantees calibrated confidence or reliable detection of every out-of-distribution input.

Uncertainty matters in pattern recognition because an application can flag ambiguous predictions for review instead of treating every classification as equally reliable. This experiment measures the behaviour of one approximation and does not establish suitability for safety-critical decisions.
'''
    algorithm = algorithm.replace('T = 30', f"T = {p['config']['mc_samples']}")
    config_rows = [(k, v) for k, v in p['environment'].items() if k != 'executable'] + list(p['config'].items()) + [('dataset', 'MNIST'), ('train_samples', p['train_samples']), ('test_samples', p['test_samples']), ('best_epoch', p['best_epoch'])]
    settings = table(['Setting', 'Value'], config_rows) + f"\n\nPixels remain in [0,1]. Noise is added before clipping to [0,1]. All six noise levels use all 10,000 test images and {p['config']['mc_samples']} MC passes. A fixed noise generator is independent of dropout sampling, and the same underlying noise draws are scaled across levels. Only one seed and one noise realization per image are evaluated.\n"
    e, c = p['environment'], p['config']
    report_settings = table(['Setting', 'Value'], [
        ['Python', e['python']], ['PyTorch', e['pytorch']], ['Torchvision', e['torchvision']],
        ['Operating system', e['operating_system']], ['Compute device', e['device']],
        ['CUDA available', e['cuda_available']], ['GPU used', e['gpu_name']],
        ['Dataset and samples', f"MNIST; {p['train_samples']} train / {p['test_samples']} test"],
        ['Epochs and batch size', f"{c['epochs']} epochs; batch size {c['batch_size']}"],
        ['Optimizer and learning rate', f"{c['optimizer']}; learning rate {c['learning_rate']}"],
        ['Loss', c['loss']], ['Random seed', c['seed']],
        ['Dropout rates', f"Convolution {c['dropout_convolution']}; fully connected {c['dropout_fc']}"],
        ['MC samples', c['mc_samples']], ['Input preprocessing', c['preprocessing']],
        ['CPU threads', c['cpu_threads']], ['Selected checkpoint epoch', p['best_epoch']],
    ]) + '\n\n' + settings.split('\n\n', 1)[1]
    classification_table = table(['Method', 'Accuracy', 'Macro precision', 'Macro recall', 'Macro F1'],
        [[name, f"{m['accuracy']:.4%}", f"{m['precision']:.6f}", f"{m['recall']:.6f}", f"{m['f1']:.6f}"] for name, m in [('Deterministic CNN', d), ('MC Dropout', clean)]])
    noise_table = table(['Noise SD', 'MC accuracy', 'Mean confidence', 'Mean entropy in nats'],
        [[f"{r['noise_level']:.1f}", f"{r['accuracy']:.4%}", f"{r['average_confidence']:.6f}", f"{r['average_predictive_entropy']:.6f}"] for r in p['noise_results']])
    examples_table = table(['Group', 'Index', 'True', 'Predicted', 'Confidence', 'Entropy'],
        [['Low' if i < 5 else 'High', r['index'], r['true_label'], r['predicted_label'], f"{r['confidence']:.6f}", f"{r['predictive_entropy']:.6g}"] for i, r in enumerate(p['selected_examples'])])
    low = p['selected_examples'][:5]
    high = p['selected_examples'][5:]
    wrong_low = sum(r['true_label'] != r['predicted_label'] for r in low)
    wrong_high = sum(r['true_label'] != r['predicted_label'] for r in high)
    entropy_monotone = all(b['average_predictive_entropy'] >= a['average_predictive_entropy'] for a, b in zip(p['noise_results'], p['noise_results'][1:]))
    last = p['noise_results'][-1]
    results = f'''## Training and clean test classification

The best deterministic test accuracy was {p['best_test_accuracy']:.4%}, reached at epoch {p['best_epoch']}. The final epoch test loss was {p['final_epoch_test_loss']:.6f}; the reloaded best checkpoint test loss was {d['loss']:.6f}. Training accuracy is measured with dropout active, whereas test accuracy is measured with dropout disabled, so the curves are not directly equivalent.

{classification_table}

![Training curves](../results/training_curves.png)

Figure 1. Training and test loss and accuracy over all training epochs.

![Confusion matrix](../results/confusion_matrix.png)

Figure 2. Deterministic predictions on the full MNIST test set. Each cell is an image count.

## MC Dropout and selected examples

On clean test images, mean confidence was {clean['average_confidence']:.6f}, mean predictive entropy was {clean['average_predictive_entropy']:.6f} nats, and mean class-wise predictive variance was {clean['average_predictive_variance']:.8f}. MC accuracy differed from deterministic accuracy by {100*(clean['accuracy']-d['accuracy']):+.3f} percentage points. Accuracy improvement is not the main objective of MC Dropout.

{examples_table}

![Uncertainty examples](../results/uncertainty_examples.png)

Figure 3. The five lowest entropy images above and the five highest entropy images below, selected from all test images.

There were {wrong_low} incorrect predictions in the five lowest entropy examples and {wrong_high} in the five highest entropy examples. These are selected extremes, not a representative estimate of the error rate. The visual form of the digits should be interpreted alongside the labels and measured probabilities; image clarity is not independently scored.

{visual_notes}

## Gaussian noise experiment

{noise_table}

![Noise experiment](../results/noisy_uncertainty.png)

Figure 4. MC accuracy and mean predictive entropy under increasing Gaussian noise standard deviation.

From noise SD 0 to 0.5, accuracy changed from {clean['accuracy']:.4%} to {last['accuracy']:.4%}, and mean entropy changed from {clean['average_predictive_entropy']:.6f} to {last['average_predictive_entropy']:.6f} nats. Mean entropy {'was nondecreasing' if entropy_monotone else 'was not monotonically nondecreasing'} across the measured noise levels. The reported values are retained without imposing an expected trend.

## Limitations

The best checkpoint was selected using test accuracy across {p['config']['epochs']} epochs. Repeated use of the test set can make the selected accuracy optimistic; an independent validation set should be used in a stronger evaluation. This assignment follows the requested protocol. Only one training seed, one model, and one noise realization per image were tested; no error bars or statistical significance claims are made. Gaussian noise changes the input distribution but does not represent all real handwriting degradation. Confidence calibration and out-of-distribution detection were not separately evaluated.
'''
    implementation = '''The AI assistant created the project code and implemented the small PyTorch CNN. After an initial D drive GPU environment attempt was stopped at the student's request, the experiment used the existing CPU PyTorch installation. Training uses cross entropy and Adam. The best checkpoint is reloaded before evaluation. Standard inference calls eval(), and MC inference activates only dropout modules while preserving the original module states afterwards.

The pipeline downloads MNIST to the project data directory, records training history, evaluates deterministic classification, scans test images with MC Dropout, ranks uncertainty examples, and repeats MC evaluation for six noise levels. Numeric CSV and JSON outputs are the source for the report. The program uses bounded batches to limit memory use.
'''
    discovery = '''The algorithm and MNIST dataset were specified in the student's supplied project brief. The AI assistant did not independently choose the algorithm. It verified the original MC Dropout paper and translated the proposed approach into an executable experiment. The paper provides the theoretical basis for relating dropout to approximate Bayesian inference.
'''
    learning = '''Reflection draft for the student to review and confirm.

This experiment illustrates that classification accuracy and predictive uncertainty answer different questions. A model can produce a label while assigning substantial probability to alternative classes. MC Dropout uses repeated predictions to summarize this uncertainty, although it is only an approximation to Bayesian inference.

The measured results show how the predictive distribution changes when the input is degraded. Entropy should be interpreted together with prediction correctness and confidence rather than treated as a guarantee of reliability. The evaluation also illustrates why selecting a checkpoint with test data limits the strength of the final accuracy claim.

AI assistance still requires checking the environment, verifying the mathematical definitions, inspecting real outputs, and documenting limitations. The student should adapt this reflection to their own understanding before submission.
'''
    report = f'''# MNIST Classification and Uncertainty with Monte Carlo Dropout

CISC3024 Pattern Recognition

AI Assignment #2

Student Name: {student_name}

Student Number: {student_number}

This report studies handwritten digit classification and predictive uncertainty using a small CNN with Monte Carlo Dropout. The deterministic clean test accuracy was {d['accuracy']:.4%}; the MC accuracy was {clean['accuracy']:.4%}. The experiments compare clean images with Gaussian noise and discuss the limitations of the measurements.

# 1 How AI finds the algorithm

{discovery}
# 2 Algorithm Description

{algorithm}
# 3 How AI implements the algorithm

{implementation}
# 4 Experiment Settings and Results

## Experiment settings

{report_settings}
{results}
# 5 What I Have Learnt

{learning}
# 6 Source Code Webpage Link

{webpage}

# References

{REFERENCE}

PyTorch. Previous PyTorch Versions. https://pytorch.org/get-started/previous-versions/

NVIDIA. CUDA Toolkit 12.4 Release Notes. https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html
'''
    for filename, content in [('algorithm_summary.md', algorithm+'\n'+REFERENCE), ('experiment_settings.md', settings),
                               ('experiment_results.md', results), ('final_report.md', report)]:
        (out / filename).write_text(content, encoding='utf-8')
    summary = f"Best Test Accuracy: {p['best_test_accuracy']:.6f}\nBest Epoch: {p['best_epoch']}\nFinal Test Loss: {p['final_epoch_test_loss']:.6f}\nBest Checkpoint Test Loss: {d['loss']:.6f}\nPrecision (macro): {d['precision']:.6f}\nRecall (macro): {d['recall']:.6f}\nF1-score (macro): {d['f1']:.6f}\nMC Dropout Samples: {p['config']['mc_samples']}\nMC Accuracy: {clean['accuracy']:.6f}\nAverage Predictive Entropy: {clean['average_predictive_entropy']:.6f}\nAverage Confidence: {clean['average_confidence']:.6f}\n"
    summary += '\n'.join(f"Noise {r['noise_level']:.1f} Accuracy: {r['accuracy']:.6f}; Confidence: {r['average_confidence']:.6f}; Entropy: {r['average_predictive_entropy']:.6f}" for r in p['noise_results'])
    (target / 'results/summary.txt').write_text(summary+'\n', encoding='utf-8')
