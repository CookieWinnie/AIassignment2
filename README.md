# MNIST Classification and Uncertainty with Monte Carlo Dropout

CISC3024 Pattern Recognition — AI Assignment #2

This project classifies MNIST handwritten digits using a small CNN and estimates predictive uncertainty through 30 Monte Carlo Dropout passes. It compares deterministic classification, high and low uncertainty images, and Gaussian noise with standard deviations 0 to 0.5. All reported numbers are measured by the program.

## Environment and run

The completed experiment uses the existing Anaconda Python and CPU PyTorch. No GPU installation is needed. On this computer, run from PowerShell:

```powershell
cd <project-folder>
python -B verify.py
python -B -u main.py
```

For another computer, install the versions recorded in `requirements.txt` inside your own chosen environment. CPU PyTorch wheels use the official CPU index. Review available versions for that platform before installation. The project does not modify drivers or system settings.

## Options

```powershell
python -B -u main.py --skip-training
python -B -u main.py --threads 4
```

Default settings: 10 epochs, batch size 128, Adam learning rate 0.001, cross entropy, seed 42, and 30 MC samples. `--skip-training` requires a compatible checkpoint and complete training history. Default execution trains again and replaces generated outputs. Run directories are not automatically deleted. Each experiment is a single-seed measurement; exact results may vary across platforms.

## Model and probability theory

Two 3×3 convolutions (1→32→64), each with ReLU, max pooling and dropout 0.25, feed a 128-unit hidden layer with dropout 0.5 and ten output logits. There is no BatchNorm. During MC inference only dropout modules are active, and prior module states are restored afterwards.

Mean softmax probabilities determine the MC prediction and confidence. Predictive entropy uses natural logs and measures total predictive uncertainty. Population variance is saved for each class, with the mean over classes as a scalar summary. MC Dropout is approximate Bayesian inference, not an exact posterior calculation. See [Gal and Ghahramani 2016](https://proceedings.mlr.press/v48/gal16.html).

## Data and evaluation

MNIST downloads only to `data/`: 60,000 training and 10,000 test images, each 1×28×28. ToTensor maps pixels to [0,1]; no further normalization is applied. Noise is added in pixel space and clipped to [0,1]. A separate seeded generator controls noise, with matched underlying noise draws across levels. Noise zero reuses the clean MC evaluation.

Accuracy, macro precision, macro recall and macro F1 use all test images. The best checkpoint is selected by test accuracy as requested by the assignment brief. This reuses test data for model selection and can produce optimistic estimates; a separate validation set is recommended for stronger evaluation. Training accuracy has dropout enabled, so it is not directly comparable with deterministic test accuracy.

## Structure and outputs

```text
main.py                    complete experiment entry point
verify.py                  small-scale correctness checks
src/model.py               CNN
src/train.py               training and checkpoint selection
src/evaluate.py            deterministic metrics
src/uncertainty.py         MC inference and noise scans
src/plots.py               report figures
src/report.py              measured report material
src/utils.py               paths, random seeds and serialization
data/                      MNIST files (ignored by Git)
models/best_model.pth      checkpoint (ignored by Git)
results/                   measured CSV, JSON, summary and figures
report_material/           English Markdown and Word report
build_report.py            Word report builder using bundled document runtime
```

Figures: `training_curves.png`, `confusion_matrix.png`, `uncertainty_examples.png`, and `noisy_uncertainty.png`. Tables: `training_history.csv`, `confusion_matrix.csv`, `uncertainty_samples.csv`, six `uncertainty_noise_*.csv` files, and `results.csv`. `metrics.json` captures settings and measured results; `summary.txt` is a readable summary. All accuracy values in CSV are fractions, not percentages; entropy is in nats.

The final report contains the six assignment sections. Its learning reflection is a draft for student confirmation. The source webpage uses the repository URL recorded in project_info.json. No remote repository is created or pushed automatically.

`make_equations.py` uses the existing ML environment to create formula images for the Word report. The report text and tables are editable; formulas are high-resolution images because native equation rendering substituted some symbols during QA. The Markdown version retains the formula definitions in text. Regenerate formula assets before running `build_report.py` if they are missing.

All output/cache/temp paths are inside the project. No bulk deletion is used. The Word builder is separate from the ML dependencies and uses the Codex bundled document runtime rather than installing document packages into Anaconda.

Personal identification is read only from the ignored `.private/student.json` file when generating a local submission report. The public report omits personal identification.
