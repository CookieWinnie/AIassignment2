## Training and clean test classification

The best deterministic test accuracy was 99.2600%, reached at epoch 10. The final epoch test loss was 0.023009; the reloaded best checkpoint test loss was 0.023009. Training accuracy is measured with dropout active, whereas test accuracy is measured with dropout disabled, so the curves are not directly equivalent.

| Method | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Deterministic CNN | 99.2600% | 0.992524 | 0.992541 | 0.992525 |
| MC Dropout | 99.2200% | 0.992100 | 0.992120 | 0.992101 |

![Training curves](../results/training_curves.png)

Figure 1. Training and test loss and accuracy over all training epochs.

![Confusion matrix](../results/confusion_matrix.png)

Figure 2. Deterministic predictions on the full MNIST test set. Each cell is an image count.

## MC Dropout and selected examples

On clean test images, mean confidence was 0.977782, mean predictive entropy was 0.082582 nats, and mean class-wise predictive variance was 0.00117026. MC accuracy differed from deterministic accuracy by -0.040 percentage points. Accuracy improvement is not the main objective of MC Dropout.

| Group | Index | True | Predicted | Confidence | Entropy |
| --- | --- | --- | --- | --- | --- |
| Low | 5385 | 2 | 2 | 1.000000 | 2.50316e-07 |
| Low | 1890 | 2 | 2 | 1.000000 | 5.2089e-07 |
| Low | 8925 | 4 | 4 | 1.000000 | 7.29197e-07 |
| Low | 3397 | 4 | 4 | 1.000000 | 1.03491e-06 |
| Low | 8835 | 5 | 5 | 1.000000 | 2.68095e-06 |
| High | 3503 | 9 | 8 | 0.284507 | 1.64688 |
| High | 6572 | 1 | 1 | 0.411596 | 1.64274 |
| High | 3316 | 7 | 7 | 0.460342 | 1.60261 |
| High | 726 | 7 | 5 | 0.304703 | 1.6023 |
| High | 9679 | 6 | 3 | 0.297955 | 1.55788 |

![Uncertainty examples](../results/uncertainty_examples.png)

Figure 3. The five lowest entropy images above and the five highest entropy images below, selected from all test images.

There were 0 incorrect predictions in the five lowest entropy examples and 3 in the five highest entropy examples. These are selected extremes, not a representative estimate of the error rate. The visual form of the digits should be interpreted alongside the labels and measured probabilities; image clarity is not independently scored.

The five lowest entropy samples show recognizable 2, 2, 4, 4 and 5 shapes, and all five predictions are correct. The high entropy group includes a faint, disconnected digit 1 (index 6572), an unusual looped digit 7 (index 3316), and a disconnected digit 6 (index 9679). These visible forms suggest possible ambiguity, but clarity was not measured independently.

The highest entropy sample (index 3503) has true label 9 and predicted label 8, with confidence 0.284507. The model also confuses a 7 with a 5 (index 726) and a 6 with a 3 (index 9679). Three of the five high entropy predictions are incorrect, whereas two remain correct. High uncertainty therefore indicates ambiguity, not necessarily an incorrect prediction.

These observations refer to the checkpoint and selected examples generated in this run. They must be reviewed again if the model is retrained. Values in the figures and report tables are rounded for display; the per-image CSV retains full numeric precision.


## Gaussian noise experiment

| Noise SD | MC accuracy | Mean confidence | Mean entropy in nats |
| --- | --- | --- | --- |
| 0.0 | 99.2200% | 0.977782 | 0.082582 |
| 0.1 | 99.1800% | 0.973743 | 0.100240 |
| 0.2 | 98.8600% | 0.954034 | 0.178327 |
| 0.3 | 95.8100% | 0.862818 | 0.473114 |
| 0.4 | 76.8800% | 0.683602 | 0.933644 |
| 0.5 | 48.4600% | 0.581321 | 1.175818 |

![Noise experiment](../results/noisy_uncertainty.png)

Figure 4. MC accuracy and mean predictive entropy under increasing Gaussian noise standard deviation.

From noise SD 0 to 0.5, accuracy changed from 99.2200% to 48.4600%, and mean entropy changed from 0.082582 to 1.175818 nats. Mean entropy was nondecreasing across the measured noise levels. The reported values are retained without imposing an expected trend.

## Limitations

The best checkpoint was selected using test accuracy across 10 epochs. Repeated use of the test set can make the selected accuracy optimistic; an independent validation set should be used in a stronger evaluation. This assignment follows the requested protocol. Only one training seed, one model, and one noise realization per image were tested; no error bars or statistical significance claims are made. Gaussian noise changes the input distribution but does not represent all real handwriting degradation. Confidence calibration and out-of-distribution detection were not separately evaluated.
