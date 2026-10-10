## Bayes theorem and neural networks

Bayes theorem updates a belief after observing data: P(w|D) = P(D|w)P(w) / P(D).
Here w denotes the neural network parameters and D is the training data. The prior P(w) describes beliefs before observing data. The likelihood P(D|w) describes how well the parameters explain the data. The posterior P(w|D) describes beliefs after observing the data.

A conventional neural network uses one fitted set of parameters. A Bayesian neural network instead represents uncertainty through a distribution over parameters. Its predictive distribution is P(y|x,D) = integral P(y|x,w)P(w|D) dw. This integral is usually too difficult to compute exactly for a deep network.

## Monte Carlo Dropout

Gal and Ghahramani (2016) connect dropout training to approximate Bayesian inference under specific modelling assumptions. This project uses a CNN with dropout as a practical approximation, rather than computing an exact posterior over all weights. Dropout remains active during inference, creating a different random mask on each forward pass. The predictive probability is approximated by p_bar = (1/T) sum_t p_t with T = 30.

The predicted class is argmax(p_bar), and confidence is max(p_bar). The class-wise predictive variance is (1/T) sum_t (p_t - p_bar)^2. A scalar variance summary averages these ten class-wise variances. Predictive entropy is H(p_bar) = -sum_k p_bar_k log(p_bar_k), using natural logarithms, so values are in nats and range from 0 to ln(10).

Low entropy indicates concentrated class probabilities; high entropy indicates a more ambiguous predictive distribution. Entropy reflects total predictive uncertainty and is not a pure measure of epistemic uncertainty. The variance measures disagreement across dropout predictions. Neither quantity guarantees calibrated confidence or reliable detection of every out-of-distribution input.

Uncertainty matters in pattern recognition because an application can flag ambiguous predictions for review instead of treating every classification as equally reliable. This experiment measures the behaviour of one approximation and does not establish suitability for safety-critical decisions.

Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. Proceedings of ICML, PMLR 48, 1050-1059. https://proceedings.mlr.press/v48/gal16.html