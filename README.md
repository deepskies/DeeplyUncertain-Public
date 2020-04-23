# Deeply Uncertain

Code repository for the paper "Deeply Uncertain: Comparing Methods of Uncertainty Quantification in Deep Learning Algorithms", by @joaocaldeira and @bnord. The paper can be found at <a href="https://arxiv.org/abs/2004.10710">arXiv:2004.10710</a>.

## Abstract

We present a comparison of methods for uncertainty quantification (UQ) in deep learning algorithms in the context of a simple physical system. Three of the most common uncertainty quantification methods - Bayesian Neural Networks (BNN), Concrete Dropout (CD), and Deep Ensembles (DE) - are compared to the standard analytic error propagation. We discuss this comparison in terms endemic to both machine learning ("epistemic" and "aleatoric") and the physical sciences ("statistical" and "systematic"). The comparisons are presented in terms of the results on a single pendulum experiment, which is a prototypical physics experiment. Our results highlight some pitfalls that may occur when using these UQ methods. For instance, when the variation of noise in the training set is small, all methods predicted the same relative uncertainty independently of the inputs. This state is particularly hard to avoid in BNN. On the other hand, when the test set contains samples far from the training distribution, we found that no methods sufficiently increased the uncertainties associated to their predictions. This was particularly true for CD. In light of these results, we make some recommendations for usage and interpretation of UQ methods.

## Reproducing the results

To reproduce the results in the paper, run the script `run_training.sh` followed by `run_tests.sh` and all the cells in the notebook `make_plots.ipynb`. To recreate the environment, you can install packages from the `requirements.txt` file, or use the docker image `tensorflow/tensorflow:latest-gpu-py3-jupyter` and install `scikit-learn`, `tensorflow-probability`, and `matplotlib==3.2`.
