# Deeply Uncertain

Code repository for the paper "Deeply Uncertain: Comparing Methods of Uncertainty Quantification in Deep Learning Algorithms", by @joaocaldeira and @bnord. The paper can be found at <a href="https://arxiv.org/abs/2004.10710">arXiv:2004.10710</a>.

## Paper Abstract

We present a comparison of methods for uncertainty quantification (UQ) in deep learning algorithms in the context of a simple physical system. Three of the most common uncertainty quantification methods - Bayesian Neural Networks (BNN), Concrete Dropout (CD), and Deep Ensembles (DE) - are compared to the standard analytic error propagation. We discuss this comparison in terms endemic to both machine learning ("epistemic" and "aleatoric") and the physical sciences ("statistical" and "systematic"). The comparisons are presented in terms of simulated experimental measurements of a single pendulum - a prototypical physical system for studying measurement and analysis techniques. Our results highlight some pitfalls that may occur when using these UQ methods. For example, when the variation of noise in the training set is small, all methods predicted the same relative uncertainty independently of the inputs. This issue is particularly hard to avoid in BNN. On the other hand, when the test set contains samples far from the training distribution, we found that no methods sufficiently increased the uncertainties associated to their predictions. This problem was particularly clear for CD. In light of these results, we make some recommendations for usage and interpretation of UQ methods. 

## Reproducing the results

To reproduce the results in the paper, run the script `run_training.sh` followed by `run_tests.sh` and all the cells in the notebook `make_plots.ipynb`. To recreate the environment, you can install packages from the `requirements.txt` file, or use the docker image `tensorflow/tensorflow:latest-gpu-py3-jupyter` and install `scikit-learn`, `tensorflow-probability`, and `matplotlib==3.2`.

## Basic explanation of code

1. data_gen.py: generates simulations of pendulum swing data
2. preprocssin.py: rescales data 
3. train_network.py: generates data and trains one kind of UQ neural network
4. test_network.py: evaluates one kind of trained neural network
5. run_train.sh: run all model training to replicate the paper results
6. run_test.sh: test all trained models to replicate the paper results
7. make_plots.ipynb: notebook to generate plots tested and trained models
8. models/cd.py: concrete dropout
9. models/mlp_tf: flipout for BNN model

