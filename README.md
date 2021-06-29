# CDE-diagnostics

The ''CDE-diagnostics'' package provides tools for validating the quality of conditional density estimators in Python. It contains source code for the paper ''Diagnostics for Conditional Density Estimators and Bayesian Inference Algorithms'' (https://arxiv.org/abs/2102.10473).

Author: David Zhao


## Installation

CDE-diagnostics is available from PyPI; to install use

```{shell}
pip install cde-diagnostics
```

## References

If you use this code in your research, please cite:

```
@InProceedings{zhao2021diagnostics,
    title = {{Diagnostics for conditional density models and Bayesian inference algorithms}},
    author = {Zhao, David and Dalmasso, Niccol\`{o} and Izbicki, Rafael and Lee, Ann B.},
    abstract = {There has been growing interest in the AI community for precise uncertainty quantification. Conditional density models f(y|x), where x represents potentially high-dimensional features, are an integral part of uncertainty quantification in prediction and Bayesian inference. However, it is challenging to assess conditional density estimates and gain insight into modes of failure. While existing diagnostic tools can determine whether an approximated conditional density is compatible overall with a data sample, they lack a principled framework for identifying, locating, and interpreting the nature of statistically significant discrepancies over the entire feature space. In this paper, we present rigorous and easy-to-interpret diagnostics such as (i) the ``Local Coverage Test'' (LCT), which distinguishes an arbitrarily misspecified model from the true conditional density of the sample, and (ii) ``Amortized Local P-P plots'' (ALP) which can quickly provide interpretable graphical summaries of distributional differences at any location x in the feature space. Our validation procedures scale to high dimensions and can potentially adapt to any type of data at hand. We demonstrate the effectiveness of LCT and ALP through a simulated experiment and applications to prediction and parameter inference for image data.},
    booktitle = {Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI)},
    year = {2021},
    volume = {125},
    series = {Proceedings of Machine Learning Research},
    month = {26--29 Jul},
    publisher = {PMLR}
}
```
