# nninfo
#### A Python Package for the Analysis of Deep Neural Networks using Information Theory
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)]()


The package has been developed for the analysis of deep neural networks using different information-theoretic tools, especially Partial Information Decomposition.

It has been developed as part of the publication [*A Measure of the Complexity of Neural Representations based on Partial Information Decomposition*](https://openreview.net/pdf?id=R8TU3pfzFr) which appeared in May 2023 in Transactions on Machine Learning Research (TMLR):

```
@article{ehrlich2023measure,
  title={A Measure of the Complexity of Neural Representations based on Partial Information Decomposition},
  author={Ehrlich, David A. and Schneider, Andreas C. and Priesemann, Viola and Wibral, Michael and Makkeh, Abdullah},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

### Contributions

The *nninfo* software package was originally envisioned and developed by [Andreas C. Schneider](https://github.com/ac-schneider). The package has been majorly refactored and extended for the TMLR publication by [David A. Ehrlich](https://github.com/daehrlich). We further thank [Valentin Neuhaus](https://github.com/vneuhaus) for his contributions to the code. The *nninfo* package is actively maintained by Andreas C. Schneider and David A. Ehrlich.

### Installation
All requirements for the *nninfo* package are summarized in the *env.yaml* file.

We recommend setting up a new virtual environment using the conda command:

    conda env create -f env.yaml
    conda activate nninfo

Afterwards, the package can be installed using pip:

    pip install -e .

### Scripts for reproducing research results
To recreate the main figures in the TMLR paper, we provide five scripts, which can be found in the *tmlr_scripts* directory:
- *1_tmlr_mnist_8levels_onehot.ipynb* creates Figure 3.B
- *2_tmlr_mnist_8levels_binary.ipynb* creates Figure 4.A
- *3_tmlr_mnist_4levels_onehot.ipynb* creates Figures 5.B and 5.D
- *4_tmlr_cifar.ipynb* creates Figure 4.B


The scripts include the whole experiment pipeline from training over analysis to plotting. Note that the software package has been majorly refactored since the submission of the paper and the scripts have been adapted to the new version.

As the empirical results (.ipynb files 1-4) of our work were computed on a scientific compute cluster over several days, we recommend not to rerun the experiments entirely but rather to take the scripts as a guideline to examine the nninfo package. 

To make the examination of the experiment pipeline easier, we include a fifth script that runs the analysis on a toy network and dataset:

- *5_tmlr_demo.ipynb* demo script using the Task by Tishby et al.

 This demo script can be run entirely in approximately two hours on a modern desktop computer.