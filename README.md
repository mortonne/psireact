# PsiReact
[![PyPI version](https://badge.fury.io/py/psireact.svg)](https://badge.fury.io/py/psireact)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4282758.svg)](https://doi.org/10.5281/zenodo.4282758)

Hierarchical Bayesian modeling of response time data.

This package currently implements the linear ballistic accumulator (LBA) model (Brown & Heathcote 2008), with potential for other decision models to be added later.
LBA has a closed-form solution, making it mathmatically tractable, while capturing many important properties of response time distributions.
Importantly, while many response time models only support simulation of tasks with only two response options, LBA can simulate tasks with many response options.

<p align="center">
  <img src="https://github.com/mortonne/psireact/blob/master/jupyter/lba_24afc.png" alt="probability density function" width="400">
</p>

In this example, there are 24 possible responses with different levels of support.
Each curve shows the probability density function for one response according to the LBA model.
The overall height of each curve reflects the probability of that response, and the shape reflects the probability of different response times.

PsiReact can be used to:
 * Fit data to estimate model parameters
 * Use hierarchical models to estimate both group-level tendencies and individual differences
 * Generate simulated response time data for analysis
 * Compare different models of response behavior

## Installation

First, it is strongly recommended that you set up either a [Conda environment](https://conda.io/en/latest/) or a [Python virtual environment](https://docs.python.org/3/library/venv.html). 
This helps keep packages installed for different projects separate. 
For example, using Conda:

```bash
conda create -n psireact python=3.8
conda activate psireact
```

You can install the latest stable version of PsiReact using pip:

```bash
pip install psireact
```

You can also install the development version directly from the code
repository on GitHub:

```bash
pip install git+git://github.com/mortonne/psireact
```

## Getting started

To get an intuition for what sort of behavior the LBA model can produce, look at the [LBA demo notebook](https://github.com/mortonne/psireact/blob/master/jupyter/lba_demo.ipynb). 
The LBA model can be used to both generate simulated data and to estimate parameters based on observed data. 
See the [LBA parameter recovery](https://github.com/mortonne/psireact/blob/master/jupyter/lba_recovery.ipynb) notebook for examples.

## Road map

Directions for future development:
 * notebook illustrating definition and use of a hierarchical model
 * support for additional response time models
 * testing and demo of maximum likelihood estimation

## Citation

If you use PsiReact, please cite the following:

 * Morton NW, Schlichting ML, Preston AR. In press. Representations of common event structure in medial temporal lobe and frontoparietal cortex support efficient inference.
 * Morton NW. 2020. PsiReact: Modeling of response time data. Zenodo. http://doi.org/10.5281/zenodo.4282758
