# PsiReact
Hierarchical Bayesian modeling of response time data.

This package currently implements the linear ballistic accumulator (LBA) model (Brown & Heathcote 2008), with potential for other decision models to be added later. LBA has a closed-form solution, making it mathmatically tractable, while capturing many important properties of response time distributions. Importantly, while many response time models only support simulation of tasks with only two response options, LBA can simulate tasks with many response options.

## Installation

First, it is strongly recommended that you set up either a [Conda environment](https://conda.io/en/latest/) or a [Python virtual environment](https://docs.python.org/3/library/venv.html). This helps keep packages installed for different projects separate. For example, using Conda:

```bash
conda create -n psireact python=3.8
conda activate psireact
```

Once you've activated the environment you'll be using, install [Theano](http://deeplearning.net/software/theano/install.html). Theano compiles code on the fly, so installation is relatively system dependent. Follow the instructions on the Theano page. Next, install [PyMC3](https://docs.pymc.io/).

Finally, download the source code for PsiReact and install:

```bash
git clone git@github.com:mortonne/psireact.git
cd psireact
python setup.py install
```

## Getting started

To get an intuition for what sort of behavior the LBA model can produce, look at the [LBA demo notebook](https://github.com/mortonne/psireact/blob/master/jupyter/lba_demo.ipynb). The LBA model can be used to both generate simulated data and to estimate parameters based on observed data. See the [LBA parameter recovery](https://github.com/mortonne/psireact/blob/master/jupyter/lba_recovery.ipynb) notebook for examples.

## Road map

Directions for future development:
 * notebook illustrating definition and use of a hierarchical model
 * support for additional response time models
 * testing and demo of maximum likelihood estimation
 * update to use PyMC4 when available; this will switch the back end from Theano to TensorFlow

## Citation

If you use PsiReact for research, please cite the following paper:

Morton NW, Schlichting ML, Preston AR. In revision. Representations of common event structure in medial temporal lobe and frontoparietal cortex support efficient inference.
