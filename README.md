# psireact
Hierarchical Bayesian modeling of response time data.

## Installation

First, it is strongly recommended that you set up either a [Conda environment](https://conda.io/en/latest/) or a [Python virtual environment](https://docs.python.org/3/library/venv.html). This helps keep packages installed for different projects separate.

Once you've activated the environment you'll be using, install [Theano](http://deeplearning.net/software/theano/install.html). Theano compiles code on the fly, so installation is relatively system dependent. Follow the instructions on the Theano page. Next, install [pyMC3](https://docs.pymc.io/).

Finally, download the source code for PsiReact and install:

```bash
git clone git@github.com:mortonne/psireact.git
cd psireact
python setup.py install
```
