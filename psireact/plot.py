"""Plot observed or simulated reaction time data."""

import numpy as np
import pandas as pd
import seaborn as sns


def plot_fit(data, sim):
    """Plot fit as a function of test type and response."""

    data.loc[:, 'source'] = 'Data'
    sim.loc[:, 'source'] = 'Model'
    full = pd.concat((data, sim), join='inner', ignore_index=True)
    full.loc[:, 'order'] = full.loc[:, 'response']
    full.loc[full.order == 0, 'order'] = 2
    g = sns.FacetGrid(full, col='test', hue='source', row='order', sharey=True)
    g.map(sns.distplot, 'rt', norm_hist=True, kde=False,
          bins=np.linspace(0, 15, 16))
    g.axes[0, 0].legend()
    g.set_titles('')
    g.set_xlabels('Reaction time')
    g.axes[0, 0].set_ylabel('Correct (relative frequency)')
    g.axes[1, 0].set_ylabel('Incorrect (relative frequency)')
    g.axes[0, 0].set_title('Direct')
    g.axes[0, 1].set_title('Inference')
    return g


def plot_fit_subj(data, sim, test=None):
    """Plot fit by subject."""

    data.loc[:, 'source'] = 'Data'
    sim.loc[:, 'source'] = 'Model'
    full = pd.concat((data, sim), join='inner', ignore_index=True)
    if test is not None:
        full = full.loc[full.test == test]

    g = sns.FacetGrid(full, col='subj_idx', col_wrap=5, hue='source', height=2)
    g.map(sns.distplot, 'rt', norm_hist=True, kde=False,
          bins=np.linspace(0, 15, 16))
    g.set_ylabels('Relative frequency')
    g.set_xlabels('Reaction time')
    g.set_titles('')
    g.axes[0].legend(['Data', 'Model'])
    return g


def plot_fit_scatter(data, sim):
    """Plot mean RT fit by subject."""

    d1 = data.groupby(['subj_idx', 'test']).mean()
    d1.loc[:, 'data'] = d1.loc[:, 'rt']
    d2 = sim.groupby(['subj_idx', 'test']).mean()
    d2.loc[:, 'model'] = d2.loc[:, 'rt']
    lim = np.max([np.max(d1.rt), np.max(d2.rt)])
    lim += lim / 20
    lim = np.ceil(lim)
    full = pd.concat((d1, d2), axis=1)
    full.reset_index(inplace=True)
    g = sns.FacetGrid(full, col='test')
    g.axes[0, 0].plot([0, lim], [0, lim], '-', color='gray')
    g.axes[0, 1].plot([0, lim], [0, lim], '-', color='gray')
    g.map_dataframe(sns.scatterplot, x='data', y='model')
    g.axes[0, 0].set_title('Direct')
    g.axes[0, 1].set_title('Inference')
    g.axes[0, 0].set_aspect('equal', 'box')
    g.axes[0, 1].set_aspect('equal', 'box')
    ticks = g.axes[0, 0].get_yticks()
    g.set_xlabels('Data')
    g.set_ylabels('Model')
    g.axes[0, 0].set_xticks(ticks)
    g.axes[0, 0].set_yticks(ticks)
    g.axes[0, 0].set_xlim(0, lim)
    g.axes[0, 0].set_ylim(0, lim)
    return g
