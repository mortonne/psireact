"""Linear ballistic accumulator model."""

import math
import numpy as np
import scipy.stats as st
import pymc3 as pm


def sample_finish_time(A, b, v, s, tau, size):
    """Sample finish time for a set of accumulators."""
    # select starting point
    k = st.uniform.rvs(loc=0, scale=A, size=size)

    t = np.zeros((len(v), size))
    for i, vi in enumerate(v):
        # sample drift rate, calculate time to threshold
        d = st.norm.rvs(loc=vi, scale=s, size=size)
        ti = tau + ((b - k) / d)

        # time is invalid if drift rate is negative
        ti[d < 0] = np.nan
        t[i, :] = ti
    return t


def sample_response(A, b, v, s, tau, size):
    """Sample response from a set of accumulators."""

    # get finish time for each accumulator
    t = sample_finish_time(A, b, v, s, tau, size)

    # determine winner on each valid trial
    valid = np.any(np.logical_not(np.isnan(t)), 0)
    t_valid = t[:, valid]
    t_winner = np.nanmin(t_valid, 0)
    i_winner = np.nanargmin(t_valid, 0)

    # initialize full matrix
    response = np.empty(size)
    response.fill(np.nan)
    rt = np.empty(size)
    rt.fill(np.nan)

    # fill in valid trials
    rt[valid] = t_winner
    response[valid] = i_winner
    return rt, response


def normpdf(x):
    return (1 / pm.math.sqrt(2 * math.pi)) * pm.math.exp(-(x ** 2) / 2)


def normcdf(x):
    return (1 / 2) * (1 + pm.math.erf(x / pm.math.sqrt(2)))


def tpdf(t, A, b, v, sv):
    """Probability distribution function over time."""
    g = (b - A - t * v) / (t * sv)
    h = (b - t * v) / (t * sv)
    f = (-v * normcdf(g) + sv * normpdf(g) +
         v * normcdf(h) - sv * normpdf(h)) / A
    return f


def tcdf(t, A, b, v, s):
    """Cumulative distribution function over time."""
    e1 = ((b - A - t * v) / A) * normcdf((b - A - t * v) / (t * s))
    e2 = ((b - t * v) / A) * normcdf((b - t * v) / (t * s))
    e3 = ((t * s) / A) * normpdf((b - A - t * v) / (t * s))
    e4 = ((t * s) / A) * normpdf((b - t * v) / (t * s))
    F = 1 + e1 - e2 + e3 - e4
    return F
