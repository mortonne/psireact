"""Linear ballistic accumulator model."""

import math
import numpy as np
import scipy.stats as st
import theano
import theano.tensor as tt
import pymc3 as pm
from . import model


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


def resp_pdf(t, i, A, b, v, s):
    """Probability density function for response i at time t."""
    all_neg, updates = theano.reduce(
        fn=lambda v_i, tot, s: normcdf(-v_i / s) * tot,
        outputs_info=tt.ones(1, dtype='float64'), sequences=v, non_sequences=s)

    # 1 - cdf for all other accumulators
    v_ind = tt.arange(v.shape[0])
    v_other = v[tt.nonzero(tt.neq(v_ind, i))]
    ncdf_all, updates = theano.reduce(
        fn=lambda v_i, tot, t, A, b, s: 1 - tcdf(t, A, b, v_i, s) * tot,
        outputs_info=tt.ones_like(t), sequences=v_other,
        non_sequences=[t, A, b, s])

    # pdf for this and no finish yet for others
    pdf = (tpdf(t, A, b, v[i], s) * ncdf_all) / (1 - all_neg)
    pdf_cond = tt.switch(tt.gt(t, 0), pdf, 0)
    return pdf_cond


class LBA(model.ReactModel):
    """Linear Ballistic Accumulator model."""

    def tensor_pdf(self, rt, response, test, param):
        tau = param['tau']
        sub_param = param.copy()
        del sub_param['tau']
        return resp_pdf(rt - tau, response, **sub_param)

    def function_pdf(self):
        t = tt.dvector('t')
        i = tt.iscalar('i')
        A = tt.dscalar('A')
        b = tt.dscalar('b')
        v = tt.dvector('v')
        s = tt.dscalar('s')
        pdf = resp_pdf(t, i, A, b, v, s)
        f = theano.function([t, i, A, b, v, s], pdf)
        return f

    def rvs_test(self, test, param, size):
        rt, resp = sample_response(**param, size=size)
        return rt, resp
