"""Linear ballistic accumulator model."""

import math
import numpy as np
import scipy.stats as st
import theano
import theano.tensor as tt
import pymc3 as pm
from psireact import model


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


def ncdf(t, A, b, v, s):
    """Probability of no response from a set of accumulators."""
    ncdf_all, updates = theano.reduce(
        fn=lambda v_i, tot, t, A, b, s: (1 - tcdf(t, A, b, v_i, s)) * tot,
        sequences=v, outputs_info=tt.ones_like(t),
        non_sequences=[t, A, b, s])
    return ncdf_all


def resp_pdf(t, ind, A, b, v, s):
    """Probability density function for response i at time t."""
    p_neg, updates = theano.reduce(
        fn=lambda v_i, tot, s: normcdf(-v_i / s) * tot,
        sequences=v, outputs_info=tt.ones(1, dtype='float64'), non_sequences=s)

    # PDF for i and no finish yet for others
    v_ind = tt.arange(v.shape[0])
    i = tt.cast(ind, 'int64')
    res, updates = theano.scan(
        fn=(lambda t_j, i_j, v_ind, A, b, v, s:
            (tpdf(t_j, A, b, v[i_j], s) *
             ncdf(t_j, A, b, v[tt.nonzero(tt.neq(v_ind, i_j))], s))),
        sequences=[t, i], non_sequences=[v_ind, A, b, v, s])

    # conditionalize on any response
    pdf = res / (1 - p_neg)

    # define probability of negative times to zero
    pdf_cond = tt.switch(tt.gt(t, 0), pdf, 0)
    return pdf_cond


def trial_resp_pdf(t, ind, A, b, v, s):
    """Probability density function for response i at time t."""
    p_neg, updates = theano.reduce(
        fn=lambda v_i, tot, s: normcdf(-v_i / s) * tot,
        sequences=v, outputs_info=tt.ones(1, dtype='float64'), non_sequences=s)

    # PDF for i and no finish yet for others
    v_ind = tt.arange(v.shape[0])
    i = tt.cast(ind, 'int64')
    pdf = (tpdf(t, A, b, v[i], s) *
           ncdf(t, A, b, v[tt.nonzero(tt.neq(v_ind, i))], s)) / (1 - p_neg)

    # define probability of negative times to zero
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
        # time and response vary by trial
        t = tt.dvector('t')
        i = tt.ivector('i')

        # parameters are fixed over trial
        A = tt.dscalar('A')
        b = tt.dscalar('b')
        v = tt.dvector('v')
        s = tt.dscalar('s')
        tau = tt.dscalar('tau')
        pdf = resp_pdf(t - tau, i, A, b, v, s)
        f = theano.function([t, i, A, b, v, s, tau], pdf)
        return f

    def rvs_test(self, test, param, size):
        rt, resp = sample_response(**param, size=size)
        return rt, resp
