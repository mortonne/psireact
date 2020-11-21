"""Model of behavior in mistr data."""

import abc
import numpy as np
import scipy.stats as st
import scipy.optimize as optim
import pandas as pd
import pymc3 as pm
import theano.tensor as tt


def log_prob(p):
    """Calculate log probability."""
    # ensure that the probability is not zero before taking log
    eps = 10e-10
    logp = pm.math.log(pm.math.clip(p, eps, np.Inf))
    return logp


def param_search(f_fit, data, bounds, nrep=1, verbose=False):
    """
    Run a parameter search, with optional replication.

    Parameters
    ----------
    f_fit : callable
        Function that evaluates likelihood, given an input vector of
        parameters values.

    data : tuple
        Additional input needed to evaluate the parameters.

    bounds : sequence or scipy.optimize.Bounds
        Bounds for the parameters.

    nrep : int, optional
        Number of times to repeat searches.

    verbose : bool, optional
        If true, print more information.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        Optimization results.
    """
    f_optim = optim.differential_evolution
    if nrep > 1:
        val = np.zeros(nrep)
        rep = []
        for i in range(nrep):
            if verbose:
                print(f'Starting search {i + 1}/{nrep}...')
            res = f_optim(f_fit, bounds, data, disp=False, tol=.1,
                          init='random')
            rep.append(res)
            val[i] = res['fun']
            if verbose:
                print(f"Final f(x)= {res['fun']:.2f}")
        res = rep[np.argmin(val)]
    else:
        res = f_optim(f_fit, bounds, data, disp=verbose)
    return res


def param_bounds(var_bounds, var_names):
    """
    Set boundaries of group-level parameters.

    Parameters
    ----------
    var_bounds : dict of (str: tuple of float)
        Lower and upper bounds of each parameter.

    var_names : list of str
        Names of parameters to include.

    Returns
    -------
    bounds : scipy.optimize.Bounds
        Bounds object for use with scipy optimization functions.
    """
    group_lb = [var_bounds[k][0] for k in [*var_names]]
    group_ub = [var_bounds[k][1] for k in [*var_names]]
    bounds = optim.Bounds(group_lb, group_ub)
    return bounds


def subj_bounds(var_bounds, group_vars, subj_vars, n_subj):
    """
    Set boundaries of group and subject parameters.

    Parameters
    ----------
    var_bounds : dict of (str: tuple of float)
        Lower and upper bounds of each parameter.

    group_vars : list of str
        Names of parameters included as group parameters.

    subj_vars : list of str
        Names of parameters included as subject parameters.

    n_subj : int
        Number of subjects.

    Returns
    -------
    bounds : scipy.optimize.Bounds
        Bounds object for use with scipy optimization functions.
    """
    group_lb = [var_bounds[k][0] for k in [*group_vars]]
    group_ub = [var_bounds[k][1] for k in [*group_vars]]

    subj_lb = np.hstack(
        [np.tile(var_bounds[k][0], n_subj) for k in [*subj_vars]]
    )
    subj_ub = np.hstack(
        [np.tile(var_bounds[k][1], n_subj) for k in [*subj_vars]]
    )
    bounds = optim.Bounds(
        np.hstack((group_lb, subj_lb)), np.hstack((group_ub, subj_ub))
    )
    return bounds


def unpack_subj(fixed, x, group_vars, subj_vars):
    """
    Unpack subject-varying parameters.

    Parameters
    ----------
    fixed : dict of (str: float)
        Parameters with fixed values.

    x : numpy.ndarray
        Parameter values.

    group_vars : list of str
        Names of group-level parameters.

    subj_vars : list of str
        Names of subject-level parameters.

    Returns
    -------
    param : dict of (str: float)
        Group-level parameters.

    subj_param : list of (dict of (str: float))
        Parameters for each subject.
    """
    # unpack group parameters
    param = fixed.copy()
    param.update(dict(zip(group_vars, x)))

    # split up subject-varying parameters
    n_group = len(group_vars)
    xs = x[n_group:]
    if len(xs) % len(subj_vars) != 0:
        raise ValueError('Parameter vector has incorrect length.')
    n_subj = int(len(xs) / len(subj_vars))
    split = [
        xs[(i * n_subj):(i * n_subj + n_subj)] for i in range(len(subj_vars))
    ]

    # construct subject-specific parameters
    subj_param = [dict(zip(subj_vars, pars)) for pars in zip(*split)]
    return param, subj_param


def trace_df(trace):
    """
    Create a data frame from a trace object.

    Parameters
    ----------
    trace : pymc3.backends.base.MultiTrace
        Results from sampling using pymc3.

    Returns
    -------
    df : pandas.DataFrame
        Trace samples, with one column for each variable.
    """
    # exclude transformed variables
    var_names = [n for n in trace.varnames if not n.endswith('__')]
    d_var = {var: trace.get_values(var) for var in var_names}
    df = pd.DataFrame(d_var)
    return df


def sample_hier_drift(sd, alpha, beta, size=1):
    """
    Sample a hierarchical drift parameter.

    Parameters
    ----------
    sd : float
        Standard deviation of the group mu distribution.

    alpha : float
        Alpha parameter of the group standard deviation distribution.

    beta : float
        Beta parameter of the group standard deviation distribution.

    size : int, optional
        Number of values to sample.

    Returns
    -------
    x : numpy.ndarray or scalar
        Randomly sampled drift parameter for each simulated subject.
    """
    group_mu = st.halfnorm.rvs(sd)
    group_sd = st.gamma.rvs(alpha, 1/beta)
    x = st.norm.rvs(group_mu, group_sd, size)
    return x


def sample_params(fixed, param, subj_param, n_subj):
    """
    Create a random sample of parameters.

    Parameters
    ----------
    fixed : dict of (str: float)
        Values of fixed parameters.

    param : dict of (str: callable)
        Callables to sample group-level parameter values.

    subj_param : dict of (str: callable)
        Callables to sample subjct-level parameter values.

    n_subj : int
        Number of subjects to randomly sample.

    Returns
    -------
    gen_param : dict of (str: float)
        Sampled group-level parameters.

    gen_param_subj : list of (dict of (str: float))
        Sampled subject-level parameters.
    """
    d_group = {name: f() for name, f in param.items()}
    d_subj = {name: f() for name, f in subj_param.items()}
    gen_param_subj = [
        {name: val[i] for name, val in d_subj.items()} for i in range(n_subj)
    ]
    gen_param = fixed.copy()
    gen_param.update(d_group)
    return gen_param, gen_param_subj


def post_param(trace, fixed, group_vars, subj_vars=None):
    """
    Create parameter set from mean of the posterior distribution.

    Parameters
    ----------
    trace : pymc3.backends.base.MultiTrace
        Results from sampling using pymc3.

    fixed : dict of (str: float)
        Values of fixed parameters.

    group_vars : list of str
        Names of group-level parameters.

    subj_vars : list of str, optional
        Names of subject-level parameters.

    Returns
    -------
    param : dict of (str: float)
        Group-level parameters.

    subj_param : list of (dict of (str: float))
        Parameters for each subject.
    """
    param = fixed.copy()
    for name in group_vars:
        param[name] = np.mean(trace.get_values(name), 0)

    if subj_vars is not None:
        m_subj = {name: np.mean(trace.get_values(name), 0)
                  for name in subj_vars}
        n_subj = len(m_subj[subj_vars[0]])
        subj_param = [
            dict(zip(subj_vars, [m_subj[name][ind] for name in subj_vars]))
            for ind in range(n_subj)
        ]
    else:
        subj_param = {}
    return param, subj_param


class ReactModel:
    """Base class for RT models."""
    def __init__(self):
        self.fixed = None
        self.group_vars = None
        self.subj_vars = None

    @abc.abstractmethod
    def tensor_pdf(self, rt, response, test, param):
        """
        Probability density function for a set of parameters.

        Parameters
        ----------
        rt : numpy.ndarray
            Response times on each trial.

        response : numpy.ndarray
            Response options selected on each trial.

        test : numpy.ndarray
            Type of each test trial.

        param : dict of (str: float)
            Parameter values.

        Returns
        -------
        numpy.ndarray
            Probability density function value for each trial.
        """
        pass

    @abc.abstractmethod
    def function_pdf(self):
        """
        Compiled Theano probability density function.

        Returns
        -------
        theano.compile.function_module.Function
            Compiled function that takes data and parameters and
            returns the PDF.
        """
        pass

    @abc.abstractmethod
    def rvs_test(self, test_type, param, size):
        """
        Generate responses for a given test type.

        Parameters
        ----------
        test_type : int
            Test trial type.

        param : dict of (str: float)
            Parameter values.

        size : int
            Number of trials to simulate.

        Returns
        -------
        rt : numpy.ndarray
            Simulated response times.

        response : numpy.ndarray
            Simulated responses.
        """
        pass

    def rvs(self, test, param):
        """
        Generate responses for all test types.

        Parameters
        ----------
        test : numpy.ndarray
            Test trial type.

        param : dict of (str: float)
            Parameter values.

        Returns
        -------
        rt : numpy.ndarray
            Simulated response times.

        response : numpy.ndarray
            Simulated responses.
        """
        n_trial = len(test)
        response = np.zeros(n_trial)
        rt = np.zeros(n_trial)
        test_types = np.unique(test)
        for this_test in test_types:
            ind = test == this_test
            test_rt, test_response = self.rvs_test(this_test, param,
                                                   size=np.count_nonzero(ind))
            response[ind] = test_response
            rt[ind] = test_rt
        return rt, response

    def rvs_subj(self, test, subj_idx, param, subj_param):
        """
        Generate responses based on subject-varying parameters.

        Parameters
        ----------
        test : numpy.ndarray
            Test trial type.

        subj_idx : numpy.ndarray
            Index of the subject to simulate for each trial.

        param : dict of (str: float)
            Parameter values.

        subj_param : list of (dict of (str: float))
            Parameter values for each subject.

        Returns
        -------
        rt : numpy.ndarray
            Simulated response times.

        response : numpy.ndarray
            Simulated responses.
        """
        unique_idx = np.unique(subj_idx)
        rt = np.zeros(test.shape)
        response = np.zeros(test.shape)
        for idx in unique_idx:
            ind = subj_idx == idx
            param.update(subj_param[idx])
            rt[ind], response[ind] = self.rvs(test[ind], param)
        return rt, response

    def gen(self, test, param, subj_idx=None, nrep=1, subj_param=None):
        """
        Generate a simulated dataset.

        Parameters
        ----------
        test : numpy.ndarray
            Test type of each trial.

        param : dict of (str: float)
            Parameter values.

        subj_idx : numpy.ndarray, optional
            Index of the subject to simulate for each trial.

        nrep : int, optional
            Number of replications to simulate.

        subj_param : list of (dict of (str: float)), optional
            Parameter values for each subject.

        Returns
        -------
        data : pandas.DataFrame
            Simulated data.
        """
        data_list = []
        for i in range(nrep):
            if subj_param is not None:
                rt, response = self.rvs_subj(test, subj_idx, param, subj_param)
            else:
                rt, response = self.rvs(test, param)
            rep = pd.DataFrame({'test': test, 'rt': rt,
                                'response': response.astype('int32')})
            if subj_idx is not None:
                rep.loc[:, 'subj_idx'] = subj_idx
            rep.loc[:, 'rep'] = i
            data_list.append(rep)
        data = pd.concat(data_list, ignore_index=True)
        return data

    def tensor_logp(self, param):
        """
        Function to evaluate the log PDF for a given response.

        Parameters
        ----------
        param : dict of (str: float)
            Parameter values.

        Returns
        -------
        logp : callable
            Function that takes rt, response, and test and returns log
            probability.
        """
        def logp(rt, response, test):
            p = self.tensor_pdf(rt, response, test, param)
            return log_prob(p)
        return logp

    def tensor_logp_subj(self, param, subj_vars):
        """
        Function to evaluate the log PDF with subject-varying parameters.

        Parameters
        ----------
        param : dict of (str: float)
            Parameter values.

        subj_vars : list of str
            Names of parameters that vary by subject.

        Returns
        -------
        logp : callable
            Function that takes rt, response, test, and subj_param and returns
            log probability.
        """
        def logp(rt, response, test, subj_idx):
            i = tt.cast(subj_idx, 'int64')
            subj_param = param.copy()
            for var in subj_vars:
                subj_param[var] = param[var][i]
            p = self.tensor_pdf(rt, response, test, subj_param)
            return log_prob(p)
        return logp

    def total_logl(self, rt, response, test, param, f_l=None):
        """
        Calculate log likelihood.

        Parameters
        ----------
        rt : numpy.ndarray
            Response times on each trial.

        response : numpy.ndarray
            Response options selected on each trial.

        test : numpy.ndarray
            Type of each test trial.

        param : dict of (str: float)
            Parameter values.

        f_l : callable, optional
            Function that takes rt, response, test, and param and
            returns likelihood for each trial.

        Returns
        -------
        logl : float
            Log likelihood of all trials.
        """
        if f_l is None:
            f_l = self.function_pdf()
        eps = 0.000001
        # evaluate log likelihood
        lik = f_l(rt, response, test, **param)
        lik[lik < eps] = eps
        logl = np.sum(np.log(lik))
        if np.isnan(logl) or np.isinf(logl):
            return -10e10
        return logl

    def total_logl_subj(self, rt, response, test, subj_idx, param, indiv_param,
                        f_l=None):
        """
        Calculate log likelihood using subject-varying parameters.

        Parameters
        ----------
        rt : numpy.ndarray
            Response times on each trial.

        response : numpy.ndarray
            Response options selected on each trial.

        test : numpy.ndarray
            Type of each test trial.

        subj_idx : numpy.ndarray
            Index of the subject to simulate for each trial.

        param : dict of (str: float)
            Parameter values.

        indiv_param : list of (dict of (str: float))
            Parameter values for each subject.

        f_l : callable, optional
            Function that takes rt, response, test, and param and
            returns likelihood for each trial.

        Returns
        -------
        logl : float
            Log likelihood of all trials.
        """
        if f_l is None:
            f_l = self.function_pdf()

        logl = 0
        for idx, subj_param in enumerate(indiv_param):
            subj_rt = rt[subj_idx == idx]
            subj_response = response[subj_idx == idx]
            subj_test = test[subj_idx == idx]

            param.update(subj_param)
            subj_logl = self.total_logl(subj_rt, subj_response, subj_test,
                                        param, f_l)
            logl += subj_logl
        return logl

    def function_logl(self, fixed, var_names):
        """
        Generate log likelihood function for use with fitting.

        Parameters
        ----------
        fixed : dict of (str: float)
            Values of fixed parameters.

        var_names : list of str
            Names of variable parameters.

        Returns
        -------
        fit_logl : callable
            Function that takes parameters, rt, response, and test and
            returns log likelihood.
        """
        param = fixed.copy()
        f_l = self.function_pdf()

        def fit_logl(x, rt, response, test):
            # unpack parameters
            param.update(dict(zip(var_names, x)))
            logl = self.total_logl(rt, response, test, param, f_l)
            return -logl

        return fit_logl

    def function_logl_subj(self, fixed, group_vars, subj_vars):
        """
        Generate log likelihood function for subject fitting.

        Parameters
        ----------
        fixed : dict of (str: float)
            Values of fixed parameters.

        group_vars : list of str
            Names of variable group parameters.

        subj_vars : list of str
            Names of variable subject parameters.

        Returns
        -------
        fit_logl_subj : callable
            Function that takes parameters, rt, response, test, and
            subject index, and returns log likelihood.
        """
        f_l = self.function_pdf()

        def fit_logl_subj(x, rt, response, test, subj_idx):
            # unpack parameters
            param, subj_param = unpack_subj(fixed, x, group_vars, subj_vars)

            # evaluate all subjects
            logl = self.total_logl_subj(rt, response, test, subj_idx,
                                        param, subj_param, f_l)
            return -logl

        return fit_logl_subj

    def fit(self, rt, response, test, fixed, var_names, var_bounds,
            nrep=1, verbose=False):
        """
        Estimate maximum likelihood parameters.

        rt : numpy.ndarray
            Response times on each trial.

        response : numpy.ndarray
            Response options selected on each trial.

        test : numpy.ndarray
            Type of each test trial.

        fixed : dict of (str: float)
            Values of fixed parameters.

        var_names : list of str
            Names of variable parameters.

        var_bounds : dict of (str: tuple of float)
            Lower and upper bounds of each parameter.

        nrep : int, optional
            Number of times to repeat searches.

        verbose : bool, optional
            If true, print more information.

        Returns
        -------
        param : dict of (str: float)
            Maximum likelihood parameter values.

        stats : dict
            Log likelihood, number of parameters, number of datapoints
            fit, and Bayesian information criterion.
        """
        # maximum likelihood estimation
        fit_logl = self.function_logl(fixed, var_names)
        bounds = param_bounds(var_bounds, var_names)
        data = (rt, response, test)
        res = param_search(fit_logl, data, bounds, nrep=nrep, verbose=verbose)

        # fitted parameters
        param = fixed.copy()
        param.update(dict(zip(var_names, res['x'])))

        # statistics
        logl = -res['fun']
        k = len(var_names)
        n = len(rt)
        bic = np.log(n) * k - 2 * logl
        stats = {'logl': logl, 'k': k, 'n': n, 'bic': bic}

        return param, stats

    def fit_subj(self, rt, response, test, subj_idx,
                 fixed, group_vars, subj_vars, var_bounds,
                 nrep=1, verbose=False):
        """
        Estimate maximum likelihood parameters for each subject.

        Parameters
        ----------
        rt : numpy.ndarray
            Response times on each trial.

        response : numpy.ndarray
            Response options selected on each trial.

        test : numpy.ndarray
            Type of each test trial.

        subj_idx : numpy.ndarray
            Index of the subject to simulate for each trial.

        fixed : dict of (str: float)
            Values of fixed parameters.

        group_vars : list of str
            Names of group-level variable parameters.

        subj_vars : list of str
            Names of subject-level variable parameters.

        var_bounds : dict of (str: tuple of float)
            Lower and upper bounds of each parameter.

        nrep : int, optional
            Number of times to repeat searches.

        verbose : bool, optional
            If true, print more information.

        Returns
        -------
        param : list of (dict of (str: float))
            Maximum likelihood parameter values for each subject.

        stats : dict
            Log likelihood, number of parameters, number of datapoints
            fit, and Bayesian information criterion.
        """
        # maximum likelihood estimation
        fit_logl = self.function_logl_subj(fixed, group_vars, subj_vars)

        # pack parameter bound information
        n_subj = len(np.unique(subj_idx))
        bounds = subj_bounds(var_bounds, group_vars, subj_vars, n_subj)

        data = (rt, response, test, subj_idx)
        res = param_search(fit_logl, data, bounds, nrep=nrep, verbose=verbose)

        # fitted parameters
        param = unpack_subj(fixed, res['x'], group_vars, subj_vars)

        # statistics
        logl = -res['fun']
        k = len(group_vars) + len(subj_vars) * n_subj
        n = len(rt)
        bic = np.log(n) * k - 2 * logl
        stats = {'logl': logl, 'k': k, 'n': n, 'bic': bic}

        return param, stats
