"""
Tools n. 2: loss_and_minimizer.
It includes the definition of the loss functions and its minimization.
It includes also the splitting into training and test set `select_traintest` and the tool `validation`.
"""

import copy
import time
import numpy.random as random
from scipy.optimize import minimize

# numpy is required for loadtxt and for gradient arrays with L-BFGS-B minimization (rather than jax.numpy)
import numpy
import jax
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)

# %% B. Functions to compute Gamma (they act on numpy arrays):
# %% B1. compute_js

def compute_js(n_experiments):
    """
    This tool computes the indices `js` (defined by cumulative sums) for lambdas corresponding to different molecular systems and
    types of observables. Be careful to follow always the same order: let's choose it as that of `data.n_experiments`,
    which is a dictionary `n_experiments[name_sys][name]`.
    """

    js = []

    for i_sys, name_sys in enumerate(n_experiments.keys()):
        js.append([])
        for name in n_experiments[name_sys].keys():
            js[i_sys].append(n_experiments[name_sys][name])
        js[i_sys] = [0] + np.cumsum(np.array(js[i_sys])).tolist()

    js[0] = np.array(js[0])

    if len(n_experiments.keys()) > 1:
        for i_sys in range(1, len(n_experiments.keys())):
            js[i_sys] = np.array(js[i_sys]) + js[i_sys-1][-1]

    return js

# %% B2. compute_new_weights
# to reweight given original weights and correction

def compute_new_weights(weights: numpy.ndarray, correction: numpy.ndarray):
    """
    This tool computes the new weights as weights*exp(-correction).
    It modifies Parameters `weights` are normalized and `correction` is shifted by `correction -= shift`, where `shift = np.min(correction)`.
    It returns two variables: a Numpy array `new_weights` and a float `logZ`.
    """

    weights = weights/np.sum(weights)

    """ shift is such that the physical Z is = Z/np.exp(shift) """
    shift = np.min(correction)
    correction -= shift

    new_weights = np.exp(-correction)*weights

    assert not np.isnan(new_weights).any(), 'Error: new_weights contains None'

    logZ = np.log(np.sum(new_weights))-shift
    new_weights = new_weights/np.sum(new_weights)

    return new_weights, logZ

# %% B3. gamma_function

def gamma_function(lambdas: numpy.ndarray, g: numpy.ndarray, gexp: numpy.ndarray, weights: numpy.ndarray, alpha: float, if_gradient: bool = False):
    """
    This tool computes gamma function and (if `if_gradient`) its derivatives and the average values of the observables `av_g`.
    Make sure that `lambdas` follow the same order as `g`, `gexp` (let's use that of `data.n_experiments`).

    Parameters
    ----------
    
    lambdas : array_like
        Numpy 1-dimensional array of length N, where `lambdas[j]` is the lambda value for the j-th observable.
    
    g : array_like
        Numpy 2-dimensional array (M x N); `g[i,j]` is the j-th observable computed in the i-th frame.
    
    gexp : array_like
        Numpy 2-dimensional array (N x 2); `gexp[j,0]` is the experimental value of the j-th observable, `gexp[j,1]` is the associated experimental uncertainty.
    
    weights : array_like
        Numpy 1-dimensional array of length M; `w[i]` is the weight of the i-th frame (possibly non-normalized).
    
    alpha : float
        The value of the alpha hyperparameter.
    
    if_gradient : bool
        If true, return also the gradient of the gamma function.
    """
    correction_lambdas = np.matmul(g, lambdas)
    newweights, logZlambda = compute_new_weights(weights, correction_lambdas)

    gammaf = np.matmul(lambdas, gexp[:, 0]) + 1/2*alpha*np.matmul(lambdas**2, gexp[:, 1]**2) + logZlambda

    if if_gradient:
        av_g = np.einsum('i,ij', newweights, g)
        grad = -(av_g-gexp[:, 0]-alpha*lambdas*gexp[:, 1]**2)
        grad = numpy.array(grad)
        return gammaf, grad, av_g
    else:
        return gammaf

# %% B4. normalize_observables

def normalize_observables(gexp, g, weights=None):
    """
    This tool normalizes `g` and `gexp`. Since experimental observables have different units, it is better to normalize them, in order that
    varying any lambda coefficient by the same value epsilon would result in comparable effects to the ensemble.
    This results to be useful in the minimization of `gamma_function`.

    Parameters
    ----------
    gexp, g : dicts
        Dictionaries corresponding to `data.sys[name_sys].gexp` and `data.sys[name_sys].g`.
        
    weights : array-like
        Numpy 1-dimensional array, by default `None` (namely, equal weight for each frame).
    --------

    Returns:
    --------
    norm_g, norm_gexp : dict
        Dictionaries for normalized g and gexp.
    
    norm_gmean, norm_gstd : dict
        Dictionaries for the reference values for normalization (average and standard deviation).
    """
    norm_g = {}
    norm_gexp = {}
    norm_gmean = {}
    norm_gstd = {}

    for name in g.keys():
        if weights is None:
            norm_gmean[name] = np.mean(g[name], axis=0)
            norm_gstd[name] = np.std(g[name], axis=0)
        else:
            norm_gmean[name] = np.average(g[name], axis=0, weights=weights)
            norm_gstd[name] = np.sqrt(np.average(g[name]**2, axis=0, weights=weights))-norm_gmean[name]**2

        norm_gexp[name] = np.vstack([(gexp[name][:, 0]-norm_gmean[name])/norm_gstd[name], gexp[name][:, 1]/norm_gstd[name]]).T
        norm_g[name] = (g[name]-norm_gmean[name])/norm_gstd[name]

    return norm_g, norm_gexp, norm_gmean, norm_gstd

# %% C. Functions to compute and minimize lossf_nested:
# %% C1. compute_ff_correction


# """
# This functions **compute_ff_correction** computes the force-field correction.
# BE CAREFUL to correctly match pars with f inside user-defined **ff_correction**.
# """


# def compute_ff_correction(ff_correction, f, pars):

#     if ff_correction == 'linear':
#         correction_ff = np.matmul(f, pars)
#     else:
#         correction_ff = ff_correction(pars, f)

#     return correction_ff

# %% C2. compute_D_KL


def compute_D_KL(weights_P: numpy.ndarray, correction_ff: numpy.ndarray, temperature: float, logZ_P: float):
    """
    This tool computes the Kullback-Leibler divergence of P(x) = 1/Z P_0 (x) e^(-V(x)/T)
    with respect to P_0 as `av(V)/T + log Z` where av(V) is the average value of the potential V(x) over P(x).
    
    Parameters
    ----------
    weights_P : array_like
        Numpy 1-dimensional array for the normalized weights P(x).

    correction_ff : array_like
        Numpy 1-dimensional array for the reweighting potential V(x).
    
    temperature: float
        The value of temperature T, in measure units consistently with V(x), namely, such that V(x)/T is adimensional.
    
    logZ_P: float
        The value of log Z.
    """
    weighted_ff = weights_P*np.array(correction_ff)
    av_ff = np.nansum(weighted_ff, axis=0)
    D_KL = -(av_ff/temperature + logZ_P)

    return D_KL

# %% C3. l2_regularization


def l2_regularization(pars: numpy.ndarray, choice: str = 'plain l2'):
    """
    This tool computes the L2 regularization for the force-field correction coefficients `pars` as specified by `choice`. It includes:
    
    - `'plain l2'` (plain L2 regularization of `pars`);
    
    - L2 regularization for alchemical calculations with charges (as described by Valerio Piomponi et al., see main paper):
    `pars[:-1]` are the charges and `pars[-1]` is V_eta; there is the constraint on the total charge, and there are 3 `pars[4]` charges in the molecule;
     so, `'constraint 1'` is the L2 regularization on charges, while `'constraint 2'` is the L2 regularization on charges and on V_eta.
    

    Output values: lossf_reg and gradient (floats).
    """
    lossf_reg = None
    gradient = None

    if choice == 'plain l2':
        lossf_reg = np.sum(pars**2)
        gradient = 2*pars

    elif choice == 'constraint 1':
        lossf_reg = np.sum(pars[:-1]**2)+(np.sum(pars[:-1])+2*pars[4])**2
        n = np.array([1, 1, 1, 1, 3])
        gradient = 2*(pars[:-1]+(np.sum(pars[:-1])+2*pars[4])*n)

    elif choice == 'constraint 2':
        lossf_reg = np.sum(pars**2)+(np.sum(pars[:-1])+2*pars[4])**2
        n = np.array([1, 1, 1, 1, 3, 0])
        gradient = 2*(pars+(np.sum(pars[:-1])+2*pars[4])*n)

    return lossf_reg, gradient

# %% C4. compute_chi2

def compute_chi2(ref, weights, g, gexp, if_separate=False):
    """
    This tool computes the chi2 (for a given molecular system).
    
    Parameters
    ----------
    ref : dict
        Dictionary for references (`=`, `>`, `<`, `><`) used to compute the appropriate chi2.
    
    weights : array_like
        Numpy 1-dimensional array of weights.
    
    g : dict
        Dictionary of observables specific for the given molecular system.

    gexp : dict
        Dictionary of experimental values specific for the given molecular system (coherently with `g`).

    if_separate: bool
        Boolean variable, True if you are distinguishing between LOWER and UPPER bounds (`name_type + ' LOWER'` or
        `name_type + ' UPPER'`), needed for minimizations with double bounds.
    --------

    Returns:
    --------
    This tool returns 4 variables: 3 dictionaries (with keys running over different kinds of observables) and 1 float:

    av_g : dict
        Dictionary of average values of the observables `g`.

    chi2 : dict
        Dictionary of chi2.
    
    rel_diffs: dict
        Dicionary of relative differences.
    
    tot_chi2: float
        Total chi2 for the given molecular system.
    """
    av_g = {}
    rel_diffs = {}
    chi2 = {}
    tot_chi2 = 0

    for name_type in gexp.keys():

        if ref[name_type] == '><':
            # av_g UPPER is equal to av_g LOWER but (if if_separate) you have to distinguish them
            if if_separate:
                av_g[name_type+' LOWER'] = np.einsum('i,ij', weights, g[name_type+' LOWER'])
                av_g[name_type+' UPPER'] = av_g[name_type+' LOWER']

                rel_diffs[name_type+' UPPER'] = np.maximum(
                    av_g[name_type+' UPPER']-gexp[name_type+' UPPER'][:, 0],
                    np.zeros(len(av_g[name_type+' UPPER'])))/gexp[name_type+' UPPER'][:, 1]
                rel_diffs[name_type+' LOWER'] = np.minimum(
                    av_g[name_type+' LOWER']-gexp[name_type+' LOWER'][:, 0],
                    np.zeros(len(av_g[name_type+' LOWER'])))/gexp[name_type+' LOWER'][:, 1]

            else:
                av_g[name_type] = np.einsum('i,ij', weights, g[name_type])

                rel_diffs[name_type+' UPPER'] = np.maximum(
                    av_g[name_type]-gexp[name_type+' UPPER'][:, 0],
                    np.zeros(len(av_g[name_type])))/gexp[name_type+' UPPER'][:, 1]
                rel_diffs[name_type+' LOWER'] = np.minimum(
                    av_g[name_type]-gexp[name_type+' LOWER'][:, 0],
                    np.zeros(len(av_g[name_type])))/gexp[name_type+' LOWER'][:, 1]

                # either one of the two is zero and the other non-zero
                rel_diffs[name_type] = rel_diffs[name_type+' LOWER']+rel_diffs[name_type+' UPPER']
                del rel_diffs[name_type+' LOWER'], rel_diffs[name_type+' UPPER']

        elif ref[name_type] == '=':
            av_g[name_type] = np.einsum('i,ij', weights, g[name_type])
            rel_diffs[name_type] = (av_g[name_type]-gexp[name_type][:, 0])/gexp[name_type][:, 1]

        elif ref[name_type] == '<':
            av_g[name_type] = np.einsum('i,ij', weights, g[name_type])
            rel_diffs[name_type] = np.maximum(
                av_g[name_type]-gexp[name_type][:, 0], np.zeros(len(av_g[name_type])))/gexp[name_type][:, 1]

        elif ref[name_type] == '>':
            av_g[name_type] = np.einsum('i,ij', weights, g[name_type])
            rel_diffs[name_type] = np.minimum(
                av_g[name_type]-gexp[name_type][:, 0], np.zeros(len(av_g[name_type])))/gexp[name_type][:, 1]

        else:
            print('error')

    for k in rel_diffs.keys():
        chi2[k] = np.sum(rel_diffs[k]**2)
        tot_chi2 += chi2[k]

    return av_g, chi2, rel_diffs, tot_chi2

# %% C5. compute_DeltaDeltaG_terms

def compute_DeltaDeltaG_terms(data, logZ_P):
    """
    This tool computes the chi2 for Delta Delta G (free-energy differences from thermodynamic cycles),
    contributing to the loss function with alchemical calculations.

    Parameters
    ----------
    data : class instance
        Object `data`; here, `data._global_` has the attribute `cycle_names` (list of names of the thermodynamic cycles);
        `for s in data._global_.cycle_names`: `data.cycle[s]` has attributes `temperature` (of the cycle) and `gexp_DDG`;
        `for s in my_list` (where `my_list` is the list of system names associated to a thermodynamic cycle
        `my_list = [x2 for x in list(data._global_.cycle_names.values()) for x2 in x]`):
        `data.sys[s]` has attributes `temperature` (of the system) and `logZ`.
        
    logZ_P : dict
        Dictionary for logarithm of the partition function Z_P, namely, average value of e^{-V_\phi(x)/temperature} over the original ensemble.
    --------
    Returns:
    --------
    new_av_DG : dict
        Dictionary of reweighted averages of Delta G.

    chi2 : dict
        Dictionary of chi2 (one for each thermodynamic cycle).
    
    loss : float
        Total contribution to the loss function from free-energy differences Delta Delta G,
        given by 1/2 of the total chi2.
    """
    cycle_names = data._global_.cycle_names

    new_av_DG = {}
    chi2 = {}
    loss = 0

    for cycle_name in cycle_names:
        my_list = data._global_.cycle_names[cycle_name]
        for s in my_list:
            if (s in logZ_P.keys()) and (not logZ_P[s] == 0):
                # correction only on MD
                new_av_DG[s] = -data.sys[s].temperature*(logZ_P[s] + data.sys[s].logZ)
            if s not in logZ_P:
                logZ_P[s] = 0

        DeltaDeltaG = -(
            logZ_P[my_list[3]] + data.sys[my_list[3]].logZ  # MD
            - logZ_P[my_list[1]] - data.sys[my_list[1]].logZ)  # AD

        DeltaDeltaG += logZ_P[my_list[2]] + data.sys[my_list[2]].logZ  # MS
        - logZ_P[my_list[0]] - data.sys[my_list[0]].logZ  # AS

        DeltaDeltaG = DeltaDeltaG*data.cycle[cycle_name].temperature

        chi2[cycle_name] = ((DeltaDeltaG - data.cycle[cycle_name].gexp_DDG[0])/data.cycle[cycle_name].gexp_DDG[1])**2
        loss += 1/2*chi2[cycle_name]

    return new_av_DG, chi2, loss

# %% C6. compute_details_ER


def compute_details_ER(weights_P, g, data, lambdas, alpha):
    """
    This is an internal tool of `loss_function` which computes explicitely the contribution to the loss function due to Ensemble Refinement
    (namely, 1/2 chi2 + alpha D_KL) and compare this value with -alpha*Gamma (they are equal in the minimum: check).
    It cycles over different systems. It acts after the minimization of the loss function inside `loss_function` (not for the minimization
    itself, since we exploit the Gamma function).

    Be careful to use either: normalized values for `lambdas` and `g` (if `hasattr(data.sys[name_sys],'normg_mean')`) or non-normalized ones
    (if `not hasattr(data.sys[name_sys],'normg_mean')`).
    
    Parameters
    ----------
    weights_P : dict
        Dictionary of Numpy arrays, namely, the weights on which Ensemble Refinement acts (those with force-field correction
        in the fully combined refinement).
        
    g : dict
        Dictionary of dictionaries, like for `data.sys[name_sys].g`, corresponding to the observables (computed with updated forward-model coefficients).
    
    data : dict
        The original data object.
    
    lambdas : dict
        Dictionary of Numpy arrays, corresponding to the coefficients for Ensemble Refinement.
    
    alpha : float
        The alpha hyperparameter, for Ensemble Refinement.
    """

    class Details_class:
        def __init__(self):
            self.loss_explicit = 0
            # loss_explicit is loss function computed explicitly as 1/2*chi2 + alpha*D_KL (rather than with Gamma function)
            self.weights_new = {}
            self.logZ_new = {}
            self.av_g = {}
            self.chi2 = {}
            self.D_KL_alpha = {}
            self.abs_difference = {}

    Details = Details_class()

    system_names = data._global_.system_names

    for name_sys in system_names:

        if hasattr(data.sys[name_sys], 'normg_mean'):
            print('WARNING: you are using normalized observables!')

        flatten_g = np.hstack([g[name_sys][k] for k in data.sys[name_sys].n_experiments.keys()])
        flatten_gexp = np.vstack([data.sys[name_sys].gexp[k] for k in data.sys[name_sys].n_experiments.keys()])
        correction = np.einsum('ij,j', flatten_g, lambdas[name_sys])

        out = compute_new_weights(weights_P[name_sys], correction)
        Details.weights_new[name_sys] = out[0]
        Details.logZ_new[name_sys] = out[1]

        out = compute_chi2(data.sys[name_sys].ref, Details.weights_new[name_sys], g[name_sys], data.sys[name_sys].gexp)
        Details.av_g[name_sys] = out[0]
        Details.chi2[name_sys] = out[1]
        loss1 = 1/2*out[3]

        Details.D_KL_alpha[name_sys] = compute_D_KL(Details.weights_new[name_sys], correction, 1, Details.logZ_new[name_sys])
        loss1 += alpha*Details.D_KL_alpha[name_sys]
        Details.loss_explicit += loss1

        """ You could also use lambdas to evaluate immediately chi2 and D_KL
        (if lambdas are determined from the given frames) """
        loss2 = -alpha*gamma_function(lambdas[name_sys], flatten_g, flatten_gexp, weights_P[name_sys], alpha)

        Details.abs_difference[name_sys] = np.abs(loss2-loss1)

    return Details

# %% C7. loss_function


def loss_function(
    pars_ff_fm: numpy.ndarray, data: dict, regularization: dict,
        alpha: float = +np.inf, beta: float = +np.inf, gamma: float = +np.inf,
        fixed_lambdas: numpy.ndarray = None, gtol_inn: float = 1e-3, if_save: bool = False, bounds: dict = None):
    """
    This tool computes the fully-combined loss function (to minimize), taking advantage of the inner minimization with Gamma function.
    
    If `not np.isinf(alpha)`:

    - if `fixed_lambdas == None`, then do the inner minimization of Gamma (in this case, you have the global variable `lambdas`,
        corresponding to the starting point of the minimization; it is a Numpy array sorted as in `compute_js`).

    - else: `lambdas` is fixed (`fixed_lambdas` is not `None`) and the Gamma function is evaluated at this value of lambda, which must
        correspond to its point of minimum, otherwise there is a mismatch between the Gamma function and the Ensemble Refinement loss.

    The order followed for `lambdas` is the one of `compute_js`, which is not modified in any step.

    If `if_save`: `loss_function` will return `Details` class instance with the detailed results; otherwise, it will return just the loss value.

    The input data will not be modified by `loss_function` (neither explicitely by `loss_function` nor by its inner functions):
    for forward-model updating, you are going to define a new variable `g` (through `copy.deepcopy`).

    Parameters
    ----------
    pars_ff_fm: array_like
        Numpy 1-dimensional array with parameters for force-field corrections and/or forward models.
        These parameters are sorted as: first force-field correction (ff), then forward model (fm);
        order for ff: `names_ff_pars = []`; `for k in system_names: [names_ff_pars.append(x) for x in data[k].f.keys() if x not in names_ff_pars]`;
        order for fm: the same as `data.forward_coeffs_0`.

    data: dict
        Dictionary of class instances as organised in `load_data`, which constitutes the `data` object.
    
    regularization: dict
        Dictionary for the force-field and forward-model correction regularizations (see `MDRefinement`).

    alpha, beta, gamma: floats
        The hyperparameters of the three refinements (respectively, to: the ensemble, the force-field, the forward-model);
        (`+np.inf` by default, namely no refinement in that direction).
    
    fixed_lambdas: array_like
        Numpy 1-dimensional array of fixed values of `lambdas` (coefficients for Ensemble Refinement, organized as in `compute_js`).  (`None` by default).
    
    gtol_inn: float
        Tolerance `gtol` for the inner minimization of Gamma function (`1e-3` by default).
    
    if_save: bool
        Boolean variable (`False` by default).
    
    bounds: dict
        Dictionary of boundaries for the inner minimization (`None` by default).
    """
    assert alpha > 0, 'alpha must be strictly positive'
    assert beta >= 0, 'beta must be positive or zero'
    assert gamma >= 0, 'gamma must be positive or zero'

    system_names = data._global_.system_names

    if_fixed_lambdas = None  # to avoid error in Pylint
    if not np.isinf(alpha):
        if (fixed_lambdas is None):
            if_fixed_lambdas = False
            global lambdas
            if 'lambdas' not in globals():
                lambdas = np.zeros(data._global_.tot_n_experiments(data))
        else:
            if_fixed_lambdas = True
            lambdas = fixed_lambdas

    if not np.isinf(beta):
        names_ff_pars = data._global_.names_ff_pars
        pars_ff = pars_ff_fm[:len(names_ff_pars)]

    pars_fm = None  # to avoid error in Pylint
    if not np.isinf(gamma):
        if np.isinf(beta):
            pars_fm = pars_ff_fm
        else:
            pars_fm = pars_ff_fm[len(names_ff_pars):]

    loss = 0

    weights_P = {}

    if not np.isinf(beta):
        correction_ff = {}
    logZ_P = {}

    g = {}

    for name_sys in system_names:

        """ 1a. compute force-field corrections and corresponding re-weights """

        if not np.isinf(beta):
            if hasattr(data.sys[name_sys], 'ff_correction'):
                correction_ff[name_sys] = data.sys[name_sys].ff_correction(pars_ff, data.sys[name_sys].f)
                weights_P[name_sys], logZ_P[name_sys] = compute_new_weights(
                    data.sys[name_sys].weights, correction_ff[name_sys]/data.sys[name_sys].temperature)

            else:  # if beta is not infinite, but there are systems without force-field corrections:
                weights_P[name_sys] = data.sys[name_sys].weights
                logZ_P[name_sys] = 0
        else:
            weights_P[name_sys] = data.sys[name_sys].weights
            logZ_P[name_sys] = 0

        """ 1b. if np.isinf(gamma): g is re-computed observables data.g through updated forward model
            (notice you also have some observables directly as data.g without updating of forward model);
            else: g is data.g (initial data.g[name_sys] if gamma == np.inf). """

        if np.isinf(gamma):
            if hasattr(data.sys[name_sys], 'g'):
                g[name_sys] = copy.deepcopy(data.sys[name_sys].g)
        else:
            if hasattr(data.sys[name_sys], 'g'):
                g[name_sys] = copy.deepcopy(data.sys[name_sys].g)
            else:
                g[name_sys] = {}

            if hasattr(data.sys[name_sys], 'selected_obs'):
                selected_obs = data.sys[name_sys].selected_obs
            else:
                selected_obs = None

            fm_observables = data.sys[name_sys].forward_model(pars_fm, data.sys[name_sys].forward_qs, selected_obs)

            for name in fm_observables.keys():

                g[name_sys][name] = fm_observables[name]
                if hasattr(data.sys[name_sys], 'normg_mean'):
                    g[name_sys][name] = (g[name_sys][name]-data.sys[name_sys].normg_mean[name])/data.sys[name_sys].normg_std[name]

            del fm_observables

        if (np.isinf(gamma) and hasattr(data.sys[name_sys], 'g')) or not np.isinf(gamma):
            for name in data.sys[name_sys].ref.keys():
                if data.sys[name_sys].ref[name] == '><':
                    g[name_sys][name+' LOWER'] = g[name_sys][name]
                    g[name_sys][name+' UPPER'] = g[name_sys][name]
                    del g[name_sys][name]

    """ 2. compute chi2 (if np.isinf(alpha)) or Gamma function (otherwise) """

    if np.isinf(alpha):

        av_g = {}
        chi2 = {}

        if hasattr(data._global_, 'cycle_names'):
            out = compute_DeltaDeltaG_terms(data, logZ_P)
            av_g = out[0]
            chi2 = out[1]
            loss += out[2]

        for name_sys in system_names:
            if hasattr(data.sys[name_sys], 'g'):
                out = compute_chi2(data.sys[name_sys].ref, weights_P[name_sys], g[name_sys], data.sys[name_sys].gexp, True)
                av_g[name_sys] = out[0]
                chi2[name_sys] = out[1]
                loss += 1/2*out[3]

    else:

        my_dict = {}
        for k in system_names:
            my_dict[k] = data.sys[k].n_experiments
        js = compute_js(my_dict)

        x0 = {}
        flatten_g = {}
        flatten_gexp = {}

        for i_sys, name_sys in enumerate(system_names):

            x0[name_sys] = np.array(lambdas[js[i_sys][0]:js[i_sys][-1]])
            flatten_g[name_sys] = np.hstack([g[name_sys][k] for k in data.sys[name_sys].n_experiments.keys()])
            flatten_gexp[name_sys] = np.vstack([data.sys[name_sys].gexp[k] for k in data.sys[name_sys].n_experiments.keys()])

        gamma_value = 0

        if if_fixed_lambdas:
            for name_sys in system_names:
                args = (x0[name_sys], flatten_g[name_sys], flatten_gexp[name_sys], weights_P[name_sys], alpha)
                gamma_value += gamma_function(*args)
        else:

            global minis
            minis = {}
            mini_x = []

            for name_sys in system_names:

                if bounds is not None:
                    boundaries = bounds[name_sys]
                    method = 'L-BFGS-B'
                else:
                    boundaries = None
                    method = 'BFGS'

                options = {'gtol': gtol_inn}
                if method == 'L-BFGS-B':
                    options['ftol'] = 0

                args = (flatten_g[name_sys], flatten_gexp[name_sys], weights_P[name_sys], alpha, True)
                mini = minimize(
                    gamma_function, x0[name_sys], args=args, method=method, bounds=boundaries, jac=True, options=options)

                minis[name_sys] = mini
                mini_x.append(mini.x)
                gamma_value += mini.fun

            lambdas = np.concatenate(mini_x)

        loss -= alpha*gamma_value

    """ 3. add regularization of force-field correction """

    if not np.isinf(beta):
        if not isinstance(regularization['force_field_reg'], str):
            reg_ff = regularization['force_field_reg'](pars_ff)
            loss += beta*reg_ff
        elif not regularization['force_field_reg'] == 'KL divergence':
            reg_ff = l2_regularization(pars_ff, regularization['force_field_reg'])[0]
            loss += beta*reg_ff
        else:
            reg_ff = {}
            for name_sys in correction_ff.keys():
                reg_ff[name_sys] = compute_D_KL(
                    weights_P[name_sys], correction_ff[name_sys], data.sys[name_sys].temperature, logZ_P[name_sys])
                loss += beta*reg_ff[name_sys]

    """ 4. add regularization of forward-model coefficients """
    if not np.isinf(gamma):
        reg_fm = regularization['forward_model_reg'](pars_fm, data._global_.forward_coeffs_0)
        loss += gamma*reg_fm

    """ 5. if if_save, save values (in detail) """
    if if_save:

        class Details_class:
            pass
        Details = Details_class()

        Details.loss = loss

        if not np.isinf(alpha) and not if_fixed_lambdas:
            Details.minis = minis

        if not np.isinf(beta):
            Details.weights_P = weights_P
            Details.logZ_P = logZ_P
            Details.reg_ff = reg_ff

        # just with correction to the force field and to the forward model (not to the ensemble)
        if np.isinf(alpha):
            Details.av_g = av_g
            Details.chi2 = chi2

        if not np.isinf(gamma):
            Details.reg_fm = reg_fm

        if not hasattr(Details, 'loss_explicit'):
            Details.loss_explicit = None  # for pylint

        if not np.isinf(alpha):

            """ Details_ER has attributes with names different from those of Details, as defined up to now """
            dict_lambdas = {}
            for i_sys, name_sys in enumerate(system_names):
                dict_lambdas[name_sys] = np.array(lambdas[js[i_sys][0]:js[i_sys][-1]])

            Details_ER = compute_details_ER(weights_P, g, data, dict_lambdas, alpha)

            my_keys = [x for x in dir(Details_ER) if not x.startswith('__')]
            for k in my_keys:
                setattr(Details, k, getattr(Details_ER, k))
            del Details_ER

            if hasattr(Details, 'loss_explicit'):
                if not np.isinf(beta):
                    for name_sys in system_names:
                        Details.loss_explicit += beta*reg_ff[name_sys]
                if not np.isinf(gamma):
                    Details.loss_explicit += gamma*reg_fm
            else:
                print('error in loss_explicit')

        """  just to improve the readability of the output: """
        if np.isinf(alpha):
            if np.isinf(beta) and np.isinf(gamma):
                print('all the hyperparameters are infinite')  # , namely, return original ensembles')
            elif not np.isinf(beta):
                Details.weights_new = Details.weights_P
                Details.logZ_new = Details.logZ_P
                del Details.weights_P, Details.logZ_P

        if np.isinf(alpha) and np.isinf(beta) and not np.isinf(gamma):
            Details.weights_new = {}
            for name_sys in system_names:
                Details.weights_new[name_sys] = data.sys[name_sys].weights
            print('new weights are equal to original weights')

        if Details.loss_explicit is None:
            del Details.loss_explicit  # for pylint

        return Details

    return loss


# %% C8. loss_function_and_grad


def loss_function_and_grad(
        pars: numpy.ndarray, data: dict, regularization: dict, alpha: float, beta: float, gamma: float,
        gtol_inn: float, boundaries: dict, gradient_fun):
    """
    This tool returns `loss_function` and its gradient; the gradient function, which is going to be evaluated, is computed by Jax and passed as input variable `gradient_fun`.
    If `not np.isinf(alpha)`, it appends also loss and lambdas to `intermediates.loss` and `intermediates.lambdas`, respectively.
    
    Global variable: `intermediates` (intermediate values during the minimization steps of `loss_function`).
    
    Parameters
    ----------
    
    pars : array_like
        Numpy array of parameters for force-field correction and forward model, respectively.
    
    data, regularization: dicts
        Dictionaries for `data` object and regularizations (see in `MDRefinement`).
    
    alpha, beta, gamma: floats
        Values of the hyperparameters.
    
    gtol_inn: float
        Tolerance `gtol` for the inner minimization in `loss_function`.
    
    boundaries: dict
        Dictionary of boundaries for the inner minimization in `loss_function`.
    
    gradient_fun: function
        Gradient function of `loss_function`, computed by Jax.
    """
    print('New evaluation:')
    # print('alpha, beta, gamma: ', alpha, beta, gamma)
    # print('pars: ', pars)

    loss = loss_function(pars, data, regularization, alpha, beta, gamma, None, gtol_inn, False, boundaries)

    global intermediates
    intermediates.loss.append(loss)
    intermediates.pars.append(pars)

    if not np.isinf(alpha):
        try:
            intermediates.lambdas.append(lambdas)
            intermediates.minis.append(minis)
        except:
            None

    """ now evaluate the gradient w.r.t. pars at lambdas fixed (you are in the minimum: the contribution to
    the derivative coming from lambdas is zero) """
    gradient = gradient_fun(pars, data, regularization, alpha=alpha, beta=beta, gamma=gamma, fixed_lambdas=lambdas)

    print('loss: ', loss)
    print('gradient: ', gradient, '\n')

    return loss, gradient

# %% C9. deconvolve_lambdas


def deconvolve_lambdas(data, lambdas: numpy.ndarray, if_denormalize: bool = True):
    """
    This tool deconvolves `lambdas` from Numpy array to dictionary of dictionaries (corresponding to `data.sys[name_sys].g`);
    if `if_denormalize`, then `lambdas` has been computed with normalized data, so use `data.sys[name_sys].normg_std` and `data.sys[name_sys].normg_mean`
    in order to go back to corresponding lambdas for non-normalized data. The order of `lambdas` is the one described in `compute_js`.
    """
    dict_lambdas = {}

    ns = 0

    system_names = data._global_.system_names

    for name_sys in system_names:

        dict_lambdas[name_sys] = {}

        for name in data.sys[name_sys].n_experiments.keys():
            dict_lambdas[name_sys][name] = lambdas[ns:(ns+data.sys[name_sys].n_experiments[name])]
            ns += data.sys[name_sys].n_experiments[name]

        if if_denormalize:
            assert hasattr(data.sys[name_sys], 'normg_std'), 'Error: missing normalized std values!'
            for name in data.sys[name_sys].ref.keys():
                if data.sys[name_sys].ref[name] == '><':
                    # you can sum since one of the two is zero
                    dict_lambdas[name_sys][name] = (
                        dict_lambdas[name_sys][name+' LOWER']/data.sys[name_sys].normg_std[name+' LOWER'])

                    dict_lambdas[name_sys][name] += (
                        dict_lambdas[name_sys][name+' UPPER']/data.sys[name_sys].normg_std[name+' UPPER'])

                    del dict_lambdas[name_sys][name+' LOWER'], dict_lambdas[name_sys][name+' UPPER']
                else:
                    dict_lambdas[name_sys][name] = dict_lambdas[name_sys][name]/data.sys[name_sys].normg_std[name]
        else:
            for name in data.sys[name_sys].ref.keys():
                if data.sys[name_sys].ref[name] == '><':
                    dict_lambdas[name_sys][name] = dict_lambdas[name_sys][name+' LOWER']
                    + dict_lambdas[name_sys][name+' UPPER']
                    del dict_lambdas[name_sys][name+' LOWER'], dict_lambdas[name_sys][name+' UPPER']

    return dict_lambdas

# %% C10. minimizer


class intermediates_class:
    """Class for the intermediate steps of the minimization of the loss function."""
    def __init__(self, alpha):
        
        self.loss = []
        self.pars = []

        if not np.isinf(alpha):
            self.lambdas = []
            self.minis = []


def minimizer(
        original_data, *, regularization: dict = None, alpha: float = +numpy.inf, beta: float = +numpy.inf, gamma: float = +numpy.inf,
        gtol: float = 1e-3, gtol_inn: float = 1e-3, data_test: dict = None, starting_pars: numpy.ndarray = None):
    """
    This tool minimizes loss_function on `original_data` and do `validation` on `data_test` (if `not None`), at given hyperparameters.

    Parameters
    ----------
    original_data: dict
        Dictionary for `data`-like object employed for the minimization of `loss_function`.
    
    regularization: dict
        Dictionary for the regularizations (see in `MDRefinement`).
    
    alpha, beta, gamma: floats
        Values of the hyperparameters for combined refinement (`+np.inf` by default: no refinement in that direction).
    
    gtol, gtol_inn: floats
        Tolerances `gtol` for the minimizations of `loss_function` and inner `gamma_function`, respectively.
    
    data_test: dict
        Dictionary for `data`-like object employed as test set (`None` by default, namely no validation, just minimization).
    
    starting_pars: array_like
        Numpy 1-dimensional array for pre-defined starting point of `loss_function` minimization (`None` by default).
    """
    assert alpha > 0, 'alpha must be > 0'
    assert beta >= 0, 'beta must be >= 0'
    assert gamma >= 0, 'gamma must be >= 0'

    time1 = time.time()

    system_names = original_data._global_.system_names

    """ copy original_data and act only on the copy, preserving original_data """

    data = copy.deepcopy(original_data) ## does it work?

    # data = {}
    # for k1 in original_data.keys():
    #     class my_new_class:
    #         pass
    #     my_keys = [x for x in dir(original_data[k1]) if not x.startswith('__')]
    #     for k2 in my_keys:
    #         setattr(my_new_class, k2, copy.deepcopy(getattr(original_data[k1], k2)))
    #     data[k1] = my_new_class

    """ normalize observables """
    for name_sys in system_names:
        if hasattr(data.sys[name_sys], 'g'):
            out = normalize_observables(data.sys[name_sys].gexp, data.sys[name_sys].g, data.sys[name_sys].weights)
            data.sys[name_sys].g = out[0]
            data.sys[name_sys].gexp = out[1]
            data.sys[name_sys].normg_mean = out[2]
            data.sys[name_sys].normg_std = out[3]

    """ starting point for lambdas """
    if not np.isinf(alpha):

        global lambdas

        tot_n_exp = 0

        for name in system_names:
            for item in data.sys[name].n_experiments.values():
                tot_n_exp += item

        lambdas = np.zeros(tot_n_exp)

        """ here you could duplicate lambdas for observables with both lower/upper limits """

    else:
        lambdas = None

    """ if needed, define boundaries for minimization over lambdas """

    if not np.isinf(alpha):

        my_list = []
        for k in data._global_.system_names:
            my_list = my_list + list(data.sys[k].ref.values())

        if ('>' in my_list) or ('<' in my_list) or ('><' in my_list):

            bounds = {}

            for name_sys in data._global_.system_names:
                bounds[name_sys] = []
                for name_type in data.sys[name_sys].n_experiments.keys():
                    if name_type in data.sys[name_sys].ref.keys():
                        if data.sys[name_sys].ref[name_type] == '=':
                            bounds[name_sys] = bounds[name_sys] + [(-np.inf, +np.inf)]*data.sys[name_sys].g[name_type].shape[1]
                        elif data.sys[name_sys].ref[name_type] == '<':
                            bounds[name_sys] = bounds[name_sys] + [(0, +np.inf)]*data.sys[name_sys].g[name_type].shape[1]
                        elif data.sys[name_sys].ref[name_type] == '>':
                            bounds[name_sys] = bounds[name_sys] + [(-np.inf, 0)]*data.sys[name_sys].g[name_type].shape[1]
                    elif data.sys[name_sys].ref[name_type[:-6]] == '><':
                        bounds[name_sys] = bounds[name_sys] + [(-np.inf, 0)]*data.sys[name_sys].g[name_type].shape[1]
                        # bounds = bounds + [[0,+np.inf]]*data.g[name_sys][name_type+' LOWER'].shape[1]
        else:
            bounds = None
    else:
        bounds = None

    """ minimization """

    global intermediates
    intermediates = intermediates_class(alpha)
    global minis

    if (np.isinf(beta) and np.isinf(gamma)):

        class Result_class:
            pass
        Result = Result_class()

        pars_ff_fm = None

        Result.loss = loss_function(pars_ff_fm, data, regularization, alpha, beta, gamma, None, gtol_inn, False, bounds)

        if not np.isinf(alpha):
            # since lambdas is global, it is updated inside loss_function with optimal value
            min_lambdas = lambdas
            Result.min_lambdas = deconvolve_lambdas(data, min_lambdas)
            Result.minis = minis

    else:

        """ starting point for the inner minimization """

        if starting_pars is None:
            pars_ff_fm_0 = []
            if not np.isinf(beta):
                names_ff_pars = data._global_.names_ff_pars
                pars_ff_fm_0 = pars_ff_fm_0 + list(np.zeros(len(names_ff_pars)))

            if not np.isinf(gamma):
                pars_ff_fm_0 = pars_ff_fm_0 + list(data._global_.forward_coeffs_0)
            pars_ff_fm_0 = np.array(pars_ff_fm_0)
        else:
            pars_ff_fm_0 = starting_pars

        """ minimize """
        gradient_fun = jax.grad(loss_function, argnums=0)

        args = (data, regularization, alpha, beta, gamma, gtol_inn, bounds, gradient_fun)
        mini = minimize(loss_function_and_grad, pars_ff_fm_0, args=args, method='BFGS', jac=True, options={'gtol': gtol})

        pars_ff_fm = mini.x

        class Result_class():
            def __init__(self, mini):
                self.loss = mini.fun
                self.pars = pars_ff_fm
                # self.pars = dict(zip(names, pars_ff_fm))
                self.mini = mini

        Result = Result_class(mini)

        intermediates.loss = np.array(intermediates.loss)
        intermediates.pars = np.array(intermediates.pars)

        if not np.isinf(alpha):
            """ get optimal lambdas """

            i_min = np.argmin(intermediates.loss)
            min_lambdas = intermediates.lambdas[i_min]
            minis = intermediates.minis[i_min]

            """ denormalize and deconvolve lambdas """
            Result.min_lambdas = deconvolve_lambdas(data, min_lambdas)
            Result.minis = minis

            intermediates.lambdas = np.array(intermediates.lambdas)

        Result.intermediates = intermediates

    """ return output values """

    time2 = time.time()

    Result.time = time2-time1

    """ use non-normalized data and non-normalized lambdas """
    if not np.isinf(alpha):
        flatten_lambda = []
        for name_sys in system_names:
            flatten_lambda = flatten_lambda + list(
                np.hstack(Result.min_lambdas[name_sys][k] for k in data.sys[name_sys].n_experiments.keys()))

        flatten_lambda = np.array(flatten_lambda)
    else:
        flatten_lambda = None

    Details = loss_function(
        pars_ff_fm, original_data, regularization, alpha, beta, gamma, flatten_lambda, gtol_inn, True, bounds)
    if not np.isinf(alpha):
        del Details.loss_explicit

    for k in vars(Details).keys():
        setattr(Result, k, getattr(Details, k))
    del Details

    if data_test is not None:
        Details_test = validation(
            pars_ff_fm, flatten_lambda, data_test, regularization=regularization, alpha=alpha, beta=beta, gamma=gamma,
            which_return='details')

        if not np.isinf(alpha):
            Details_test.loss = Details_test.loss_explicit
            del Details_test.loss_explicit
            # del Details_test.minis

        for k in vars(Details_test).keys():
            if not (k[-7:] == 'new_obs'):
                k1 = k + '_test'
            else:
                k1 = k
            setattr(Result, k1, getattr(Details_test, k))
        del Details_test

    return Result

# %% C11. select_traintest

class class_test:
    """
    Class for test data set, with similar structure as `data_class`.
    """
    def __init__(self, data_sys, test_frames_sys, test_obs_sys, if_all_frames, data_train_sys):

        # A. split weights
        try:
            w = data_sys.weights[test_frames_sys]
        except:
            w = data_sys.weights[list(test_frames_sys)]
        self.logZ = np.log(np.sum(w))
        self.weights = w/np.sum(w)
        self.n_frames = np.shape(w)[0]

        # B. split force-field terms
        if hasattr(data_sys, 'f'):
            self.ff_correction = data_sys.ff_correction
            try:
                self.f = data_sys.f[test_frames_sys, :]
            except:
                self.f = data_sys.f[list(test_frames_sys), :]

        # C. split experimental values gexp, normg_mean and normg_std, observables g

        if hasattr(data_sys, 'gexp'):
            self.gexp_new = {}
            self.n_experiments_new = {}

            for name_type in data_sys.gexp.keys():

                try:
                    self.gexp_new[name_type] = data_sys.gexp[name_type][list(test_obs_sys[name_type])]
                except:
                    self.gexp_new[name_type] = data_sys.gexp[name_type][test_obs_sys[name_type]]

                self.n_experiments_new[name_type] = len(test_obs_sys[name_type])

        if hasattr(data_sys, 'names'):

            self.names_new = {}

            for name_type in data_sys.names.keys():
                self.names_new[name_type] = data_sys.names[name_type][list(test_obs_sys[name_type])]

        if hasattr(data_sys, 'g'):

            self.g_new = {}
            if if_all_frames:
                self.g_new_old = {}
            self.g = {}

            for name_type in data_sys.g.keys():

                # split g into: train, test1 (non-trained obs, all frames or only non-used ones),
                # test2 (trained obs, non-used frames)
                # if not test_obs[name_sys][name_type] == []:
                self.g_new[name_type] = (data_sys.g[name_type][test_frames_sys, :].T)[test_obs_sys[name_type], :].T

                if if_all_frames:  # new observables on trained frames
                    self.g_new_old[name_type] = np.delete(
                        data_sys.g[name_type], test_frames_sys, axis=0)[:, list(test_obs_sys[name_type])]

                g3 = np.delete(data_sys.g[name_type], test_obs_sys[name_type], axis=1)
                self.g[name_type] = g3[test_frames_sys, :]

        if hasattr(data_sys, 'forward_qs'):

            self.forward_qs = {}

            for name_type in data_sys.forward_qs.keys():
                self.forward_qs[name_type] = data_sys.forward_qs[name_type][list(test_frames_sys), :]

            if if_all_frames:
                self.forward_qs_trained = data_train_sys.forward_qs

        if hasattr(data_sys, 'forward_model'):
            self.forward_model = data_sys.forward_model

        self.ref = data_sys.ref
        self.selected_obs = data_train_sys.selected_obs  # same observables as in training
        self.selected_obs_new = test_obs_sys

        self.gexp = data_train_sys.gexp
        self.n_experiments = data_train_sys.n_experiments
        self.temperature = data_sys.temperature


class class_train:
    """
    Class for training data set, with similar structure as `data_class`.
    """
    def __init__(self, data_sys, test_frames_sys, test_obs_sys):

        # training observables
        train_obs = {}
        for s in data_sys.n_experiments.keys():
            train_obs[s] = [i for i in range(data_sys.n_experiments[s]) if i not in test_obs_sys[s]]
        self.selected_obs = train_obs

        # A. split weights
        w = np.delete(data_sys.weights, test_frames_sys)
        self.logZ = np.log(np.sum(w))
        self.weights = w/np.sum(w)
        self.n_frames = np.shape(w)[0]

        # B. split force-field terms

        if hasattr(data_sys, 'f'):
            self.ff_correction = data_sys.ff_correction
            self.f = np.delete(data_sys.f, test_frames_sys, axis=0)

        # C. split experimental values gexp, normg_mean and normg_std, observables g

        if hasattr(data_sys, 'gexp'):

            self.gexp = {}
            self.n_experiments = {}

            for name_type in data_sys.gexp.keys():
                self.gexp[name_type] = np.delete(data_sys.gexp[name_type], test_obs_sys[name_type], axis=0)
                self.n_experiments[name_type] = np.shape(self.gexp[name_type])[0]

        if hasattr(data_sys, 'names'):

            self.names = {}

            for name_type in data_sys.names.keys():
                self.names[name_type] = data_sys.names[name_type][train_obs[name_type]]

        if hasattr(data_sys, 'g'):

            self.g = {}

            for name_type in data_sys.g.keys():
                train_g = np.delete(data_sys.g[name_type], test_frames_sys, axis=0)
                self.g[name_type] = np.delete(train_g, test_obs_sys[name_type], axis=1)

        if hasattr(data_sys, 'forward_qs'):

            self.forward_qs = {}

            for name_type in data_sys.forward_qs.keys():
                self.forward_qs[name_type] = np.delete(data_sys.forward_qs[name_type], test_frames_sys, axis=0)

        if hasattr(data_sys, 'forward_model'):
            self.forward_model = data_sys.forward_model

        self.ref = data_sys.ref

        self.temperature = data_sys.temperature


def select_traintest(
        data, *, test_frames_size: float = 0.2, test_obs_size: float = 0.2, random_state: int = None,
        test_frames: dict = None, test_obs: dict = None, if_all_frames: bool = False, replica_infos: dict = None):
    """
    This tool splits the data set into training and test set. You can either randomly select the frames and/or the observables (accordingly to `test_frames_size`, `test_obs_size`, `random_state`) or pass the dictionaries `test_obs` and/or `test_frames`.

    Parameters
    ----------
    data : class instance
        Class instance for the `data` object.
    
    test_frames_size, test_obs_size : float
        Values for the fractions of frames and observables for the test set, respectively. Each of them is a number in (0,1) (same fraction for every system),
        by default `0.2`.
    
    random_state : int
        The random state (or seed), used to make the same choice for different hyperparameters; if `None`,
        it is randomly taken.
    
    test_frames, test_obs : dicts
        Dictionaries for the test frames and observables.
    
    if_all_frames : bool
        Boolean variable, `False` by default; if `True`, then use all the frames for the test observables in the test set,
        otherwise just the test frames.
    
    replica_infos : dict
        Dictionary of information used to select frames based on continuous trajectories ("demuxing"), by default `None` (just randomly select frames).
        It includes: `n_temp_replica`, `path_directory`, `stride`. If not `None`, `select_traintest` will read `replica_temp.npy` files
        with shape `(n_frames, n_replicas)` containing numbers from 0 to `n_replicas - 1` which indicate corresponding
        temperatures (for each replica index in `axis=1`).
    --------
    Returns:
    --------
    data_train, data_test : class instances
        Class instances for training and test data; `data_test` includes:
        trained observables and non-trained (test) frames (where it is not specified `new`);
        non-trained (test) observables and non-trained/all (accordingly to `if_all_frames`) frames (where specified `new`).
    
    test_obs, test_frames : dicts
        Dictionaries for the observables and frames selected for the test set.
    """
    # PART 1: IF NONE, SELECT TEST OBSERVABLES AND TEST FRAMES

    system_names = data._global_.system_names
    rng = None

    if (test_frames is None) or (test_obs is None):

        if random_state is None:
            # try:
            random_state = random.randint(1000)
            # except:
            #     print('error: Jax requires to specify random state')
            #     return
            print('random_state: ', random_state)

        rng = random.default_rng(seed=random_state)
        # except: key = random.PRNGKey(random_state)

        assert (test_obs_size > 0 and test_obs_size < 1), 'error on test_obs_size'
        assert (test_frames_size > 0 and test_frames_size < 1), 'error on test_frames_size'

        # check_consistency(test_obs_size,data.n_experiments,0,data.g)
        # check_consistency(test_frames_size,data.n_frames,1,data.g)

        if test_frames is not None:
            print('Input random_state employed only for test_obs since test_frames are given')
        elif test_obs is not None:
            print('Input random_state employed only for test_frames since test_obs are given')
        else:
            print('Input random_state employed both for test_obs and test_frames')

    elif random_state is not None:
        print('Input random_state not employed, since both test_frames and test_obs are given')

    # 1B. FRAMES TEST

    if test_frames is None:

        test_frames = {}
        test_replicas = {}

        for name_sys in system_names:

            if (replica_infos is not None) and (hasattr(replica_infos, name_sys)) and ('n_temp_replica' in replica_infos[name_sys].keys()):
                # if you have demuxed trajectories, select replicas and the corresponding frames
                # pos_replcias has the indices corresponding to the different replicas

                path = replica_infos['global']['path_directory']
                stride = replica_infos['global']['stride']
                n_temp = replica_infos[name_sys]['n_temp_replica']

                replica_temp = np.load('%s/%s/replica_temp.npy' % (path, name_sys))[::stride]

                n_replicas = len(replica_temp.T)
                replica_index = replica_temp.argsort(axis=1)

                pos_replicas = []
                for i in range(n_replicas):
                    pos_replicas.append(np.argwhere(replica_index[:, n_temp] == i)[:, 0])

                n_replicas_test = np.int16(np.round(test_frames_size*n_replicas))
                test_replicas[name_sys] = np.sort(rng.choice(n_replicas, n_replicas_test, replace=False))

                fin = np.array([])
                for i in range(n_replicas_test):
                    fin = np.concatenate((fin, pos_replicas[test_replicas[name_sys][i]]))
                test_frames[name_sys] = np.array(fin).astype(int)
                del fin

            else:

                n_frames_test = np.int16(np.round(test_frames_size*data.sys[name_sys].n_frames))
                test_frames[name_sys] = np.sort(rng.choice(data.sys[name_sys].n_frames, n_frames_test, replace=False))
                # except:
                # test_frames[name_sys] = random.choice(key, data.sys[name_sys].n_frames,(n_frames_test[name_sys],),
                # replace = False)

        if test_replicas == {}:
            del test_replicas

    # 1C. OBSERVABLES TEST

    if test_obs is None:

        n_obs_test = {}
        test_obs = {}

        """ here you select with the same fraction for each type of observable"""
        # for name_sys in data.weights.keys():
        #     n_obs_test[name_sys] = {}
        #     test_obs[name_sys] = {}

        #     for name_type in data.g[name_sys].keys():
        #         n_obs_test[name_sys][name_type] = np.int16(np.round(test_obs_size*data.n_experiments[name_sys][name_type]))
        #         test_obs[name_sys][name_type] = np.sort(rng.choice(data.n_experiments[name_sys][name_type],
        #           n_obs_test[name_sys][name_type],replace = False))

        """ here instead you select the same fraction for each system and then take the corresponding observables
        (in this way, no issue for types of observables with only 1 observable """
        for name_sys in system_names:

            n_obs_test[name_sys] = {}
            test_obs[name_sys] = {}

            n = np.sum(np.array(list(data.sys[name_sys].n_experiments.values())))
            vec = np.sort(rng.choice(n, np.int16(np.round(n*test_obs_size)), replace=False))
            # except: vec = np.sort(jax.random.choice(key, n, (np.int16(np.round(n*test_obs_size)),), replace = False))

            sum = 0
            for name_type in data.sys[name_sys].n_experiments.keys():

                test_obs[name_sys][name_type] = vec[(vec >= sum) & (vec < sum + data.sys[name_sys].n_experiments[name_type])] - sum
                n_obs_test[name_sys][name_type] = len(test_obs[name_sys][name_type])

                sum += data.sys[name_sys].n_experiments[name_type]

        del sum, n, vec

    # PART 2: GIVEN test_frames and test_obs, RETURN data_test AND data_train
    # train, test1 ('non-trained' obs, all or 'non-used' frames), test2 ('trained' obs, 'non-used' frames)


    # global properties:
    
    class my_data_traintest:
        def __init__(self, data):
            self._global_ = data._global_
            self.sys = {}

    data_train = my_data_traintest(data)
    data_test = my_data_traintest(data)

    # for over different systems:

    for name_sys in system_names:

        data_train.sys[name_sys] = class_train(data.sys[name_sys], test_frames[name_sys], test_obs[name_sys])
        data_test.sys[name_sys] = class_test(
            data.sys[name_sys], test_frames[name_sys], test_obs[name_sys], if_all_frames, data_train.sys[name_sys])

    # """ if some type of observables are not included in test observables, delete them to avoid empty items """
    # for name_sys in system_names:
    #     for name_type in test_obs[name_sys].keys():
    #         if len(test_obs[name_sys][name_type]) == 0:
    #             del data_test.sys[name_sys].gexp_new[name_type]
    #             if name_type in data_test.sys[name_sys].g_new.keys():
    #                 del data_test.sys[name_sys].g_new[name_type]
    #                 if if_all_frames: del data_test.sys[name_sys].g_new_old[name_type]

    for s1 in test_obs.keys():
        my_list1 = []
        my_list2 = []

        for s2 in test_obs[s1].keys():
            if len(test_obs[s1][s2]) == 0:
                my_list1.append(s2)
            elif len(test_obs[s1][s2]) == data.sys[s1].n_experiments[s2]:
                my_list2.append(s2)

        for s2 in my_list1:
            """ no test observables of this kind """
            del data_test.sys[s1].gexp_new[s2], data_test.sys[s1].g_new[s2], data_test.sys[s1].n_experiments_new[s2]
            del data_test.sys[s1].selected_obs_new[s2]  # , data_test[s1].names_new[s2]

        for s2 in my_list2:
            """ no training observables of this kind"""
            del data_test.sys[s1].gexp[s2], data_test.sys[s1].g[s2], data_test.sys[s1].n_experiments[s2]
            del data_test.sys[s1].selected_obs[s2]  # , data_test[s1].names[s2]
            del data_train.sys[s1].gexp[s2], data_train.sys[s1].g[s2], data_train.sys[s1].n_experiments[s2]
            del data_train.sys[s1].selected_obs[s2]  # , data_train[s1].names[s2]

        for s2 in my_list1:
            test_obs[s1][s2] = np.int64(np.array([]))

    # if pos_replicas is None:
    return data_train, data_test, test_obs, test_frames
    # else:
    #     return data_train, data_test, test_obs, test_frames, test_rep

# %% C12. validation


def validation(
        pars_ff_fm, lambdas, data_test, *, regularization=None, alpha=np.inf, beta=np.inf, gamma=np.inf,
        data_train=None, which_return='details'):
    """
    This tool evaluates `loss_function` in detail over the test set; then,

    - if `which_return == 'chi2 validation'`, it returns the total chi2 on the `'validation'` data set (training observables, test frames);
    this is required to compute the derivatives of the chi2 in `'validation'` with Jax;
    
    - elif `which_return == 'chi2 test'`, it returns the total chi2 on the `'test'` data set (test observables, test frames
        or all frames if `data_train is not None`); this is required to compute the derivatives of the chi2 in `'test'` with Jax;
    
    - else, it returns `Validation_values` class instance, with all the computed values (both chi2 and regularizations).

    Parameters
    ----------
    
    pars_ff_fm: array_like
        Numpy 1-dimensional array for the force-field and forward-model coefficients.
    
    lambdas: array_like
        Numpy 1-dimensional array of lambdas coefficients (those for ensemble refinement).
    
    data_test: dict
        Dictionary for the test data set, `data`-like object, as returned by `select_traintest`.
    
    regularization: dict
        Dictionary for the regularizations (see in `MDRefinement`), by default, `None`.
    
    alpha, beta, gamma: floats
        Values for the hyperparameters (by default, `+np.inf`, namely, no refinement).
    
    data_train: dict
        Dictionary for the training data set, `data`-like object, as returned by `select_traintest` (`None` by default,
        namely use only test frames for new observables).
    
    which_return: str
        String described above (by default `'details'`).
    """
    assert alpha > 0, 'alpha must be > 0'
    assert beta >= 0, 'beta must be >= 0'
    assert gamma >= 0, 'gamma must be >= 0'

    system_names = data_test._global_.system_names
    names_ff_pars = []

    if not np.isinf(beta):
        names_ff_pars = data_test._global_.names_ff_pars

    pars_fm = None  # to avoid error in pylint
    if not np.isinf(gamma):
        pars_fm = pars_ff_fm[len(names_ff_pars):]
    if names_ff_pars == []:
        del names_ff_pars

    """ Compute loss_function in detail for validating set (same observables as in training, new frames). """
    Validation_values = loss_function(pars_ff_fm, data_test, regularization, alpha, beta, gamma, lambdas, if_save=True)

    if which_return == 'chi2 validation':
        tot_chi2 = 0
        for s1 in Validation_values.chi2.keys():
            for item in Validation_values.chi2[s1].values():
                tot_chi2 += item
        return tot_chi2

    # let's compute firstly the average of non-trained (validating) observables on new frames

    Validation_values.avg_new_obs = {}
    Validation_values.chi2_new_obs = {}

    # if hasattr(data_test,'selected_obs'):
    #     for name in data_test.forward_qs.keys():
    #         for type_name in data_test.forward_qs[name].keys():
    #             data_test.forward_qs[name][type_name] = data_test.forward_qs[name][type_name]
    #               #[:,data_test.selected_obs[name][type_name]]

    g = {}

    for name_sys in system_names:

        if np.isinf(gamma):
            if hasattr(data_test.sys[name_sys], 'g_new'):
                g[name_sys] = copy.deepcopy(data_test.sys[name_sys].g_new)
        else:
            if hasattr(data_test.sys[name_sys], 'g_new'):
                g[name_sys] = copy.deepcopy(data_test.sys[name_sys].g_new)
            else:
                g[name_sys] = {}

            if hasattr(data_test.sys[name_sys], 'selected_obs'):
                selected_obs = data_test.sys[name_sys].selected_obs_new
            else:
                selected_obs = None

            fm_observables = data_test.sys[name_sys].forward_model(pars_fm, data_test.sys[name_sys].forward_qs, selected_obs)

            for name in fm_observables.keys():

                g[name_sys][name] = fm_observables[name]
                if hasattr(data_test.sys[name_sys], 'normg_mean'):
                    g[name_sys][name] = (
                        g[name_sys][name]-data_test.sys[name_sys].normg_mean[name])/data_test.sys[name_sys].normg_std[name]

            del fm_observables

    for name_sys in system_names:

        args = (data_test.sys[name_sys].ref, Validation_values.weights_new[name_sys], g[name_sys], data_test.sys[name_sys].gexp_new)
        out = compute_chi2(*args)

        Validation_values.avg_new_obs[name_sys] = out[0]

        if not hasattr(data_test.sys[name_sys], 'forward_qs_trained'):
            Validation_values.chi2_new_obs[name_sys] = out[1]

    # then, if you want to include also trained frames for validating observables:

    if hasattr(data_test.sys[system_names[0]], 'forward_qs_trained') and (data_train is not None):  # forward qs on trained frames

        Details_train = loss_function(pars_ff_fm, data_train, regularization, alpha, beta, gamma, lambdas, if_save=True)

        g = {}

        for name_sys in system_names:
            if np.isinf(gamma):
                if hasattr(data_test.sys[name_sys], 'g_new_old'):
                    g[name_sys] = copy.deepcopy(data_test.sys[name_sys].g_new_old)
            else:
                if hasattr(data_test.sys[name_sys], 'g_new_old'):
                    g[name_sys] = copy.deepcopy(data_test.sys[name_sys].g_new_old)
                else:
                    g[name_sys] = {}

                if hasattr(data_test.sys[name_sys], 'selected_obs'):
                    selected_obs = data_test.sys[name_sys].selected_obs
                else:
                    selected_obs = None

                fm_observables = data_test.sys[name_sys].forward_model(pars_fm, data_test.sys[name_sys].forward_qs, selected_obs)

                for name in fm_observables.keys():

                    g[name_sys][name] = fm_observables[name]
                    if hasattr(data_test.sys[name_sys], 'normg_mean'):
                        g[name_sys][name] = (
                            g[name_sys][name]-data_test.sys[name_sys].normg_mean[name])/data_test.sys[name_sys].normg_std[name]

                del fm_observables

            Validation_values.chi2_new_obs[name_sys] = {}

            args = (data_test.sys[name_sys].ref, Details_train.weights_new[name_sys], g[name_sys], data_test.sys[name_sys].gexp_new)
            out = compute_chi2(*args)[0]

            log_fact_Z = data_test.sys[name_sys].logZ + Validation_values.logZ_new[name_sys]
            - Details_train.logZ_new[name_sys] - data_train.sys[name_sys].logZ

            if hasattr(Validation_values, 'logZ_P'):
                log_fact_Z += Validation_values.logZ_P_test[name_sys] - Details_train.logZ_P[name_sys]

            for name_type in data_test.sys[name_sys].n_experiments.keys():
                Validation_values.avg_new_obs[name_sys][name_type] = 1/(1+np.exp(log_fact_Z))*out[name_type]
                + 1/(1+np.exp(-log_fact_Z))*Validation_values.avg_new_obs[name_sys][name_type]

                Validation_values.chi2_new_obs[name_sys][name_type] = np.sum(((
                    Validation_values.avg_new_obs[name_sys][name_type]
                    - data_test.sys[name_sys].gexp_new[name_type][:, 0])/data_test.sys[name_sys].gexp_new[name_type][:, 1])**2)

    if which_return == 'chi2 test':
        tot_chi2 = 0
        for s1 in Validation_values.chi2_new_obs.keys():
            for item in Validation_values.chi2_new_obs[s1].values():
                tot_chi2 += item
        return tot_chi2
    return Validation_values
