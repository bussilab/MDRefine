"""
Tools n. 3: `hyperminimizer`.
It performs the automatic search for the optimal hyperparameters.
"""

import copy
from scipy.optimize import minimize
from joblib import Parallel, delayed

# numpy is required for loadtxt and for gradient arrays with L-BFGS-B minimization (rather than jax.numpy)
import numpy
import jax
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)

from .loss_and_minimizer import compute_js, compute_new_weights, gamma_function, loss_function, minimizer
from .loss_and_minimizer import split_dataset, validation, print_references

# %% D. (automatic) optimization of the hyper parameters through minimization of chi2


""" Use implicit function theorem to compute the derivatives of the pars_ff_fm and lambdas w.r.t. hyper parameters. """


# %% D1. compute_hyperderivatives


def compute_hyperderivatives(
        pars_ff_fm, lambdas, data, regularization, derivatives_funs,
        log10_alpha=+np.inf, log10_beta=+np.inf, log10_gamma=+np.inf):
    """
    This is an internal tool of `compute_hypergradient` which computes the derivatives of parameters with respect to hyperparameters,
    which are going to be used later to compute the derivatives of chi2 w.r.t. hyperparameters.
    It returns an instance of the class `derivatives`, which includes as attributes the numerical values of 
    the derivatives `dlambdas_dlogalpha`, `dlambdas_dpars`, `dpars_dlogalpha`, `dpars_dlogbeta`, `dpars_dloggamma`.

    Parameters
    ----------
    
    pars_ff_fm: array_like
        Numpy array for force-field and forward-model coefficients.
    
    lambdas: array_like
        Numpy array for lambdas coefficients (those for ensemble refinement).
    
    data: dict
        The `data` object.
    
    regularization: dict
        The regularization of force-field and forward-model corrections (see in `MDRefinement`).
    
    derivatives_funs: class instance
        Instance of the `derivatives_funs_class` class of derivatives functions computed by Jax.
    
    log10_alpha, log10_beta, log10_gamma: floats
        Logarithms (in base 10) of the corresponding hyperparameters alpha, beta, gamma (`np.inf` by default).
    """
    system_names = data.properties.system_names

    if np.isposinf(log10_beta) and np.isposinf(log10_gamma) and not np.isinf(log10_alpha):

        alpha = np.float64(10**log10_alpha)

        data_n_experiments = {}
        for k in system_names:
            data_n_experiments[k] = data.mol[k].n_experiments
        js = compute_js(data_n_experiments)

        class derivatives:
            pass

        derivatives.dlambdas_dlogalpha = []

        for i_sys, name_sys in enumerate(system_names):

            my_lambdas = lambdas[js[i_sys][0]:js[i_sys][-1]]
            # indices = np.nonzero(my_lambdas)[0]

            refs = []
            for name in data.mol[name_sys].n_experiments.keys():
                refs.extend(data.mol[name_sys].ref[name]*data.mol[name_sys].n_experiments[name])

            # indices of lambdas NOT on constraints
            indices = np.array([k for k in range(len(my_lambdas)) if ((not my_lambdas[k] == 0) or (refs[k] == '='))])

            if len(indices) == 0:
                print('all lambdas of system %s are on boundaries!' % name_sys)

            else:

                my_lambdas = my_lambdas[indices]

                g = np.hstack([data.mol[name_sys].g[k] for k in data.mol[name_sys].n_experiments.keys()])[:, indices]
                gexp = np.vstack([data.mol[name_sys].gexp[k] for k in data.mol[name_sys].n_experiments.keys()])[indices]

                my_args = (my_lambdas, g, gexp, data.mol[name_sys].weights, alpha)
                Hess_inv = np.linalg.inv(derivatives_funs.d2gamma_dlambdas2(*my_args))

                derivatives.dlambdas_dlogalpha.append(
                    -np.matmul(Hess_inv, derivatives_funs.d2gamma_dlambdas_dalpha(*my_args))*alpha*np.log(10))

    elif not (np.isposinf(log10_beta) and np.isposinf(log10_gamma)):

        pars_ff_fm = np.array(pars_ff_fm)

        class derivatives:
            pass

        alpha = np.float64(10**log10_alpha)
        beta = np.float64(10**log10_beta)
        gamma = np.float64(10**log10_gamma)

        args = (pars_ff_fm, data, regularization, alpha, beta, gamma, lambdas)

        if not np.isinf(alpha):

            d2loss_dpars_dlambdas = derivatives_funs.d2loss_dpars_dlambdas(*args)

            data_n_experiments = {}
            for k in system_names:
                data_n_experiments[k] = data.mol[k].n_experiments
            js = compute_js(data_n_experiments)

            """
            Here use Gamma function, in this way you do multiple inversions, rather than a single inversion
            of a very big matrix: different systems have uncorrelated Ensemble Refinement
            BUT you have to evaluate Gamma at given phi, theta !!
            """

            derivatives.dlambdas_dlogalpha = []
            derivatives.dlambdas_dpars = []

            terms = []  # terms to add to get d2loss_dmu2 deriving from lambdas contribution
            terms2 = []

            names_ff_pars = []

            """ compute new weights with ff correction phi """
            if not np.isposinf(beta):

                names_ff_pars = data.properties.names_ff_pars
                pars_ff = pars_ff_fm[:len(names_ff_pars)]

                correction_ff = {}
                weights_P = {}
                logZ_P = {}

                for name in system_names:
                    if hasattr(data.mol[name], 'ff_correction'):
                        correction_ff[name] = data.mol[name].ff_correction(pars_ff, data.mol[name].f)
                        correction_ff[name] = correction_ff[name]/data.mol[name].temperature
                        weights_P[name], logZ_P[name] = compute_new_weights(data.mol[name].weights, correction_ff[name])

                    else:  # if beta is not infinite, but there are systems without force-field corrections:
                        weights_P[name] = data.mol[name].weights
                        logZ_P[name] = 0
            else:
                weights_P = {}
                for name in system_names:
                    weights_P[name] = data.mol[name].weights

            """ compute forward quantities through (new) forward coefficients theta"""

            pars_fm = pars_ff_fm[len(names_ff_pars):]

            g = {}

            if np.isposinf(gamma):

                for name in system_names:
                    if hasattr(data.mol[name], 'g'):
                        g[name] = copy.deepcopy(data.mol[name].g)
            else:

                for name_sys in system_names:
                    if hasattr(data.mol[name_sys], 'g'):
                        g[name_sys] = copy.deepcopy(data.mol[name_sys].g)
                    else:
                        g[name_sys] = {}

                    if hasattr(data.mol[name_sys], 'selected_obs'):
                        selected_obs = data.mol[name_sys].selected_obs
                    else:
                        selected_obs = None

                    fm_observables = data.mol[name_sys].forward_model(pars_fm, data.mol[name_sys].forward_qs, selected_obs)

                    for name in fm_observables.keys():
                        g[name_sys][name] = fm_observables[name]

                    del fm_observables

            """ Compute derivatives and Hessian. """

            for i_sys, name_sys in enumerate(system_names):

                my_lambdas = lambdas[js[i_sys][0]:js[i_sys][-1]]

                """ use indices to select lambdas NOT on constraints """
                refs = []
                for name in data.mol[name_sys].n_experiments.keys():
                    refs.extend(data.mol[name_sys].ref[name]*data.mol[name_sys].n_experiments[name])

                # indices of lambdas NOT on constraints
                indices = np.array([k for k in range(len(my_lambdas)) if ((not my_lambdas[k] == 0) or (refs[k] == '='))])

                if len(indices) == 0:
                    print('all lambdas of system %s are on boundaries!' % name_sys)

                else:

                    my_lambdas = my_lambdas[indices]

                    my_g = np.hstack([g[name_sys][k] for k in data.mol[name_sys].n_experiments])[:, indices]
                    my_gexp = np.vstack([data.mol[name_sys].gexp[k] for k in data.mol[name_sys].n_experiments])[indices]

                    my_args = (my_lambdas, my_g, my_gexp, weights_P[name_sys], alpha)

                    Hess_inn_inv = np.linalg.inv(derivatives_funs.d2gamma_dlambdas2(*my_args))

                    derivatives.dlambdas_dlogalpha.append(
                        -np.matmul(Hess_inn_inv, derivatives_funs.d2gamma_dlambdas_dalpha(*my_args))*alpha*np.log(10))

                    matrix = d2loss_dpars_dlambdas[:, js[i_sys][0]:js[i_sys][-1]][:, indices]
                    derivatives.dlambdas_dpars.append(+np.matmul(Hess_inn_inv, matrix.T)/alpha)
                    terms.append(np.einsum('ij,jk,kl->il', matrix, Hess_inn_inv, matrix.T))
                    terms2.append(np.matmul(matrix, derivatives.dlambdas_dlogalpha[-1]))

            if not terms == []:
                Hess = +np.sum(np.array(terms), axis=0)/alpha + derivatives_funs.d2loss_dpars2(*args)
                terms2 = np.sum(np.array(terms2), axis=0)
            else:
                Hess = derivatives_funs.d2loss_dpars2(*args)
                terms2 = np.zeros(Hess.shape[0])

        else:
            Hess = derivatives_funs.d2loss_dpars2(*args)

        inv_Hess = np.linalg.inv(Hess)

        if not np.isinf(alpha):
            d2loss_dpars_dlogalpha = derivatives_funs.d2loss_dpars_dalpha(*args)*alpha*np.log(10)
            derivatives.dpars_dlogalpha = -np.matmul(inv_Hess, d2loss_dpars_dlogalpha + terms2)
        if not np.isposinf(beta):
            d2loss_dpars_dbeta = derivatives_funs.d2loss_dpars_dbeta(*args)
            derivatives.dpars_dlogbeta = -np.matmul(inv_Hess, d2loss_dpars_dbeta)*beta*np.log(10)
        if not np.isposinf(gamma):
            d2loss_dpars_dgamma = derivatives_funs.d2loss_dpars_dgamma(*args)
            derivatives.dpars_dloggamma = -np.matmul(inv_Hess, d2loss_dpars_dgamma)*gamma*np.log(10)

    return derivatives

# %% D2. compute_chi2_tot

def compute_chi2_tot(pars_ff_fm, lambdas, data, regularization, alpha, beta, gamma, which_set, data_train):
    """
    This function is an internal tool used in `compute_hypergradient` and `hyper_minimizer`
    to compute the total chi2 (float variable) for training or validation data set and its derivatives
    (with respect to `pars_ff_fm` and `lambdas`). The choice of the data set is indicated by `which_set`
    (`which_set = 'training'` for chi2 on the training set, `'valid_frames'` for chi2 on training observables and validation frames,
    `'validation'` for chi2 on validation observables and validation frames, through validation function).

    Parameters
    ----------
    
    pars_ff_fm, lambdas: array_like
        Numpy arrays for (force-field + forward-model) parameters and lambdas parameters, respectively.
    
    data: dict
        Dictionary of data set object.
    
    regularization: dict
        Specified regularizations of force-field and forward-model corrections (see in `MDRefinement`).
    
    alpha, beta, gamma: float
        Values of the hyperparameters.
    
    which_set: str
        String variable, chosen among `'training'`, `'valid_frames'`, `'valid_obs'` or `'validation'` as explained above.
    
    data_train: dict
        Dictionary of training dataset objects, required if `which_set = 'valid_obs'` to compute the chi2
        on validating obvervables including also training frames. 
    """
    if which_set == 'training' or which_set == 'valid_frames':
        tot_chi2 = 0

        Details = loss_function(pars_ff_fm, data, regularization, alpha, beta, gamma, fixed_lambdas=lambdas, if_save=True)

        for s1 in Details.chi2.keys():
            for item in Details.chi2[s1].values():
                tot_chi2 += item

    elif which_set == 'validation' or which_set == 'valid_obs':
        # validation set includes observables left out from training
        # 'validation' includes only left-out frames, while 'valid_obs' also training frames

        if which_set == 'valid_obs':
            tot_chi2 = validation(
                pars_ff_fm, lambdas, data, regularization=regularization, alpha=alpha, beta=beta, gamma=gamma,
                data_train=data_train, which_return='chi2 validation')
        else:
            tot_chi2 = validation(
                pars_ff_fm, lambdas, data, regularization=regularization, alpha=alpha, beta=beta, gamma=gamma,
                data_train=None, which_return='chi2 validation')

    assert tot_chi2, 'error in compute_chi2_tot' + which_set

    return tot_chi2

# %% D3. put_together

def put_together(dchi2_dpars, dchi2_dlambdas, derivatives):
    """
    This is an internal tool of `compute_hypergradient` which applies the chain rule in order to get the derivatives of chi2 w.r.t hyperparameters from
    derivatives of chi2 w.r.t. parameters and derivatives of parameters w.r.t. hyperparameters.

    Parameters
    ----------
    dchi2_dpars: array-like
        Numpy 1-dimensional array with derivatives of chi2 w.r.t. `pars_ff_fm` (force-field and forward-model parameters).
    
    dchi2_dlambdas: array-like
        Numpy 1-dimensional array with derivatives of chi2 w.r.t. `lambdas` (same order of `lambdas` in `dchi2_dlambdas` and in `derivatives`).
    
    derivatives: class instance
        Class instance with derivatives of `pars_ff_fm` and `lambdas` w.r.t. hyperparameters (determined in `compute_hyperderivatives`).

    -------
    Returns
    -------
    out: class instance
        Class instance whose attributes can include `dchi2_dlogalpha`, `dchi2_dlogbeta`, `dchi2_dloggamma`,
    depending on which hyperparameters are not fixed to `+np.inf`.
    """
    class out_class:
        pass
    out = out_class()

    if dchi2_dpars is None:
        if dchi2_dlambdas is not None:
            out.dchi2_dlogalpha = np.dot(dchi2_dlambdas, derivatives.dlambdas_dlogalpha)
        else:
            out.dchi2_dlogalpha = np.zeros(1)

    elif dchi2_dpars is not None:

        vec = dchi2_dpars

        if dchi2_dlambdas is not None:

            vec += np.einsum('i,ij', dchi2_dlambdas, derivatives.dlambdas_dpars)
            temp = np.dot(dchi2_dlambdas, derivatives.dlambdas_dlogalpha)

            out.dchi2_dlogalpha = np.dot(vec, derivatives.dpars_dlogalpha) + temp

        elif hasattr(derivatives, 'dpars_dlogalpha'):  # namely, if np.isinf(alpha) and zero contribute from lambdas
            out.dchi2_dlogalpha = np.dot(vec, derivatives.dpars_dlogalpha)

        if hasattr(derivatives, 'dpars_dlogbeta'):
            out.dchi2_dlogbeta = np.dot(vec, derivatives.dpars_dlogbeta)
        if hasattr(derivatives, 'dpars_dloggamma'):
            out.dchi2_dloggamma = np.dot(vec, derivatives.dpars_dloggamma)

    return out

# %% D4. compute_hypergradient


def compute_hypergradient(
        pars_ff_fm, lambdas, log10_alpha, log10_beta, log10_gamma, data_train, regularization,
        which_set, data_valid, derivatives_funs):
    """
    This is an internal tool of `mini_and_chi2_and_grad`, which employs previously defined functions (`compute_hyperderivatives`, `compute_chi2_tot`,
    `put_together`) to return selected chi2 and its gradient w.r.t hyperparameters.

    Parameters
    ----------
    
    pars_ff_fm: array_like
        Numpy array of (force-field and forward-model) parameters.
    
    lambdas: dict
        Dictionary of dictionaries with lambda coefficients (corresponding to Ensemble Refinement).
    
    log10_alpha, log10_beta, log10_gamma: floats
        Logarithms (in base 10) of the hyperparameters alpha, beta, gamma.
    
    data_train: class instance
        The training data set object, which is anyway required to compute the derivatives of parameters w.r.t. hyper-parameters.
    
    regularization: dict
        Specified regularizations (see in `MDRefinement`).
    
    which_set: str
        String indicating which set defines the chi2 to minimize in order to get the optimal hyperparameters (see in `compute_chi2_tot`).
    
    data_valid: class instance
        The validation data set object, which is required to compute the chi2 on the validation set (when `which_set == 'valid_frames' or 'validation'`;
        otherwise, if `which_set = 'training'`, it is useless, so it can be set to `None`).
    
    derivatives_funs: class instance
        Instance of the `derivatives_funs_class` class of derivatives functions computed by Jax Autodiff (they include those employed in `compute_hyperderivatives`
        and `dchi2_dpars` and/or `dchi2_dlambdas`). If None (default value), do not compute the derivatives of chi2.
    """
    system_names = data_train.properties.system_names

    """ make lambdas_vec, that will be used to compute derivatives of optimal pars w.r.t. hyper parameters """
    if not np.isinf(log10_alpha):
        lambdas_vec = []
        refs = []

        for name_sys in system_names:
            for name in data_train.mol[name_sys].n_experiments.keys():
                lambdas_vec.append(lambdas[name_sys][name])
                refs.extend(data_train.mol[name_sys].ref[name]*data_train.mol[name_sys].n_experiments[name])

        lambdas_vec = np.concatenate((lambdas_vec))

        """ indices of lambdas NOT on constraints """
        indices = np.array([k for k in range(len(lambdas_vec)) if ((not lambdas_vec[k] == 0) or (refs[k] == '='))])
        # indices = np.nonzero(lambdas_vec)[0]

        if len(indices) == 0:
            print('all lambdas are on boundaries!')
            if np.isinf(log10_beta) and np.isinf(log10_gamma):
                print('no suggestion on how to move in parameter space!')
                # gradient = np.zeros(1)

    else:
        lambdas_vec = None


    """ compute chi2 and its derivatives w.r.t. pars"""

    assert which_set in ['training', 'valid_frames', 'validation', 'valid_obs'], 'error on which_set'
    if which_set == 'training':
        my_data = data_train
    else:
        my_data = data_valid

    my_args = (
        pars_ff_fm, lambdas_vec, my_data, regularization, 10**(log10_alpha), 10**(log10_beta),
        10**(log10_gamma), which_set, data_train)

    chi2 = compute_chi2_tot(*my_args)  # so, lambdas follows order of system_names of my_data

    # if (len(indices) == 0) and np.isinf(log10_beta) and np.isinf(log10_gamma):
    #     return chi2, np.zeros(1)

    if derivatives_funs is None:
        return chi2
    else:
        # use non-normalized data and lambdas
        derivatives = compute_hyperderivatives(
            pars_ff_fm, lambdas_vec, data_train, regularization, derivatives_funs, log10_alpha, log10_beta, log10_gamma)
        
        if not (np.isinf(log10_beta) and np.isinf(log10_gamma)):
            dchi2_dpars = derivatives_funs.dchi2_dpars(*my_args)
        else:
            dchi2_dpars = None
        if not (np.isinf(log10_alpha) or len(indices) == 0):
            dchi2_dlambdas = derivatives_funs.dchi2_dlambdas(*my_args)
            dchi2_dlambdas = dchi2_dlambdas[indices]
        else:
            dchi2_dlambdas = None

        """ compute derivatives of chi2 w.r.t. hyper parameters (put together the previous two) """

        if hasattr(derivatives, 'dlambdas_dlogalpha') and not derivatives.dlambdas_dlogalpha == []:
            # ks = [k for k in system_names if k in derivatives.dlambdas_dlogalpha.keys()]
            derivatives.dlambdas_dlogalpha = np.concatenate(derivatives.dlambdas_dlogalpha)
        if hasattr(derivatives, 'dlambdas_dpars') and not derivatives.dlambdas_dpars == []:
            # ks = [k for k in system_names if k in derivatives.dlambdas_dpars.keys()]
            derivatives.dlambdas_dpars = np.concatenate(derivatives.dlambdas_dpars)

        gradient = put_together(dchi2_dpars, dchi2_dlambdas, derivatives)

        ### to have reduced chi2
        # n_obs = 0

        # if which_set == 'training':
        #     for s in data_train.mol.keys():
        #         n_obs += np.sum(np.array(list(data_train.mol[s].n_experiments.values())))
        # elif which_set == 'valid_frames':
        #     for s in data_valid.mol.keys():
        #         n_obs += np.sum(np.array(list(data_valid.mol[s].n_experiments.values())))
        # else:
        #     for s in data_valid.mol.keys():
        #         n_obs += np.sum(np.array(list(data_valid.mol[s].n_experiments_new.values())))

        # chi2 = chi2/n_obs

        # if 'dchi2_dlogalpha' in vars(gradient).keys():
        #     gradient.dchi2_dlogalpha = gradient.dchi2_dlogalpha/n_obs
        # if 'dchi2_dlogbeta' in vars(gradient).keys():
        #     gradient.dchi2_dlogbeta = gradient.dchi2_dlogbeta/n_obs
        # if 'dchi2_dloggamma' in vars(gradient).keys():
        #     gradient.dchi2_dloggamma = gradient.dchi2_dloggamma/n_obs

        return chi2, gradient


# %% D5. mini_and_chi2_and_grad

def mini_and_chi2_and_grad(
        data, valid_frames, valid_obs, regularization, alpha, beta, gamma,
        starting_pars, which_set, derivatives_funs):
    """
    This is an internal tool of `hyper_function` which minimizes the loss function at given hyperparameters, computes the chi2 and
    its gradient w.r.t. the hyperparameters.

    Parameters
    ----------
    data : class instance
        Class instance which constitutes the `data` object.
    
    valid_frames, valid_obs : dicts
        Dictionaries for validation frames and validation observables (for a given `random_state`).

    regularization : dict
        Dictionary for the regularizations (see in `MDRefinement`).

    alpha, beta, gamma : floats
        Values of the hyperparameters.

    starting_pars : array_like
        Numpy 1-dimensional array for starting values of the coefficients in `minimizer`.

    which_set : str
        String among `'training'`, `'valid_frames'` or `'valid'` (see in `MDRefinement`).

    derivatives_funs : `derivatives_funs_class` object
        Instance of the `derivatives_funs_class` class of derivatives functions computed by Jax Autodiff.
    """

    if which_set == 'valid_obs': if_all_frames = True  # include also training frames in the validation set (new observables)
    else: if_all_frames = False
    
    out = split_dataset(data, valid_frames=valid_frames, valid_obs=valid_obs, if_verbose=False, if_all_frames=if_all_frames)
    data_train = out[0]
    data_valid = out[1]

    mini = minimizer(
        data_train, regularization=regularization, alpha=alpha, beta=beta, gamma=gamma, starting_pars=starting_pars, if_print_biblio=False)

    if hasattr(mini, 'pars'):
        pars_ff_fm = mini.pars
    else:
        pars_ff_fm = None
    if hasattr(mini, 'min_lambdas'):
        lambdas = mini.min_lambdas
    else:
        lambdas = None

    chi2, gradient = compute_hypergradient(
        pars_ff_fm, lambdas, np.log10(alpha), np.log10(beta), np.log10(gamma), data_train, regularization,
        which_set, data_valid, derivatives_funs)

    return mini, chi2, gradient

# %% D6. hyper_function


def hyper_function(
        log10_hyperpars, map_hyperpars, data, regularization, valid_obs, valid_frames, which_set,
        derivatives_funs, starting_pars, n_parallel_jobs):
    """
    This function is an internal tool of `hyper_minimizer` which determines the optimal parameters by minimizing the loss function at given hyperparameters;
    then, it computes chi2 and its gradient w.r.t hyperparameters (for the optimal parameters).

    Parameters
    ----------
    
    log10_hyperpars: array_like
        Numpy array for log10 hyperparameters alpha, beta, gamma (in this order, when present).
    
    map_hyperpars: list
        Legend for `log10_hyperpars` (they refer to alpha, beta, gamma in this order,
        but some of them may not be present, if fixed to `+np.inf`).
    
    data: class instance
        Class instance for `data` object.
    
    regularization: dict
        Dictionaries for `regularization` object.
    
    valid_obs, valid_frames: dicts
        Dictionaries for validation observables and validation frames, indicized by seeds.
    
    which_set: str
        String, see for `compute_chi2_tot`.
    
    derivatives_funs: class instance
        Derivative functions computed by `Jax` and employed in `compute_hypergradient`.
    
    starting_pars: float
        Starting values of the parameters, if user-defined; `None` otherwise.

    n_parallel_jobs: int
        Number of parallel jobs.

    --------

    Returns
    --------
    
    tot_chi2: float
        Float value of total chi2.
    
    tot_gradient: array_like
        Numpy array for gradient of total chi2 with respect to the hyperparameters.
    
    Results: class instance
        Results given by `minimizer`.

    --------------
    Global variable: `hyper_intermediate`, in order to follow steps of minimization.
    """
    # 0. input values

    i = 0
    if 'alpha' in map_hyperpars:
        log10_alpha = log10_hyperpars[i]
        i += 1
    else:
        log10_alpha = np.inf
    if 'beta' in map_hyperpars:
        log10_beta = log10_hyperpars[i]
        i += 1
    else:
        log10_beta = np.inf
    if 'gamma' in map_hyperpars:
        log10_gamma = log10_hyperpars[i]
    else:
        log10_gamma = np.inf

    print('\nlog10 hyperpars: ', [(str(map_hyperpars[i]), log10_hyperpars[i]) for i in range(len(map_hyperpars))])

    alpha = np.float64(10**log10_alpha)
    beta = np.float64(10**log10_beta)
    gamma = np.float64(10**log10_gamma)

    names_ff_pars = []

    if not np.isinf(beta):
        names_ff_pars = data.properties.names_ff_pars
        pars0 = np.zeros(len(names_ff_pars))
    else:
        pars0 = np.array([])

    if not np.isinf(gamma):
        pars0 = np.concatenate(([pars0, np.array(data.properties.forward_coeffs_0)]))

    """ for each seed: """

    # Results = {}
    # chi2 = []
    # gradient = []  # derivatives of chi2 w.r.t. (log10) hyper parameters

    # args = (data, valid_frames[i], valid_obs[i], regularization, alpha, beta, gamma, starting_pars,
    # which_set, derivatives_funs)
    random_states = valid_obs.keys()

    if n_parallel_jobs is None:
        n_parallel_jobs = len(valid_obs)

    output = Parallel(n_jobs=n_parallel_jobs)(delayed(mini_and_chi2_and_grad)(
        data, valid_frames[seed], valid_obs[seed], regularization, alpha, beta, gamma, starting_pars,
        which_set, derivatives_funs) for seed in random_states)

    Results = [output[i][0] for i in range(len(random_states))]
    chi2 = [output[i][1] for i in range(len(random_states))]
    gradient = [output[i][2] for i in range(len(random_states))]

    av_chi2 = np.mean(np.array(chi2))

    av_gradient = []

    if 'alpha' in map_hyperpars:
        av_gradient.append(np.mean(np.array([gradient[k].dchi2_dlogalpha for k in range(len(random_states))])))
    if 'beta' in map_hyperpars:
        av_gradient.append(np.mean(np.array([gradient[k].dchi2_dlogbeta for k in range(len(random_states))])))
    if 'gamma' in map_hyperpars:
        av_gradient.append(np.mean(np.array([gradient[k].dchi2_dloggamma for k in range(len(random_states))])))

    av_gradient = numpy.array(av_gradient)

    print('av. chi2: ', av_chi2)
    print('av. gradient: ', av_gradient)

    global hyper_intermediate
    hyper_intermediate.av_chi2.append(av_chi2)
    hyper_intermediate.av_gradient.append(av_gradient)
    hyper_intermediate.log10_hyperpars.append(log10_hyperpars)

    return av_chi2, av_gradient, Results

# %% D7. hyper_minimizer


def hyper_minimizer(
        data, starting_alpha=+np.inf, starting_beta=+np.inf, starting_gamma=+np.inf,
        regularization=None, random_states=1, which_set='valid_frames',
        gtol=0.5, ftol=0.05, starting_pars=None, n_parallel_jobs=None, if_print_biblio=True):
    """
    This tool optimizes the hyperparameters by minimizing the selected chi2 ('training', 'valid_frames' or 'validation')
    over several (randomly) splits of the full data set into training/validation set.

    Parameters
    ----------
    data : class instance
        Object `data`, with the full data set previously loaded.
    
    starting_alpha, starting_beta, starting_gamma : floats
        Starting points of the hyperparameters (`+np.inf` by default, namely no refinement in that direction).
    
    regularization : dict
        Dictionary for the defined regularizations of force-field and forward-model corrections (`None` by default); see for `MDRefinement`.

    random_states : int or list
        Random states (i.e., seeds) used in `split_dataset` to split the data set into training and validation set (see `MDRefinement`); 1 by default.
    
    which_set : str
        String choosen among `'training'`, `'valid_frames'`, `'valid'` (see in `MDRefinement`); `validation` by default.
    
    gtol : float
        Tolerance `gtol` of `scipy.optimize.minimize` (0.5 by default).
    
    ftol : float
        Tolerance `ftol` of `scipy.optimize.minimize` (0.05 by default).

    starting_pars : array_like
        Numpy array of starting values for the minimization of parameters `pars_ff_fm` (`None` by default).

    n_parallel_jobs : int
        Number of jobs run in parallel (`None` by default).
    """
    if starting_alpha <= 0:
        print('alpha cannot be negative or zero; starting with alpha = 1')
        starting_alpha = 1
    if starting_beta <= 0:
        print('required beta > 0; starting with beta = 1')
        starting_beta = 1
    if starting_gamma <= 0:
        print('required gamma > 0; starting with gamma = 1')
        starting_gamma = 1

    if if_print_biblio: print_references(starting_alpha, starting_beta, starting_gamma, hasattr(data.properties, 'cycle_names'))

    class hyper_intermediate_class():
        def __init__(self):
            self.av_chi2 = []
            self.av_gradient = []
            self.log10_hyperpars = []

    global hyper_intermediate
    hyper_intermediate = hyper_intermediate_class()

    if type(random_states) is int:
        random_states = [i for i in range(random_states)]

    """ select training and validation set (several seeds) """

    valid_obs = {}
    valid_frames = {}

    # if which_set == 'valid_obs': if_all_frames = True  # validation set includes also training frames
    # here it is the same in both cases, since selecting only valid_obs and valid_frames

    for seed in random_states:
        out = split_dataset(data, random_state=seed, replica_infos=data.properties.infos)
        valid_obs[seed] = out[2]
        valid_frames[seed] = out[3]
        # here you could check to not have repeated choices

    """ derivatives """

    class derivatives_funs_class:
        def __init__(self, loss_function, gamma_function):
            # self.dloss_dpars = gradient_fun
            self.dloss_dpars = jax.grad(loss_function, argnums=0)
            self.d2loss_dpars2 = jax.hessian(loss_function, argnums=0)
            self.d2loss_dpars_dalpha = jax.jacfwd(self.dloss_dpars, argnums=3)
            self.d2loss_dpars_dbeta = jax.jacfwd(self.dloss_dpars, argnums=4)
            self.d2loss_dpars_dgamma = jax.jacfwd(self.dloss_dpars, argnums=5)

            # self.d2loss_dlambdas2 = jax.hessian(loss_function, argnums = 6)
            self.d2loss_dpars_dlambdas = jax.jacrev(self.dloss_dpars, argnums=6)
            self.dgamma_dlambdas = jax.grad(gamma_function, argnums=0)
            self.d2gamma_dlambdas_dalpha = jax.jacfwd(self.dgamma_dlambdas, argnums=4)
            self.d2gamma_dlambdas2 = jax.jacrev(self.dgamma_dlambdas, argnums=0)

            self.dchi2_dpars = jax.grad(compute_chi2_tot, argnums=0)
            self.dchi2_dlambdas = jax.grad(compute_chi2_tot, argnums=1)

    derivatives_funs = derivatives_funs_class(loss_function, gamma_function)

    log10_hyperpars0 = []
    map_hyperpars = []

    if starting_alpha <= 0:
        print("error: starting alpha is <= zero! let's start with alpha = 1")
        starting_alpha = 1
    if starting_beta < 0:
        print("error: starting beta is negative! let's start with beta = 1")
        starting_beta = 1
    if starting_gamma < 0:
        print("error: starting gamma is negative! let's start with gamma = 1")
        starting_gamma = 1

    if not np.isinf(starting_alpha):
        log10_hyperpars0.append(np.log10(starting_alpha))
        map_hyperpars.append('alpha')
    if not np.isinf(starting_beta):
        log10_hyperpars0.append(np.log10(starting_beta))
        map_hyperpars.append('beta')
    if not np.isinf(starting_gamma):
        log10_hyperpars0.append(np.log10(starting_gamma))
        map_hyperpars.append('gamma')

    # minimize
    args = (
        map_hyperpars, data, regularization, valid_obs, valid_frames, which_set, derivatives_funs,
        starting_pars, n_parallel_jobs)

    # just to check:
    # out = hyper_function(log10_hyperpars0, map_hyperpars, data, regularization, valid_obs, valid_frames, which_set,
    # derivatives_funs, starting_pars)

    """ see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html """
    """ with L-BFGS-B you can use ftol (stop when small variation of hyperparameters), useful for rough functions """
    if ftol is None:
        method = 'BFGS'
        options = {'gtol': gtol, 'maxiter': 20}
    else:
        method = 'L-BFGS-B'
        options = {'gtol': gtol, 'maxiter': 20, 'ftol': ftol}

    hyper_mini = minimize(hyper_function, log10_hyperpars0, args=args, method=method, jac=True, options=options)

    hyper_intermediate.av_chi2 = np.array(hyper_intermediate.av_chi2)
    hyper_intermediate.av_gradient = np.array(hyper_intermediate.av_gradient)
    hyper_intermediate.log10_hyperpars = np.array(hyper_intermediate.log10_hyperpars)
    hyper_mini['intermediate'] = hyper_intermediate

    return hyper_mini

