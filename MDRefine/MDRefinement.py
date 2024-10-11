"""
Main tool: MDRefinement.
It refines MD-generated trajectories with customizable refinement and reweights the trajectories.
"""

import os
import pandas
import datetime

# numpy is required for loadtxt and for gradient arrays with L-BFGS-B minimization (rather than jax.numpy)
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)

from .data_loading import load_data
from .hyperminimizer import hyper_minimizer
from .loss_and_minimizer import minimizer

# %% D8. MDRefinement


def MDRefinement(
        infos: dict, *, regularization: dict = None, stride: int = 1,
        starting_alpha: float = np.inf, starting_beta: float = np.inf, starting_gamma: float = np.inf,
        random_states = 5, which_set: str = 'validation', gtol: float = 0.5, ftol: float = 0.05,
        results_folder_name: str = 'results', n_parallel_jobs: int = None):
    """
    This is the main tool of the package: it loads data, searches for the optimal hyperparameters and minimizes the loss function on the whole data set
    by using the opimized hyperparameters. The output variables are then saved in a folder; they include `input` values, `min_lambdas` (optimal lambda coefficients for Ensemble Refinement, when performed),
    `result`, `hyper_search` (steps in the search for optimal hyperparameters) (`.csv` files) and the `.npy` arrays with the new weights determined in the refinement.

    Parameters
    ----------
    
    infos: dict
        A dictionary of information used to load data with `load_data` (see in the Examples directory).
    
    regularization: dict
        A dictionary which can include two keys: `force_field_reg` and `forward_model_reg`, to specify the regularizations to the force-field correction and the forward model, respectively;
        the first key is either a string (among `plain l2`, `constraint 1`, `constraint 2`, `KL divergence`) or a user-defined
        function which takes as input `pars_ff` and returns the regularization term to be multiplied by the hyperparameter `beta`;
        the second key is a user-defined function which takes as input `pars_fm` and `forward_coeffs_0` (current and refined forward-model coefficients) and
        returns the regularization term to be multiplied by the hyperparameter `gamma`.
    
    stride: int
        The stride of the frames used to load data employed in search for optimal hyperparameters
        (in order to reduce the computational cost, at the price of a lower representativeness of the ensembles).
    
    starting_alpha, starting_beta, starting_gamma: floats
        Starting values of the hyperparameters (`np.inf` by default, namely no refinement in that direction).
    
    random_states: int or list of integers
        Random states (i.e., seeds) used to split the data set in cross validation (if integer, then `random_states = np.arange(random_states)`.
    
    which_set: str
        String chosen among `'training'`, `'validation'` or `'test'`, which specifies how to determine optimal hyperparameters:
        if minimizing the (average) chi2 on the training set for `'training'`, on training observables and test frames for `'validation'`,
        on test observables for `'test'`.
    
    gtol: float
        Tolerance `gtol` (on the gradient) of scipy.optimize.minimize (0.5 by default).

    ftol: float
        Tolerance `ftol` of scipy.optimize.minimize (0.05 by default).

    results_folder_name: str
        String for the prefix of the folder where to save results; the complete folder name is `results_folder_name + '_' + time` where `time` is the current time
        when the algorithm has finished, in order to uniquely identify the folder with the results.
    
    n_parallel_jobs: int
        How many jobs are run in parallel (`None` by default).
    """
    data = load_data(infos, stride=stride)

    print('\nsearch for optimal hyperparameters ...')

    mini = hyper_minimizer(
        data, starting_alpha, starting_beta, starting_gamma, regularization,
        random_states, infos, which_set, gtol, ftol, n_parallel_jobs=n_parallel_jobs)

    optimal_log10_hyperpars = mini.x

    optimal_hyperpars = {}
    i = 0
    s = ''
    if not np.isinf(starting_alpha):
        alpha = 10**optimal_log10_hyperpars[i]
        optimal_hyperpars['alpha'] = alpha
        s = s + 'alpha: ' + str(alpha) + ' '
        i += 1
    else:
        alpha = starting_alpha
    if not np.isinf(starting_beta):
        beta = 10**optimal_log10_hyperpars[i]
        optimal_hyperpars['beta'] = beta
        s = s + 'beta: ' + str(beta) + ' '
        i += 1
    else:
        beta = starting_beta
    if not np.isinf(starting_gamma):
        gamma = 10**optimal_log10_hyperpars[i]
        optimal_hyperpars['gamma'] = gamma
        s = s + 'gamma: ' + str(gamma)
        # i += 1
    else:
        gamma = starting_gamma

    print('\noptimal hyperparameters: ' + s)
    print('\nrefinement with optimal hyperparameters...')  # on the full data set')

    # # for the minimization with optimal hyper-parameters use full data set
    # data = load_data(infos)

    Result = minimizer(data, regularization=regularization, alpha=alpha, beta=beta, gamma=gamma)
    Result.optimal_hyperpars = optimal_hyperpars
    Result.hyper_minimization = mini

    print('\ndone')

    """ save results in txt files """
    if not np.isinf(beta):
        coeff_names = infos['global']['names_ff_pars']
    else:
        coeff_names = []
    if not np.isinf(gamma):
        coeff_names = coeff_names + list(data._global_.forward_coeffs_0.keys())

    input_values = {
        'stride': stride, 'starting_alpha': starting_alpha, 'starting_beta': starting_beta,
        'starting_gamma': starting_gamma, 'random_states': random_states, 'which_set': which_set,
        'gtol': gtol, 'ftol': ftol}

    save_txt(input_values, Result, coeff_names, folder_name=results_folder_name)

    return Result


def unwrap_2dict(my_2dict):
    """
    Tool to unwrap a 2-layer dictionary `my_2dict` into list of values and list of keys.
    """

    res = []
    keys = []

    for key1, value1 in my_2dict.items():
        for key2, value2 in value1.items():

            key = key1 + ' ' + key2

            length = np.array(value2).shape[0]
            res.extend(list(value2))

            if length > 1:
                names = [key + ' ' + str(i) for i in range(length)]
            else:
                names = [key]

            keys.extend(names)

    return res, keys


def save_txt(input_values, Result, coeff_names, folder_name='Result'):
    """
    This is an internal tool of `MDRefinement` used to save `input_values` and output `Result` as `csv` and `npy` files in a folder whose name is
    `folder_name + '_' + date` where date is the current time when the computation ended (it uses `date_time`
    to generate unique file name, on the assumption of a single folder name at given time).

    Parameters
    ----------
    input_values : dict
        Dictionary with input values of the refinement, such as stride, starting values of the hyperparameters, random_states, which_set, tolerances (see `MDRefinement`).

    Result : class instance
        Class instance with the results of `minimizer` and the search for the optimal hyperparameters.

    coeff_names : list
        List with the names of the coefficients (force-field and forward-model corrections).

    folder_name : str
        String for the prefix of the folder name (by default, `'Result'`).
    """
    s = datetime.datetime.now()
    date = s.strftime('%Y_%m_%d_%H_%M_%S_%f')

    folder_name = folder_name + '_' + date

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    """0. save input values """
    temp = pandas.DataFrame(list(input_values.values()), index=list(input_values.keys()), columns=[date]).T
    temp.to_csv(folder_name + '/input')

    """ 1. save general results """

    # select information to be saved in txt files

    title = list(vars(Result).keys())

    remove_list = [
        'intermediates', 'abs_difference', 'av_g', 'logZ_new', 'weights_new', 'abs_difference_test',
        'av_g_test', 'logZ_new_test', 'weights_new_test', 'avg_new_obs', 'weights_P', 'logZ_P', 'weights_P_test',
        'logZ_P_test']

    if hasattr(Result, 'weights_new'):
        for name_sys in Result.weights_new.keys():
            np.save(folder_name + '/weights_new_%s' % name_sys, Result.weights_new[name_sys])
    if hasattr(Result, 'weights_P'):
        for name_sys in Result.weights_P.keys():
            np.save(folder_name + '/weights_ff_%s' % name_sys, Result.weights_P[name_sys])

    my_dict = {}
    for s in title:
        if s not in remove_list:
            if s == 'pars':
                for i, k in enumerate(coeff_names):
                    my_dict[k] = Result.pars[i]
            elif s == 'mini':
                my_dict['success'] = Result.mini.success
                my_dict['norm gradient'] = np.linalg.norm(Result.mini.jac)

            elif s == 'min_lambdas':
                flat_lambdas = unwrap_2dict(Result.min_lambdas)
                df = pandas.DataFrame(flat_lambdas[0], index=flat_lambdas[1], columns=[date]).T
                df.to_csv(folder_name + '/min_lambdas')

            elif s == 'minis':
                for name_sys in Result.minis.keys():
                    my_dict['ER success %s' % name_sys] = Result.minis[name_sys].success
            elif s == 'D_KL_alpha' or s == 'D_KL_alpha_test':
                for name_sys in vars(Result)[s].keys():
                    my_dict[s + '_' + name_sys] = vars(Result)[s][name_sys]
            elif s == 'chi2' or s == 'chi2_test' or s == 'chi2_new_obs':
                for name_sys in vars(Result)[s].keys():
                    my_dict[s + '_' + name_sys] = np.sum(np.array(list(vars(Result)[s][name_sys].values())))
            elif s == 'reg_ff' or s == 'reg_ff_test':
                if type(vars(Result)[s]) is dict:
                    for k in vars(Result)[s].keys():
                        my_dict[s + '_' + k] = vars(Result)[s][k]
                else:
                    my_dict[s] = vars(Result)[s]

            # optimization of hyper parameters
            elif s == 'optimal_hyperpars':
                for k in Result.optimal_hyperpars.keys():
                    my_dict['optimal ' + k] = Result.optimal_hyperpars[k]
            elif s == 'hyper_minimization':
                my_dict['hyper_mini success'] = Result.hyper_minimization.success

                inter = vars(Result.hyper_minimization['intermediate'])

                for i, name in enumerate(Result.optimal_hyperpars.keys()):
                    inter['av_gradient ' + name] = inter['av_gradient'][:, i]
                    inter['log10_hyperpars ' + name] = inter['log10_hyperpars'][:, i]
                del inter['av_gradient'], inter['log10_hyperpars']

                df = pandas.DataFrame(inter)
                df.to_csv(folder_name + '/hyper_search')

            else:
                my_dict[s] = vars(Result)[s]

    title = list(my_dict.keys())
    values = list(my_dict.values())

    df = pandas.DataFrame(values, index=title, columns=[date]).T
    df.to_csv(folder_name + '/result')

    return