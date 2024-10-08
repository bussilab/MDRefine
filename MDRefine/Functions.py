"""
Tools to perform reweighting using several refinements.
It also includes optimization of the hyperparameters through minimization of the chi2 on the test set.
"""

import os
import copy
import time
import pandas
import numpy.random as random
from scipy.optimize import minimize
from joblib import Parallel, delayed
import datetime

# numpy is required for loadtxt and for gradient arrays with L-BFGS-B minimization (rather than jax.numpy)
import numpy
import jax
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)

# %% A. Functions to load data:
# %% A1. check_and_skip

def check_and_skip(data, *, stride=1):
    """
    This function is an internal tool used in `load_data` to modify input `data`:

    - weights are normalized;

    - it appends observables computed through forward models (if any) to `data.g`;
    
    - if `hasattr(data, 'selected_obs')`: it removes non-selected observables from `data.forward_qs`;
    
    - select frames with given `stride`;
    
    - count n. experiments and n. frames (`data[name_sys].n_frames` and `data[name_sys].n_experiments`)
    and check corresponding matching.
    """

    output_data = {}
    output_data['global'] = data['global']

    system_names = data['global'].system_names

    for name_sys in system_names:

        my_data = data[name_sys]

        """ 1. compute observables from data.forward_qs through forward model and include them in data.g """

        if hasattr(my_data, 'forward_model') and (my_data.forward_model is not None):
            if not hasattr(my_data, 'g'):
                my_data.g = {}

            if hasattr(my_data, 'selected_obs'):
                for type_name in my_data.forward_qs.keys():
                    my_data.forward_qs[type_name] = my_data.forward_qs[type_name]  # [:,data.selected_obs[name][type_name]]

            if hasattr(my_data, 'selected_obs'):
                selected_obs = my_data.selected_obs
            else:
                selected_obs = None

            out = my_data.forward_model(np.array(data['global'].forward_coeffs_0), my_data.forward_qs, selected_obs)

            if type(out) is tuple:
                out = out[0]

            if not hasattr(my_data, 'g'):
                my_data.g = {}
            for name in out.keys():
                my_data.g[name] = out[name]

        """ 2. check data """

        b = 0

        if not hasattr(my_data, 'g'):

            if not hasattr(data[name_sys[:-3]], 'gexp_DDG'):
                print('error: missing MD data for system' % name_sys)
                b = 1
        if b == 1:
            return

        """ 3. count number of systems and number of experimental data; check: same number of frames """

        my_data.n_experiments = {}

        if hasattr(my_data, 'gexp'):
            my_data.n_experiments = {}
            for kind in my_data.gexp.keys():
                my_data.n_experiments[kind] = np.shape(my_data.gexp[kind])[0]

            """ check same number of observables as in data.gexp """

            if hasattr(my_data, 'g'):
                for kind in my_data.g.keys():
                    if my_data.ref[kind] == '><':
                        if not np.shape(my_data.gexp[kind+' LOWER'])[0] == np.shape(my_data.g[kind])[1]:
                            print('error: different number of observables for (system, kind) = (%s,%s)' % (name_sys, kind))
                        if not np.shape(my_data.gexp[kind+' UPPER'])[0] == np.shape(my_data.g[kind])[1]:
                            print('error: different number of observables for (system, kind) = (%s,%s)' % (name_sys, kind))
                    else:
                        if not np.shape(my_data.gexp[kind])[0] == np.shape(my_data.g[kind])[1]:
                            print('error: different number of observables for (system, kind) = (%s,%s)' % (name_sys, kind))

        """ check number of frames """

        n_frames = np.shape(my_data.weights)[0]

        if not (hasattr(my_data, 'g') or hasattr(my_data, 'forward_qs') or hasattr(data[name_sys[:-3]], 'gexp_DDG')):
            print('error: missing MD data')
        else:

            err_string = [
                'error: different number of frames for observable (system,kind) = (%s,%s)',
                'error: different number of frames for forward_qs (system,kind) = (%s,%s)',
                'error: different number of frames for force field terms of system %s']

            if hasattr(my_data, 'g'):
                for kind in my_data.g.keys():
                    assert np.shape(my_data.g[kind])[0] == n_frames, err_string[0] % (name_sys, kind)

            if hasattr(my_data, 'forward_qs'):
                for kind in my_data.forward_qs.keys():
                    assert np.shape(my_data.forward_qs[kind])[0] == n_frames, err_string[1] % (name_sys, kind)

        if hasattr(my_data, 'f'):
            assert len(my_data.f) == n_frames, err_string[2] % name_sys

        """ 4. do you want to skip frames? select stride (stride = 1 by default) """

        if not stride == 1:
            if hasattr(my_data, 'f'):
                my_data.f = my_data.f[::stride]
            my_data.weights = my_data.weights[::stride]
            my_data.weights = my_data.weights/np.sum(my_data.weights)

            if hasattr(my_data, 'g'):
                for name in my_data.g.keys():
                    my_data.g[name] = my_data.g[name][::stride]

            if hasattr(my_data, 'forward_qs'):
                for name in my_data.forward_qs.keys():
                    my_data.forward_qs[name] = my_data.forward_qs[name][::stride]

        """ 5. count number of frames """

        my_data.n_frames = np.shape(my_data.weights)[0]

        output_data[name_sys] = my_data
        del my_data

    if hasattr(data['global'], 'cycle_names'):
        for name in data['global'].cycle_names:
            output_data[name] = data[name]

    return output_data

# %% A2. load_data

class data_global_class:
    """Global data, common to all the investigated molecular systems.
    
    Parameters
    -----------

    info_global: dict
        Dictionary with global information:
        `info_global['system_names']` with list of names of the molecular systems;
        `info_global['cycle_names']` with list of names of the thermodynamic cycles;
        `info_global['forward_coeffs']` with string for the file name of forward coefficients;
        `info_global['names_ff_pars']` with list of names of the force-field correction coefficients.

    path_directory: str
        String with the path of the directory with input files.

    ----------------
    Instance variables:
    ----------------
    system_names : list
        List of names of the investigated molecular systems.
    
    forward_coeffs_0 : list
        List of the forward-model coefficients.
    
    names_ff_pars : list
        List of names of the force-field correction parameters.

    cycle_names : list
        List of names of the investigated thermodynamic cycles.
    """
    def __init__(self, info_global, path_directory):

        self.system_names = info_global['system_names']

        if 'forward_coeffs' in info_global.keys():
            temp = pandas.read_csv(path_directory + info_global['forward_coeffs'], header=None)
            temp.index = temp.iloc[:, 0]
            self.forward_coeffs_0 = temp.iloc[:, 1]

            # temp = pandas.read_csv(path_directory+'%s' % info_global['forward_coeffs'], index_col=0)
            # if temp.shape[0] == 1:
            #     self.forward_coeffs_0 = temp.iloc[:, 0]
            # else:
            #     self.forward_coeffs_0 = temp.squeeze()

        if 'names_ff_pars' in info_global.keys():
            self.names_ff_pars = info_global['names_ff_pars']
        
        if 'cycle_names' in info_global.keys():
            self.cycle_names = info_global['cycle_names']

    def tot_n_experiments(self, data):
        """This method computes the total n. of experiments."""
        
        tot = 0

        for k in self.system_names:
            for item in data[k].n_experiments.values():
                tot += item
        return tot


class data_class:
    """
    Data object of a molecular system.

    Parameters
    ----------------
    info: dict
        Dictionary for the information about the data of `name_sys` molecular system in `path_directory`. 

    path_directory: str
        String for the path of the directory with data of the molecular system `name_sys`.

    name_sys: str
        Name of the molecular system taken into account.
    
    --------------
    Instance variables:
    --------------
    temperature : float
        Value for the temperature at which the trajectory is simulated.
    
    gexp : dict
        Dictionary of Numpy 2-dimensional arrays (N x 2); `gexp[j,0]` is the experimental value of the j-th observable, `gexp[j,1]` is the corresponding uncertainty;
        the size N depends on the type of observable.
    
    names : dict
        Dictionary of Numpy 1-dimensional arrays of length N with the names of the observables of each type.
    
    ref : dict
        Dictionary of strings with signs `'=', '>', '<', '><' used to define the chi2 to compute,
        depending on the observable type.
    
    g : dict
        Dictionary of Numpy 2-dimensional arrays (M x N), where `g[name][i,j]` is the j-th observable of that type computed in the i-th frame.
    
    forward_qs : dict
        Dictionary of Numpy 2-dimensional arrays (M x N) with the quantities required for the forward model.
    
    forward_model: function
        Function for the forward model, whose input variables are the forward-model coefficients `fm_coeffs` and the `forward_qs` dictionary;
        a third optional argument is the `selected_obs` (dictionary with indices of selected observables).
    
    weights: array_like
        Numpy 1-dimensional array of length M with the weights (not required to be normalized).
    
    f: dict
        Numpy 2-dimensional array (M x P) of terms required to compute the force-field correction,
        where P is the n. of parameters `pars` and M is the n. of frames.
    
    ff_correction: function
        Function for the force-field correction, whose input variables are the force-field correction parameters `pars` and the `f` array (sorted consistently with each other).
    """
    def __init__(self, info, path_directory, name_sys):

        # 0. temperature

        if 'temperature' in info.keys():
            self.temperature = info['temperature']
            """`float` value for the temperature"""
        else:
            self.temperature = 1.0

        # 1. gexp (experimental values) and names of the observables

        if 'g_exp' in info.keys():

            self.gexp = {}
            """dictionary of `numpy.ndarray` containing gexp values and uncertainties"""
            self.names = {}
            """dictionary of `numpy.ndarray` containing names of experimental observables"""
            self.ref = {}  # if data.gexp are boundary or puntual values
            """dictionary of `numpy.ndarray` containing references"""

            if info['g_exp'] is None:
                if info['DDGs']['if_DDGs'] is False:
                    print('error, some experimental data is missing')
            else:
                if info['g_exp'] == []:
                    info['g_exp'] = [f[:-4] for f in os.listdir(path_directory+'%s/g_exp' % name_sys)]

                for name in info['g_exp']:
                    if type(name) is tuple:
                        if len(name) == 5:
                            for i in range(2):
                                if name[2*i+2] == '>':
                                    s = ' LOWER'
                                elif name[2*i+2] == '<':
                                    s = ' UPPER'
                                else:
                                    print('error in the sign of gexp')
                                    return

                                if os.path.isfile(path_directory+'%s/g_exp/%s%s.npy' % (name_sys, name[0], name[2*i+1])):
                                    self.gexp[name[0]+s] = np.load(
                                        path_directory+'%s/g_exp/%s%s.npy' % (name_sys, name[0], name[2*i+1]))
                                elif os.path.isfile(path_directory+'%s/g_exp/%s%s' % (name_sys, name[0], name[2*i+1])):
                                    self.gexp[name[0]+s] = numpy.loadtxt(
                                        path_directory+'%s/g_exp/%s%s' % (name_sys, name[0], name[2*i+1]))

                            self.ref[name[0]] = '><'

                        elif name[1] == '=' or name[1] == '>' or name[1] == '<':
                            if os.path.isfile(path_directory+'%s/g_exp/%s.npy' % (name_sys, name[0])):
                                self.gexp[name[0]] = np.load(path_directory+'%s/g_exp/%s.npy' % (name_sys, name[0]))
                            elif os.path.isfile(path_directory+'%s/g_exp/%s' % (name_sys, name[0])):
                                self.gexp[name[0]] = numpy.loadtxt(path_directory+'%s/g_exp/%s' % (name_sys, name[0]))
                            self.ref[name[0]] = name[1]

                        else:
                            print('error on specified sign of gexp')
                            return

                    else:
                        if os.path.isfile(path_directory+'%s/g_exp/%s.npy' % (name_sys, name)):
                            self.gexp[name] = np.load(path_directory+'%s/g_exp/%s.npy' % (name_sys, name))
                        elif os.path.isfile(path_directory+'%s/g_exp/%s' % (name_sys, name)):
                            self.gexp[name] = numpy.loadtxt(path_directory+'%s/g_exp/%s' % (name_sys, name))
                        self.ref[name] = '='

                    if type(name) is tuple:
                        name = name[0]
                    if os.path.isfile(path_directory+'%s/names/%s.npy' % (name_sys, name)):
                        self.names[name] = np.load(path_directory+'%s/names/%s.npy' % (name_sys, name))
                    elif os.path.isfile(path_directory+'%s/names/%s' % (name_sys, name)):
                        self.names[name] = numpy.loadtxt(path_directory+'%s/names/%s' % (name_sys, name))

        # 2. g (observables)

        if 'obs' in info.keys():

            self.g = {}

            if info['obs'] is not None:
                if info['obs'] == []:
                    info['obs'] = [f[:-4] for f in os.listdir(path_directory+'%s/observables' % name_sys)]
                for name in info['obs']:
                    if os.path.isfile(path_directory+'%s/observables/%s.npy' % (name_sys, name)):
                        self.g[name] = np.load(path_directory+'%s/observables/%s.npy' % (name_sys, name), mmap_mode='r')
                    elif os.path.isfile(path_directory+'%s/observables/%s' % (name_sys, name)):
                        self.g[name] = numpy.loadtxt(path_directory+'%s/observables/%s' % (name_sys, name))

        # 3. forward_qs (quantities for the forward model) and forward_model

        if 'forward_qs' in info.keys():

            # in this way, you can define forward model either with or without selected_obs (c)
            def my_forward_model(a, b, c=None):
                try:
                    out = info['forward_model'](a, b, c)
                except:
                    assert c is None, 'you have selected_obs but the forward model is not suitably defined!'
                    out = info['forward_model'](a, b)
                return out

            self.forward_model = my_forward_model  # info['forward_model']

            self.forward_qs = {}

            for name in info['forward_qs']:
                if info['forward_qs'] is not None:
                    if info['forward_qs'] == []:
                        info['forward_qs'] = [f[:-4] for f in os.listdir(path_directory+'%s/forward_qs' % name_sys)]
                    for name in info['forward_qs']:
                        if os.path.isfile(path_directory+'%s/forward_qs/%s.npy' % (name_sys, name)):
                            self.forward_qs[name] = np.load(
                                path_directory+'%s/forward_qs/%s.npy' % (name_sys, name), mmap_mode='r')
                        elif os.path.isfile(path_directory+'%s/forward_qs/%s' % (name_sys, name)):
                            self.forward_qs[name] = numpy.loadtxt(path_directory+'%s/forward_qs/%s' % (name_sys, name))

        # 4. weights (normalized)

        if os.path.isfile(path_directory+'%s/weights.npy' % name_sys):
            self.weights = np.load(path_directory+'%s/weights.npy' % name_sys)
        elif os.path.isfile(path_directory+'%s/weights' % name_sys):
            self.weights = numpy.loadtxt(path_directory+'%s/weights' % name_sys)
        else:
            if ('obs' in info.keys()) and not (info['obs'] is None):
                name = list(self.g.keys())[0]
                self.weights = np.ones(len(self.g[name]))
            elif ('forward_qs' in info.keys()) and not (info['forward_qs'] is None):
                name = list(self.forward_qs.keys())[0]
                self.weights = np.ones(len(self.forward_qs[info['forward_qs'][0]]))
            else:
                print('error: missing MD data for %s!' % name_sys)

        self.weights = self.weights/np.sum(self.weights)

        # 5. f (force field correction terms) and function

        if ('ff_correction' in info.keys()) and (info['ff_correction'] is not None):

            if info['ff_correction'] == 'linear':
                self.ff_correction = lambda pars, f: np.matmul(f, pars)
            else:
                self.ff_correction = info['ff_correction']

            ff_path = path_directory + '%s/ff_terms' % name_sys
            self.f = np.load(ff_path + '.npy')


class data_cycle_class:
    """
    Data object of a thermodynamic cycle.
    
    Parameters
    -------------------
    cycle_name : str
        String with the name of the thermodynamic cycle taken into account.
    
    DDGs_exp : pandas.DataFrame
        Pandas.DataFrame with the experimental values and uncertainties of Delta Delta G in labelled thermodynamic cycles.

    info: dict
        Dictionary for the information about the temperature of `cycle_name` thermodynamic cycle. 

    -------------------
    Instance variables:
    -------------------
    gexp_DDG : list
        List of two elements: the experimental value and uncertainty of the Delta Delta G.
    
    temperature : float
        Value of temperature.
    """
    def __init__(self, cycle_name, DDGs_exp, info):

        self.gexp_DDG = [DDGs_exp.loc[:, cycle_name].iloc[0], DDGs_exp.loc[:, cycle_name].iloc[1]]

        if 'temperature' in info.keys():
            self.temperature = info['temperature']
            """Temperature."""
        else:
            self.temperature = 1.0
            """Temperature"""

def load_data(infos, *, stride=1):
    """
    This tool loads data from specified directory as indicated by the user in `infos`
    to a dictionary `data` of classes, which includes `data['global']` (global properties) and `data[system_name]`;
    for alchemical calculations, there is also `data[cycle_name]`.
    """

    print('loading data from directory...')

    data = {}

    system_names = infos['global']['system_names']

    path_directory = infos['global']['path_directory'] + '/'
    data['global'] = data_global_class(infos['global'], path_directory)

    """ then, separately for each system: """

    for name_sys in system_names:
        print('loading ', name_sys)
        if name_sys in infos.keys():
            info = {**infos[name_sys], **infos['global']}
        else:
            info = infos['global']
        data[name_sys] = data_class(info, path_directory, name_sys)

    # quantities from alchemical calculations

    if 'cycle_names' in infos['global'].keys():

        # data['global'].cycle_names = infos['global']['cycle_names']

        logZs = pandas.read_csv(path_directory + 'alchemical/logZs', index_col=0, header=None)

        for name in infos['global']['cycle_names']:
            for s in ['MD', 'MS', 'AD', 'AS']:
                key = name + '_' + s
                if key in logZs.index:
                    data[key].logZ = logZs.loc[key][1]
                else:
                    data[key].logZ = 0.0

        DDGs_exp = pandas.read_csv(path_directory + 'alchemical/DDGs', index_col=0)

        for name in infos['global']['cycle_names']:
            if name in infos.keys():
                info = {**infos[name], **infos['global']}
            else:
                info = infos['global']

            data[name] = data_cycle_class(name, DDGs_exp, info)

    # check and skip frames with stride

    data = check_and_skip(data, stride=stride)

    # def tot_n_experiments(data):
    #     tot = 0
    #     for k in system_names:
    #         for item in data[k].n_experiments.values():
    #             tot += item
    #     return tot

    # data['global'].system_names = system_names
    # data['global'].tot_n_experiments = tot_n_experiments

    # if hasattr(data['global'], 'ff_correction') and (data['global'].ff_correction == 'linear'):
    #     list_names_ff_pars = []
    #     for k in data['global'].system_names:
    #         if hasattr(data[k], 'f'):
    #             [list_names_ff_pars.append(x) for x in data[k].f.keys() if x not in list_names_ff_pars]
    #     data['global'].names_ff_pars = list_names_ff_pars

    # elif 'names_ff_pars' in infos['global'].keys():
    #     data['global'].names_ff_pars = infos['global']['names_ff_pars']

    print('done')

    return data

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
    -----------
    
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
    ----------------
    gexp, g : dicts
        Dictionaries corresponding to `data[name_sys].gexp` and `data[name_sys].g`.
        
    weights : array-like
        Numpy 1-dimensional array, by default `None` (namely, equal weight for each frame).
    ----------------

    Output parameters:
    ------------------
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
    --------------
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
    -----------
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
    --------------

    Output variables:
    --------------
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
    ----------------
    data : dict
        Object `data`; here, `data['global']` has the attribute `cycle_names` (list of names of the thermodynamic cycles);
        `for s in data['global'].cycle_names`: `data[s]` has attributes `temperature` (of the cycle) and `gexp_DDG`;
        `for s in data['global'].cycle_names`: `data[s+k]` `for k in '_MD', '_MS', '_AD', '_AS'` has attributes `temperature` (of the system) and `logZ`.
        
    logZ_P : dict
        Dictionary for logarithm of the partition function $Z_P$, namely, average value of $e^{-V_\phi(x)/temperature}$ over the original ensemble.
    ---------------
    Output variables:
    ---------------
    new_av_DG : dict
        Dictionary of reweighted averages of Delta G.

    chi2 : dict
        Dictionary of chi2 (one for each thermodynamic cycle).
    
    loss : float
        Total contribution to the loss function from free-energy differences Delta Delta G,
        given by 1/2 of the total chi2.
    """
    cycle_names = data['global'].cycle_names

    new_av_DG = {}
    chi2 = {}
    loss = 0

    for cycle_name in cycle_names:
        for s in ['_MD', '_MS', '_AD', '_AS']:
            if (cycle_name+s in logZ_P.keys()) and (not logZ_P[cycle_name+s] == 0):
                # correction only on MD
                new_av_DG[cycle_name+s] = -data[cycle_name+s].temperature*(logZ_P[cycle_name + s] + data[cycle_name + s].logZ)
            if cycle_name+s not in logZ_P:
                logZ_P[cycle_name+s] = 0

        DeltaDeltaG = -(
            logZ_P[cycle_name+'_MD'] + data[cycle_name+'_MD'].logZ
            - logZ_P[cycle_name+'_AD'] - data[cycle_name+'_AD'].logZ)

        DeltaDeltaG += logZ_P[cycle_name+'_MS'] + data[cycle_name+'_MS'].logZ
        - logZ_P[cycle_name+'_AS'] - data[cycle_name+'_AS'].logZ

        DeltaDeltaG = DeltaDeltaG*data[cycle_name].temperature

        chi2[cycle_name] = ((DeltaDeltaG - data[cycle_name].gexp_DDG[0])/data[cycle_name].gexp_DDG[1])**2
        loss += 1/2*chi2[cycle_name]

    return new_av_DG, chi2, loss

# %% C6. compute_details_ER


def compute_details_ER(weights_P, g, data, lambdas, alpha):
    """
    This is an internal tool of `loss_function` which computes explicitely the contribution to the loss function due to Ensemble Refinement
    (namely, 1/2 chi2 + alpha D_KL) and compare this value with -alpha*Gamma (they are equal in the minimum: check).
    It cycles over different systems. It acts after the minimization of the loss function inside `loss_function` (not for the minimization
    itself, since we exploit the Gamma function).

    Be careful to use either: normalized values for `lambdas` and `g` (if `hasattr(data,'normg_mean')`) or non-normalized ones
    (if `not hasattr(data,'normg_mean')`).
    
    Parameters
    ----------
    weights_P : dict
        Dictionary of Numpy arrays, namely, the weights on which Ensemble Refinement acts (those with force-field correction
        in the fully combined refinement).
        
    g : dict
        Dictionary of dictionaries, like for `data[name_sys].g`, corresponding to the observables (computed with updated forward-model coefficients).
    
    data : dict
        The original data object.
    
    lambdas : dict
        Dictionary of Numpy arrays, corresponding to the coefficients for Ensemble Refinement.
    
    alpha : float
        The alpha hyperparameter, for Ensemble Refinement.
    """
    if hasattr(data, 'normg_mean'):
        print('WARNING: you are using normalized observables!')

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

    system_names = data['global'].system_names

    for name_sys in system_names:

        flatten_g = np.hstack([g[name_sys][k] for k in data[name_sys].n_experiments.keys()])
        flatten_gexp = np.vstack([data[name_sys].gexp[k] for k in data[name_sys].n_experiments.keys()])
        correction = np.einsum('ij,j', flatten_g, lambdas[name_sys])

        out = compute_new_weights(weights_P[name_sys], correction)
        Details.weights_new[name_sys] = out[0]
        Details.logZ_new[name_sys] = out[1]

        out = compute_chi2(data[name_sys].ref, Details.weights_new[name_sys], g[name_sys], data[name_sys].gexp)
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
    -----------------
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

    system_names = data['global'].system_names

    if_fixed_lambdas = None  # to avoid error in Pylint
    if not np.isinf(alpha):
        if (fixed_lambdas is None):
            if_fixed_lambdas = False
            global lambdas
            if 'lambdas' not in globals():
                lambdas = np.zeros(data['global'].tot_n_experiments(data))
        else:
            if_fixed_lambdas = True
            lambdas = fixed_lambdas

    if not np.isinf(beta):
        names_ff_pars = data['global'].names_ff_pars
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
            if hasattr(data[name_sys], 'ff_correction'):
                correction_ff[name_sys] = data[name_sys].ff_correction(pars_ff, data[name_sys].f)
                weights_P[name_sys], logZ_P[name_sys] = compute_new_weights(
                    data[name_sys].weights, correction_ff[name_sys]/data[name_sys].temperature)

            else:  # if beta is not infinite, but there are systems without force-field corrections:
                weights_P[name_sys] = data[name_sys].weights
                logZ_P[name_sys] = 0
        else:
            weights_P[name_sys] = data[name_sys].weights
            logZ_P[name_sys] = 0

        """ 1b. if np.isinf(gamma): g is re-computed observables data.g through updated forward model
            (notice you also have some observables directly as data.g without updating of forward model);
            else: g is data.g (initial data.g[name_sys] if gamma == np.inf). """

        if np.isinf(gamma):
            if hasattr(data[name_sys], 'g'):
                g[name_sys] = copy.deepcopy(data[name_sys].g)
        else:
            if hasattr(data[name_sys], 'g'):
                g[name_sys] = copy.deepcopy(data[name_sys].g)
            else:
                g[name_sys] = {}

            if hasattr(data[name_sys], 'selected_obs'):
                selected_obs = data[name_sys].selected_obs
            else:
                selected_obs = None

            fm_observables = data[name_sys].forward_model(pars_fm, data[name_sys].forward_qs, selected_obs)

            for name in fm_observables.keys():

                g[name_sys][name] = fm_observables[name]
                if hasattr(data[name_sys], 'normg_mean'):
                    g[name_sys][name] = (g[name_sys][name]-data[name_sys].normg_mean[name])/data[name_sys].normg_std[name]

            del fm_observables

        if (np.isinf(gamma) and hasattr(data[name_sys], 'g')) or not np.isinf(gamma):
            for name in data[name_sys].ref.keys():
                if data[name_sys].ref[name] == '><':
                    g[name_sys][name+' LOWER'] = g[name_sys][name]
                    g[name_sys][name+' UPPER'] = g[name_sys][name]
                    del g[name_sys][name]

    """ 2. compute chi2 (if np.isinf(alpha)) or Gamma function (otherwise) """

    if np.isinf(alpha):

        av_g = {}
        chi2 = {}

        if hasattr(data['global'], 'cycle_names'):
            out = compute_DeltaDeltaG_terms(data, logZ_P)
            av_g = out[0]
            chi2 = out[1]
            loss += out[2]

        for name_sys in system_names:
            if hasattr(data[name_sys], 'g'):
                out = compute_chi2(data[name_sys].ref, weights_P[name_sys], g[name_sys], data[name_sys].gexp, True)
                av_g[name_sys] = out[0]
                chi2[name_sys] = out[1]
                loss += 1/2*out[3]

    else:

        my_dict = {}
        for k in system_names:
            my_dict[k] = data[k].n_experiments
        js = compute_js(my_dict)

        x0 = {}
        flatten_g = {}
        flatten_gexp = {}

        for i_sys, name_sys in enumerate(system_names):

            x0[name_sys] = np.array(lambdas[js[i_sys][0]:js[i_sys][-1]])
            flatten_g[name_sys] = np.hstack([g[name_sys][k] for k in data[name_sys].n_experiments.keys()])
            flatten_gexp[name_sys] = np.vstack([data[name_sys].gexp[k] for k in data[name_sys].n_experiments.keys()])

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
                    weights_P[name_sys], correction_ff[name_sys], data[name_sys].temperature, logZ_P[name_sys])
                loss += beta*reg_ff[name_sys]

    """ 4. add regularization of forward-model coefficients """
    if not np.isinf(gamma):
        reg_fm = regularization['forward_model_reg'](pars_fm, data['global'].forward_coeffs_0)
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
                Details.weights_new[name_sys] = data[name_sys].weights
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
    
    Input parameters:
    -------------------
    
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
    This tool deconvolves `lambdas` from Numpy array to dictionary of dictionaries (corresponding to `data[name_sys].g`);
    if `if_denormalize`, then `lambdas` has been computed with normalized data, so use `data[name_sys].normg_std` and `data[name_sys].normg_mean`
    in order to go back to corresponding lambdas for non-normalized data. The order of `lambdas` is the one described in `compute_js`.
    """
    dict_lambdas = {}

    ns = 0

    system_names = data['global'].system_names

    for name_sys in system_names:

        dict_lambdas[name_sys] = {}

        for name in data[name_sys].n_experiments.keys():
            dict_lambdas[name_sys][name] = lambdas[ns:(ns+data[name_sys].n_experiments[name])]
            ns += data[name_sys].n_experiments[name]

        if if_denormalize:
            assert hasattr(data[name_sys], 'normg_std'), 'Error: missing normalized std values!'
            for name in data[name_sys].ref.keys():
                if data[name_sys].ref[name] == '><':
                    # you can sum since one of the two is zero
                    dict_lambdas[name_sys][name] = (
                        dict_lambdas[name_sys][name+' LOWER']/data[name_sys].normg_std[name+' LOWER'])

                    dict_lambdas[name_sys][name] += (
                        dict_lambdas[name_sys][name+' UPPER']/data[name_sys].normg_std[name+' UPPER'])

                    del dict_lambdas[name_sys][name+' LOWER'], dict_lambdas[name_sys][name+' UPPER']
                else:
                    dict_lambdas[name_sys][name] = dict_lambdas[name_sys][name]/data[name_sys].normg_std[name]
        else:
            for name in data[name_sys].ref.keys():
                if data[name_sys].ref[name] == '><':
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

    Input values:
    --------------
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

    system_names = original_data['global'].system_names

    """ copy original_data and act only on the copy, preserving original_data """

    # data = copy.deepcopy(original_data) ## it does not work!

    data = {}
    for k1 in original_data.keys():
        class my_new_class:
            pass
        my_keys = [x for x in dir(original_data[k1]) if not x.startswith('__')]
        for k2 in my_keys:
            setattr(my_new_class, k2, copy.deepcopy(getattr(original_data[k1], k2)))
        data[k1] = my_new_class

    """ normalize observables """
    for name_sys in system_names:
        if hasattr(data[name_sys], 'g'):
            out = normalize_observables(data[name_sys].gexp, data[name_sys].g, data[name_sys].weights)
            data[name_sys].g = out[0]
            data[name_sys].gexp = out[1]
            data[name_sys].normg_mean = out[2]
            data[name_sys].normg_std = out[3]

    """ starting point for lambdas """
    if not np.isinf(alpha):

        global lambdas

        tot_n_exp = 0

        for name in system_names:
            for item in data[name].n_experiments.values():
                tot_n_exp += item

        lambdas = np.zeros(tot_n_exp)

        """ here you could duplicate lambdas for observables with both lower/upper limits """

    else:
        lambdas = None

    """ if needed, define boundaries for minimization over lambdas """

    if not np.isinf(alpha):

        my_list = []
        for k in data['global'].system_names:
            my_list = my_list + list(data[k].ref.values())

        if ('>' in my_list) or ('<' in my_list) or ('><' in my_list):

            bounds = {}

            for name_sys in data['global'].system_names:
                bounds[name_sys] = []
                for name_type in data[name_sys].n_experiments.keys():
                    if name_type in data[name_sys].ref.keys():
                        if data[name_sys].ref[name_type] == '=':
                            bounds[name_sys] = bounds[name_sys] + [(-np.inf, +np.inf)]*data[name_sys].g[name_type].shape[1]
                        elif data[name_sys].ref[name_type] == '<':
                            bounds[name_sys] = bounds[name_sys] + [(0, +np.inf)]*data[name_sys].g[name_type].shape[1]
                        elif data[name_sys].ref[name_type] == '>':
                            bounds[name_sys] = bounds[name_sys] + [(-np.inf, 0)]*data[name_sys].g[name_type].shape[1]
                    elif data[name_sys].ref[name_type[:-6]] == '><':
                        bounds[name_sys] = bounds[name_sys] + [(-np.inf, 0)]*data[name_sys].g[name_type].shape[1]
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
                names_ff_pars = data['global'].names_ff_pars
                pars_ff_fm_0 = pars_ff_fm_0 + list(np.zeros(len(names_ff_pars)))

            if not np.isinf(gamma):
                pars_ff_fm_0 = pars_ff_fm_0 + list(data['global'].forward_coeffs_0)
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
                np.hstack(Result.min_lambdas[name_sys][k] for k in data[name_sys].n_experiments.keys()))

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
    ---------------
    data : dict
        Dictionary for the `data` object.
    
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
    ----------
    Output variables:
    ----------
    data_train, data_test : dicts
        Dictionaries for training and test data; `data_test` includes:
        trained observables and non-trained (test) frames (where it is not specified `new`);
        non-trained (test) observables and non-trained/all (accordingly to `if_all_frames`) frames (where specified `new`).
    
    test_obs, test_frames : dicts
        Dictionaries for the observables and frames selected for the test set.
    """
    # PART 1: IF NONE, SELECT TEST OBSERVABLES AND TEST FRAMES

    system_names = data['global'].system_names
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

                n_frames_test = np.int16(np.round(test_frames_size*data[name_sys].n_frames))
                test_frames[name_sys] = np.sort(rng.choice(data[name_sys].n_frames, n_frames_test, replace=False))
                # except:
                # test_frames[name_sys] = random.choice(key, data[name_sys].n_frames,(n_frames_test[name_sys],),
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

            n = np.sum(np.array(list(data[name_sys].n_experiments.values())))
            vec = np.sort(rng.choice(n, np.int16(np.round(n*test_obs_size)), replace=False))
            # except: vec = np.sort(jax.random.choice(key, n, (np.int16(np.round(n*test_obs_size)),), replace = False))

            sum = 0
            for name_type in data[name_sys].n_experiments.keys():

                test_obs[name_sys][name_type] = vec[(vec >= sum) & (vec < sum + data[name_sys].n_experiments[name_type])] - sum
                n_obs_test[name_sys][name_type] = len(test_obs[name_sys][name_type])

                sum += data[name_sys].n_experiments[name_type]

        del sum, n, vec

    # PART 2: GIVEN test_frames and test_obs, RETURN data_test AND data_train
    # train, test1 ('non-trained' obs, all or 'non-used' frames), test2 ('trained' obs, 'non-used' frames)

    data_train = {}
    data_test = {}

    # global properties:

    data_train['global'] = data['global']
    data_test['global'] = data['global']

    # for over different systems:

    for name_sys in system_names:

        data_train[name_sys] = class_train(data[name_sys], test_frames[name_sys], test_obs[name_sys])
        data_test[name_sys] = class_test(
            data[name_sys], test_frames[name_sys], test_obs[name_sys], if_all_frames, data_train[name_sys])

    # """ if some type of observables are not included in test observables, delete them to avoid empty items """
    # for name_sys in system_names:
    #     for name_type in test_obs[name_sys].keys():
    #         if len(test_obs[name_sys][name_type]) == 0:
    #             del data_test[name_sys].gexp_new[name_type]
    #             if name_type in data_test[name_sys].g_new.keys():
    #                 del data_test[name_sys].g_new[name_type]
    #                 if if_all_frames: del data_test[name_sys].g_new_old[name_type]

    for s1 in test_obs.keys():
        my_list1 = []
        my_list2 = []

        for s2 in test_obs[s1].keys():
            if len(test_obs[s1][s2]) == 0:
                my_list1.append(s2)
            elif len(test_obs[s1][s2]) == data[s1].n_experiments[s2]:
                my_list2.append(s2)

        for s2 in my_list1:
            """ no test observables of this kind """
            del data_test[s1].gexp_new[s2], data_test[s1].g_new[s2], data_test[s1].n_experiments_new[s2]
            del data_test[s1].selected_obs_new[s2]  # , data_test[s1].names_new[s2]

        for s2 in my_list2:
            """ no training observables of this kind"""
            del data_test[s1].gexp[s2], data_test[s1].g[s2], data_test[s1].n_experiments[s2]
            del data_test[s1].selected_obs[s2]  # , data_test[s1].names[s2]
            del data_train[s1].gexp[s2], data_train[s1].g[s2], data_train[s1].n_experiments[s2]
            del data_train[s1].selected_obs[s2]  # , data_train[s1].names[s2]

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

    Input values:
    -------------
    
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

    system_names = data_test['global'].system_names
    names_ff_pars = []

    if not np.isinf(beta):
        names_ff_pars = data_test['global'].names_ff_pars

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
            if hasattr(data_test[name_sys], 'g_new'):
                g[name_sys] = copy.deepcopy(data_test[name_sys].g_new)
        else:
            if hasattr(data_test[name_sys], 'g_new'):
                g[name_sys] = copy.deepcopy(data_test[name_sys].g_new)
            else:
                g[name_sys] = {}

            if hasattr(data_test[name_sys], 'selected_obs'):
                selected_obs = data_test[name_sys].selected_obs_new
            else:
                selected_obs = None

            fm_observables = data_test[name_sys].forward_model(pars_fm, data_test[name_sys].forward_qs, selected_obs)

            for name in fm_observables.keys():

                g[name_sys][name] = fm_observables[name]
                if hasattr(data_test[name_sys], 'normg_mean'):
                    g[name_sys][name] = (
                        g[name_sys][name]-data_test[name_sys].normg_mean[name])/data_test[name_sys].normg_std[name]

            del fm_observables

    for name_sys in system_names:

        args = (data_test[name_sys].ref, Validation_values.weights_new[name_sys], g[name_sys], data_test[name_sys].gexp_new)
        out = compute_chi2(*args)

        Validation_values.avg_new_obs[name_sys] = out[0]

        if not hasattr(data_test, 'forward_qs_trained'):
            Validation_values.chi2_new_obs[name_sys] = out[1]

    # then, if you want to include also trained frames for validating observables:

    if hasattr(data_test, 'forward_qs_trained') and (data_train is not None):  # forward qs on trained frames

        Details_train = loss_function(pars_ff_fm, data_train, regularization, alpha, beta, gamma, lambdas, if_save=True)

        g = {}

        for name_sys in system_names:
            if np.isinf(gamma):
                if hasattr(data_test[name_sys], 'g_new_old'):
                    g[name_sys] = copy.deepcopy(data_test[name_sys].g_new_old)
            else:
                if hasattr(data_test[name_sys], 'g_new_old'):
                    g[name_sys] = copy.deepcopy(data_test[name_sys].g_new_old)
                else:
                    g[name_sys] = {}

                if hasattr(data_test[name_sys], 'selected_obs'):
                    selected_obs = data_test[name_sys].selected_obs
                else:
                    selected_obs = None

                fm_observables = data_test[name_sys].forward_model(pars_fm, data_test[name_sys].forward_qs, selected_obs)

                for name in fm_observables.keys():

                    g[name_sys][name] = fm_observables[name]
                    if hasattr(data_test[name_sys], 'normg_mean'):
                        g[name_sys][name] = (
                            g[name_sys][name]-data_test[name_sys].normg_mean[name])/data_test[name_sys].normg_std[name]

                del fm_observables

            Validation_values.chi2_new_obs[name_sys] = {}

            args = (data_test.ref[name_sys], Details_train.weights_new[name_sys], g[name_sys], data_test.gexp_new[name_sys])
            out = compute_chi2(*args)[0]

            log_fact_Z = data_test.logZ[name_sys] + Validation_values.logZ_new[name_sys]
            - Details_train.logZ_new[name_sys] - data_train[name_sys].logZ

            if hasattr(Validation_values, 'logZ_P'):
                log_fact_Z += Validation_values.logZ_P_test[name_sys] - Details_train.logZ_P[name_sys]

            for name_type in data_test.n_experiments[name_sys].keys():
                Validation_values.avg_new_obs[name_sys][name_type] = 1/(1+np.exp(log_fact_Z))*out[name_type]
                + 1/(1+np.exp(-log_fact_Z))*Validation_values.avg_new_obs[name_sys][name_type]

                Validation_values.chi2_new_obs[name_sys][name_type] = np.sum(((
                    Validation_values.avg_new_obs[name_sys][name_type]
                    - data_test.gexp_new[name_sys][name_type][:, 0])/data_test.gexp_new[name_sys][name_type][:, 1])**2)

    if which_return == 'chi2 test':
        tot_chi2 = 0
        for s1 in Validation_values.chi2_new_obs.keys():
            for item in Validation_values.chi2_new_obs[s1].values():
                tot_chi2 += item
        return tot_chi2
    return Validation_values

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

    Input values:
    --------------
    
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
    system_names = data['global'].system_names

    if np.isposinf(log10_beta) and np.isposinf(log10_gamma) and not np.isinf(log10_alpha):

        alpha = np.float64(10**log10_alpha)

        data_n_experiments = {}
        for k in system_names:
            data_n_experiments[k] = data[k].n_experiments
        js = compute_js(data_n_experiments)

        class derivatives:
            pass

        derivatives.dlambdas_dlogalpha = []

        for i_sys, name_sys in enumerate(system_names):

            my_lambdas = lambdas[js[i_sys][0]:js[i_sys][-1]]
            # indices = np.nonzero(my_lambdas)[0]

            refs = []
            for name in data[name_sys].n_experiments.keys():
                refs.extend(data[name_sys].ref[name]*data[name_sys].n_experiments[name])

            # indices of lambdas NOT on constraints
            indices = np.array([k for k in range(len(my_lambdas)) if ((not my_lambdas[k] == 0) or (refs[k] == '='))])

            if len(indices) == 0:
                print('all lambdas of system %s are on boundaries!' % name_sys)

            else:

                my_lambdas = my_lambdas[indices]

                g = np.hstack([data[name_sys].g[k] for k in data[name_sys].n_experiments.keys()])[:, indices]
                gexp = np.vstack([data[name_sys].gexp[k] for k in data[name_sys].n_experiments.keys()])[indices]

                my_args = (my_lambdas, g, gexp, data[name_sys].weights, alpha)
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
                data_n_experiments[k] = data[k].n_experiments
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

                names_ff_pars = data['global'].names_ff_pars
                pars_ff = pars_ff_fm[:len(names_ff_pars)]

                correction_ff = {}
                weights_P = {}
                logZ_P = {}

                for name in system_names:
                    if hasattr(data[name], 'ff_correction'):
                        correction_ff[name] = data[name].ff_correction(pars_ff, data[name].f)
                        correction_ff[name] = correction_ff[name]/data[name].temperature
                        weights_P[name], logZ_P[name] = compute_new_weights(data[name].weights, correction_ff[name])

                    else:  # if beta is not infinite, but there are systems without force-field corrections:
                        weights_P[name] = data[name].weights
                        logZ_P[name] = 0
            else:
                weights_P = {}
                for name in system_names:
                    weights_P[name] = data[name].weights

            """ compute forward quantities through (new) forward coefficients theta"""

            pars_fm = pars_ff_fm[len(names_ff_pars):]

            g = {}

            if np.isposinf(gamma):

                for name in system_names:
                    if hasattr(data[name], 'g'):
                        g[name] = copy.deepcopy(data[name].g)
            else:

                for name_sys in system_names:
                    if hasattr(data[name_sys], 'g'):
                        g[name_sys] = copy.deepcopy(data[name_sys].g)
                    else:
                        g[name_sys] = {}

                    if hasattr(data[name_sys], 'selected_obs'):
                        selected_obs = data[name_sys].selected_obs
                    else:
                        selected_obs = None

                    fm_observables = data[name_sys].forward_model(pars_fm, data[name_sys].forward_qs, selected_obs)

                    for name in fm_observables.keys():
                        g[name_sys][name] = fm_observables[name]

                    del fm_observables

            """ Compute derivatives and Hessian. """

            for i_sys, name_sys in enumerate(system_names):

                my_lambdas = lambdas[js[i_sys][0]:js[i_sys][-1]]

                """ use indices to select lambdas NOT on constraints """
                refs = []
                for name in data[name_sys].n_experiments.keys():
                    refs.extend(data[name_sys].ref[name]*data[name_sys].n_experiments[name])

                # indices of lambdas NOT on constraints
                indices = np.array([k for k in range(len(my_lambdas)) if ((not my_lambdas[k] == 0) or (refs[k] == '='))])

                if len(indices) == 0:
                    print('all lambdas of system %s are on boundaries!' % name_sys)

                else:

                    my_lambdas = my_lambdas[indices]

                    my_g = np.hstack([g[name_sys][k] for k in data[name_sys].n_experiments])[:, indices]
                    my_gexp = np.vstack([data[name_sys].gexp[k] for k in data[name_sys].n_experiments])[indices]

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

def compute_chi2_tot(pars_ff_fm, lambdas, data, regularization, alpha, beta, gamma, which_set):
    """
    This function is an internal tool used in `compute_hypergradient` and `hyper_minimizer`
    to compute the total chi2 (float variable) for the training or test data set and its derivatives
    (with respect to `pars_ff_fm` and `lambdas`). The choice of the data set is indicated by `which_set`
    (`which_set = 'training'` for chi2 on the training set, `'validation'` for chi2 on training observables and test frames,
    `'test'` for chi2 on test observables and test frames, through validation function).

    Input values:
    ---------------
    
    pars_ff_fm, lambdas: array_like
        Numpy arrays for (force-field + forward-model) parameters and lambdas parameters, respectively.
    
    data: dict
        Dictionary of data set object.
    
    regularization: dict
        Specified regularizations of force-field and forward-model corrections (see in `MDRefinement`).
    
    alpha, beta, gamma: float
        Values of the hyperparameters.
    
    which_set: str
        String variable, chosen among `'training'`, `'validation'` or `'test'` as explained above.
    """
    if which_set == 'training' or which_set == 'validation':
        tot_chi2 = 0

        Details = loss_function(pars_ff_fm, data, regularization, alpha, beta, gamma, fixed_lambdas=lambdas, if_save=True)

        for s1 in Details.chi2.keys():
            for item in Details.chi2[s1].values():
                tot_chi2 += item

    elif which_set == 'test':

        tot_chi2 = validation(
            pars_ff_fm, lambdas, data, regularization=regularization, alpha=alpha, beta=beta, gamma=gamma,
            which_return='chi2 test')

    return tot_chi2

# %% D3. put_together

def put_together(dchi2_dpars, dchi2_dlambdas, derivatives):
    """
    This is an internal tool of `compute_hypergradient` which applies the chain rule in order to get the derivatives of chi2 w.r.t hyperparameters from
    derivatives of chi2 w.r.t. parameters and derivatives of parameters w.r.t. hyperparameters.

    Parameters
    ---------------
    dchi2_dpars: array-like
        Numpy 1-dimensional array with derivatives of chi2 w.r.t. `pars_ff_fm` (force-field and forward-model parameters).
    
    dchi2_dlambdas: array-like
        Numpy 1-dimensional array with derivatives of chi2 w.r.t. `lambdas` (same order of `lambdas` in `dchi2_dlambdas` and in `derivatives`).
    
    derivatives: class instance
        Class instance with derivatives of `pars_ff_fm` and `lambdas` w.r.t. hyperparameters (determined in `compute_hyperderivatives`).

    ---------------
    Output variable: class instance whose attributes can include `dchi2_dlogalpha`, `dchi2_dlogbeta`, `dchi2_dloggamma`,
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
        which_set, data_test, derivatives_funs):
    """
    This is an internal tool of `mini_and_chi2_and_grad`, which employs previously defined functions (`compute_hyperderivatives`, `compute_chi2_tot`,
    `put_together`) to return selected chi2 and its gradient w.r.t hyperparameters.

    Input values:
    ---------------
    
    pars_ff_fm: array_like
        Numpy array of (force-field and forward-model) parameters.
    
    lambdas: dict
        Dictionary of dictionaries with lambda coefficients (corresponding to Ensemble Refinement).
    
    log10_alpha, log10_beta, log10_gamma: floats
        Logarithms (in base 10) of the hyperparameters alpha, beta, gamma.
    
    data_train: dict
        The training data set object, which is anyway required to compute the derivatives of parameters w.r.t. hyper-parameters.
    
    regularization: dict
        Specified regularizations (see in `MDRefinement`).
    
    which_set: str
        String indicating which set defines the chi2 to minimize in order to get the optimal hyperparameters (see in `compute_chi2_tot`).
    
    data_test: dict
        The test data set object, which is required to compute the chi2 on the test set (when `which_set == 'validation' or 'test'`;
        otherwise, if `which_set = 'training'`, it is useless, so it can be set to `None`).
    
    derivatives_funs: class instance
        Instance of the `derivatives_funs_class` class of derivatives functions computed by Jax Autodiff (they include those employed in `compute_hyperderivatives`
        and `dchi2_dpars` and/or `dchi2_dlambdas`).
    """
    system_names = data_train['global'].system_names

    """ compute derivatives of optimal pars w.r.t. hyper parameters """
    if not np.isinf(log10_alpha):
        lambdas_vec = []
        refs = []

        for name_sys in system_names:
            for name in data_train[name_sys].n_experiments.keys():
                lambdas_vec.append(lambdas[name_sys][name])
                refs.extend(data_train[name_sys].ref[name]*data_train[name_sys].n_experiments[name])

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

    # use non-normalized data and lambdas
    derivatives = compute_hyperderivatives(
        pars_ff_fm, lambdas_vec, data_train, regularization, derivatives_funs, log10_alpha, log10_beta, log10_gamma)

    """ compute chi2 and its derivatives w.r.t. pars"""

    assert which_set in ['training', 'validation', 'test'], 'error on which_set'
    if which_set == 'training':
        my_data = data_train
    else:
        my_data = data_test

    my_args = (
        pars_ff_fm, lambdas_vec, my_data, regularization, 10**(log10_alpha), 10**(log10_beta),
        10**(log10_gamma), which_set)

    chi2 = compute_chi2_tot(*my_args)  # so, lambdas follows order of system_names of my_data

    # if (len(indices) == 0) and np.isinf(log10_beta) and np.isinf(log10_gamma):
    #     return chi2, np.zeros(1)

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

    return chi2, gradient


# %% D5. mini_and_chi2_and_grad

def mini_and_chi2_and_grad(
        data, test_frames, test_obs, regularization, alpha, beta, gamma,
        starting_pars, which_set, derivatives_funs):
    """
    This is an internal tool of `hyper_function` which minimizes the loss function at given hyperparameters, computes the chi2 and
    its gradient w.r.t. the hyperparameters.

    Parameters
    -------------------
    data : dict
        Dictionary which constitutes the `data` object.
    
    test_frames, test_obs : dicts
        Dictionaries for test frames and test observables (for a given `random_state`).

    regularization : dict
        Dictionary for the regularizations (see in `MDRefinement`).

    alpha, beta, gamma : floats
        Values of the hyperparameters.

    starting_pars : array_like
        Numpy 1-dimensional array for starting values of the coefficients in `minimizer`.

    which_set : str
        String among `'training'`, `'validation'` or `'test'` (see in `MDRefinement`).

    derivatives_funs : class instance
        Instance of the `derivatives_funs_class` class of derivatives functions computed by Jax Autodiff.
    """
    out = select_traintest(data, test_frames=test_frames, test_obs=test_obs)
    data_train = out[0]
    data_test = out[1]

    mini = minimizer(
        data_train, regularization=regularization, alpha=alpha, beta=beta, gamma=gamma, starting_pars=starting_pars)

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
        which_set, data_test, derivatives_funs)

    return mini, chi2, gradient

# %% D6. hyper_function


def hyper_function(
        log10_hyperpars, map_hyperpars, data, regularization, test_obs, test_frames, which_set, derivatives_funs,
        starting_pars, n_parallel_jobs):
    """
    This function is an internal tool of `hyper_minimizer` which determines the optimal parameters by minimizing the loss function at given hyperparameters;
    then, it computes chi2 and its gradient w.r.t hyperparameters (for the optimal parameters).

    Parameters
    --------------
    
    log10_hyperpars: array_like
        Numpy array for log10 hyperparameters alpha, beta, gamma (in this order, when present).
    
    map_hyperpars: list
        Legend for `log10_hyperpars` (they refer to alpha, beta, gamma in this order,
        but some of them may not be present, if fixed to `+np.inf`).
    
    data, regularization: dicts
        Dictionaries for `data` and `regularization` objects.
    
    test_obs, test_frames: dicts
        Dictionaries for test observables and test frames, indicized by seeds.
    
    which_set: str
        String, see for `compute_chi2_tot`.
    
    derivatives_funs: class instance
        Derivative functions computed by `Jax` and employed in `compute_hypergradient`.
    
    starting_pars: float
        Starting values of the parameters, if user-defined; `None` otherwise.

    n_parallel_jobs: int
        Number of parallel jobs.

    ------------
    Output variables:
    ------------
    
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
        names_ff_pars = data['global'].names_ff_pars
        pars0 = np.zeros(len(names_ff_pars))
    else:
        pars0 = np.array([])

    if not np.isinf(gamma):
        pars0 = np.concatenate(([pars0, np.array(data['global'].forward_coeffs_0)]))

    """ for each seed: """

    # Results = {}
    # chi2 = []
    # gradient = []  # derivatives of chi2 w.r.t. (log10) hyper parameters

    # args = (data, test_frames[i], test_obs[i], regularization, alpha, beta, gamma, starting_pars,
    # which_set, derivatives_funs)
    random_states = test_obs.keys()

    if n_parallel_jobs is None:
        n_parallel_jobs = len(test_obs)

    output = Parallel(n_jobs=n_parallel_jobs)(delayed(mini_and_chi2_and_grad)(
        data, test_frames[seed], test_obs[seed], regularization, alpha, beta, gamma, starting_pars,
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
        regularization=None, random_states=1, replica_infos=None, which_set='validation',
        gtol=0.5, ftol=0.05, starting_pars=None, n_parallel_jobs=None):
    """
    This tool optimizes the hyperparameters by minimizing the selected chi2 (training, validation or test)
    over several (randomly) splits of the full data set into training/test set.

    Input values:
    --------------
    data : dict
        Object `data`, with the full data set previously loaded.
    
    starting_alpha, starting_beta, starting_gamma : floats
        Starting points of the hyperparameters (`+np.inf` by default, namely no refinement in that direction).
    
    regularization : dict
        Dictionary for the defined regularizations of force-field and forward-model corrections (`None` by default); see for `MDRefinement`.
    
    replica_infos : dict
        Dictionary with information required to split frames following continuous trajectories in replica exchange ("demuxing"); see `select_traintest` for further details.

    random_states : int or list
        Random states (i.e., seeds) used in `select_traintest` to split the data set into training and test set (see `MDRefinement`); 1 by default.
    
    which_set : str
        String choosen among `'training'`, `'validation'`, `'test'` (see in `MDRefinement`); `validation` by default.
    
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

    class hyper_intermediate_class():
        def __init__(self):
            self.av_chi2 = []
            self.av_gradient = []
            self.log10_hyperpars = []

    global hyper_intermediate
    hyper_intermediate = hyper_intermediate_class()

    if type(random_states) is int:
        random_states = [i for i in range(random_states)]

    """ select training and test set (several seeds) """

    test_obs = {}
    test_frames = {}

    for seed in random_states:
        out = select_traintest(data, random_state=seed, replica_infos=replica_infos)
        test_obs[seed] = out[2]
        test_frames[seed] = out[3]

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
        map_hyperpars, data, regularization, test_obs, test_frames, which_set, derivatives_funs,
        starting_pars, n_parallel_jobs)

    # just to check:
    # out = hyper_function(log10_hyperpars0, map_hyperpars, data, regularization, test_obs, test_frames, which_set,
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
        coeff_names = coeff_names + list(data['global'].forward_coeffs_0.keys())

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
    -----------------
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
