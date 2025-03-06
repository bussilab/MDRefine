"""
Tools n. 1: `data_loading`.
It loads data into the `data` object.
"""

import os
import pandas

# numpy is required for loadtxt and for gradient arrays with L-BFGS-B minimization (rather than jax.numpy)
import numpy
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)

# %% A. Functions to load data:
# %% A1. check_and_skip

def check_and_skip(data, *, stride=1):
    """
    This function is an internal tool used in `load_data` to modify input `data`:

    - weights are normalized;

    - it appends observables computed through forward models (if any) to `data.mol[name_sys].g`;
    
    - if ` hasattr(data.mol[name_sys], 'selected_obs')`: it removes non-selected observables from `data.mol[name_sys].forward_qs`;
    
    - select frames with given `stride`;
    
    - count n. experiments and n. frames (`data.mol[name_sys].n_frames` and `data.mol[name_sys].n_experiments`)
    and check corresponding matching.
    """

    # output_data = {}
    # output_data['global'] = data.properties

    system_names = data.properties.system_names

    for name_sys in system_names:

        my_data = data.mol[name_sys]

        """ 1. compute observables from .forward_qs through forward model and include them in .g """

        if hasattr(my_data, 'forward_model') and (my_data.forward_model is not None):
            if not hasattr(my_data, 'g'):
                my_data.g = {}

            if hasattr(my_data, 'selected_obs'):
                for type_name in my_data.forward_qs.keys():
                    my_data.forward_qs[type_name] = my_data.forward_qs[type_name]  # [:,my_data.selected_obs[name][type_name]]

            if hasattr(my_data, 'selected_obs'):
                selected_obs = my_data.selected_obs
            else:
                selected_obs = None

            out = my_data.forward_model(np.array(data.properties.forward_coeffs_0), my_data.forward_qs, selected_obs)

            if type(out) is tuple:
                out = out[0]

            if not hasattr(my_data, 'g'):
                my_data.g = {}
            for name in out.keys():
                my_data.g[name] = out[name]

        """ 2. check data """

        if not hasattr(my_data, 'g'):

            my_list = [x2 for x in list(data.properties.cycle_names.values()) for x2 in x]
                
            assert hasattr(data, 'cycle'), 'error: missing MD data for system' + name_sys
            assert name_sys in my_list, 'error: missing MD data for system' + name_sys

            for s2 in data.properties.cycle_names:
                if name_sys in data.properties.cycle_names[s2]:
                    assert hasattr(data.cycle[s2], 'gexp_DDG'), 'error: missing gexp DDG for cycle' % s2

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

        if not (hasattr(my_data, 'g') or hasattr(my_data, 'forward_qs') or hasattr(data.cycle[name_sys[:-3]], 'gexp_DDG')):
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

        # output_data[name_sys] = my_data
        data.mol[name_sys] = my_data
        del my_data

    # if hasattr(data.properties, 'cycle_names'):
    #     for name in data.properties.cycle_names:
            # output_data[name] = data[name]

    return data  # output_data

# %% A2. load_data

class datapropertiesclass:
    """Global data, common to all the investigated molecular systems.
    
    Parameters
    ----------

    info_global: dict
        Dictionary with global information:
        `info_global['system_names']` with list of names of the molecular systems;
        `info_global['cycle_names']` with list of names of the thermodynamic cycles;
        `info_global['forward_coeffs']` with string for the file name of forward coefficients;
        `info_global['names_ff_pars']` with list of names of the force-field correction coefficients.

    path_directory: str
        String with the path of the directory with input files.

    --------
    Returns
    --------
    system_names : list
        List of names of the investigated molecular systems.
    
    forward_coeffs_0 : list
        List of the forward-model coefficients.
    
    names_ff_pars : list
        List of names of the force-field correction parameters.

    cycle_names : list
        List of names of the investigated thermodynamic cycles.
    """
    def __init__(self, infos, path_directory):

        self.infos = infos

        info_global = infos['global']

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
            for item in data.mol[k].n_experiments.values():
                tot += item
        return tot


class data_class:
    """
    Data object of a molecular system.

    Parameters
    ----------
    info: dict
        Dictionary for the information about the data of `name_sys` molecular system in `path_directory`. 

    path_directory: str
        String for the path of the directory with data of the molecular system `name_sys`.

    name_sys: str
        Name of the molecular system taken into account.
    
    --------

    Returns
    --------
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
    
    f: array_like
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
                    for s in c.keys():
                        if list(c[s]) == []:
                            del out[s]
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
    ----------
    cycle_name : str
        String with the name of the thermodynamic cycle taken into account.
    
    DDGs_exp : pandas.DataFrame
        Pandas.DataFrame with the experimental values and uncertainties of Delta Delta G in labelled thermodynamic cycles.

    info: dict
        Dictionary for the information about the temperature of `cycle_name` thermodynamic cycle. 

    --------
    Returns
    --------
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


class my_data:
    def __init__(self, infos):
        
        system_names = infos['global']['system_names']
        path_directory = infos['global']['path_directory']
        if not path_directory[-1] == '/': path_directory += '/'
        
        # global data
        self.properties = datapropertiesclass(infos, path_directory)

        # data for each molecular system

        self.mol = {}

        for name_sys in system_names:

            print('loading ', name_sys)
            
            if name_sys in infos.keys():
                info = {**infos[name_sys], **infos['global']}
            else:
                info = infos['global']
    
            self.mol[name_sys] = data_class(info, path_directory, name_sys)

        # data for thermodynamic cycles (alchemical calculations)

        if 'cycle_names' in infos['global'].keys():

            logZs = pandas.read_csv(path_directory + 'alchemical/logZs', index_col=0, header=None)

            for name in infos['global']['cycle_names']:
                for s in ['MD', 'MS', 'AD', 'AS']:
                    key = name + '_' + s
                    if key in logZs.index:
                        self.mol[key].logZ = logZs.loc[key][1]
                    else:
                        self.mol[key].logZ = 0.0

            self.cycle = {}

            DDGs_exp = pandas.read_csv(path_directory + 'alchemical/DDGs', index_col=0)

            for name in infos['global']['cycle_names']:
                if name in infos.keys():
                    info = {**infos[name], **infos['global']}
                else:
                    info = infos['global']

                self.cycle[name] = data_cycle_class(name, DDGs_exp, info)


def load_data(infos, *, stride=1):
    """
    This tool loads data from specified directory as indicated by the user in `infos`
    to a dictionary `data` of classes, which includes `data.properties` (global properties) and `data[system_name]`;
    for alchemical calculations, there is also `data[cycle_name]`.
    """

    print('loading data from directory...')

    data = my_data(infos)

    # check and skip frames with stride

    data = check_and_skip(data, stride=stride)

    # def tot_n_experiments(data):
    #     tot = 0
    #     for k in system_names:
    #         for item in data[k].n_experiments.values():
    #             tot += item
    #     return tot

    # data.properties.system_names = system_names
    # data.properties.tot_n_experiments = tot_n_experiments

    # if hasattr(data.properties, 'ff_correction') and (data.properties.ff_correction == 'linear'):
    #     list_names_ff_pars = []
    #     for k in data.properties.system_names:
    #         if hasattr(data[k], 'f'):
    #             [list_names_ff_pars.append(x) for x in data[k].f.keys() if x not in list_names_ff_pars]
    #     data.properties.names_ff_pars = list_names_ff_pars

    # elif 'names_ff_pars' in infos['global'].keys():
    #     data.properties.names_ff_pars = infos['global']['names_ff_pars']

    print('done')

    return data
