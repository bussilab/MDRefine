"""
Tools for the sampling of the posterior distribution, defined over a set of ensembles, by using a suitable
uninformative prior (namely, a prescription on the counting of the ensembles).
"""

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import time, pandas
from .loss_and_minimizer import loss_function
from .MDRefinement import unwrap_2dict

# for class Result:
import re
from typing import List

# to avoid verbose result use `_suppress_stdout`

import sys, os
from contextlib import contextmanager

@contextmanager
def _suppress_stdout():
    """Internal method to avoid internal printing."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# from `bussilab`
class Result(dict):
    # triple ' instead of triple " to allow using docstrings in the example
    '''Base class for objects returning results.

       It allows one to create a return type that is similar to those
       created by `scipy.optimize.minimize`.
       The string representation of such an object contains a list
       of attributes and values and is easy to visualize on notebooks.

       Examples
       --------

       The simplest usage is this one:

       ```python
       from bussilab import coretools

       class MytoolResult(coretools.Result):
           """Result of a mytool calculation."""
           pass

       def mytool():
           a = 3
           b = "ciao"
           return MytoolResult(a=a, b=b)

       m=mytool()
       print(m)
       ```

       Notice that the class variables are dynamic: any keyword argument
       provided in the class constructor will be processed.
       If you want to enforce the class attributes you should add an explicit
       constructor. This will also allow you to add pdoc docstrings.
       The recommended usage is thus:

       ````
       from bussilab import coretools

       class MytoolResult(coretools.Result):
           """Result of a mytool calculation."""
           def __init__(a, b):
               super().__init__()
               self.a = a
               """Documentation for attribute a."""
               self.b = b
               """Documentation for attribute b."""

       def mytool():
           a = 3
           b = "ciao"
           return MytoolResult(a=a, b=b)

       m = mytool()
       print(m)
       ````

    '''

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, item: str, value):
        self[item] = value

    def __delattr__(self, item: str):
        del self[item]

    def __repr__(self) -> str:
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
# when used recursively, the inner repr is properly indented:
            return '\n'.join([k.rjust(m) + ': ' + re.sub("\n", "\n"+" "*(m+2), repr(v))
                              for k, v in sorted(self.items())])
        return self.__class__.__name__ + "()"

    def __dir__(self) -> List[str]:
        return list(sorted(self.keys()))

# class Which_measure

from enum import Enum

class Which_measure(Enum):
    """ Class with the strings for `which_measure` variable. """
    FLAT = 'uniform'
    JEFFREYS = 'jeffreys'
    AVERAGE = 'average'
    DIRICHLET = 'dirichlet'

def _assert_one_finite_one_infinite(a, b):
    """
    Basic routine to assert if either `a` or `b` is a finite float value (and the other is infinite or None).
    If true, it returns the Boolean variables `a_fin` and `b_fin` (`a_fin = True` if `a` is finite,
    `False` otherwise; analogouse for `b_fin`)
    """
    # Treat None as infinite
    a_inf = (a is None) or (a is not None and np.isinf(a))
    b_inf = (b is None) or (b is not None and np.isinf(b))
    
    a_fin = (a is not None) and np.isfinite(a) and a > 0
    b_fin = (b is not None) and np.isfinite(b) and b >= 0

    assert (a_fin and b_inf) or (b_fin and a_inf), \
        "Exactly one value must be finite >0 and the other infinite or None; the second value might be zero"

    return a_fin, b_fin

#%% compute the local density of ensembles

def _make_sym_pos_def(cov, epsilon=1e-8):
    """
    Internal tool to make `cov` symmetric and positive definite
    (`cov` should be so, unless linearly-dependent observables; if it is not, this is due to round-off errors).
    """
    cov = (cov + cov.T)/2
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig <= 0 : cov += (-min_eig + epsilon)*np.eye(cov.shape[0])  # < or <= ? Implement a Bool for this choice!
    return cov

def local_density(variab, weights, which_measure = 'jeffreys'):
    """
    This function computes the local density of ensembles in the cases of ensemble refinement or force-field fitting.
    
    This density can be defined through the Jeffreys "uninformative" prior (`which_measure = 'jeffreys'`):
    in these two cases, the Jeffreys prior is given by the square root of the determinant of the covariance matrix
    (of the observables in Ensemble Refinement or the generalized forces in Force-Field Fitting,
    where the generalized forces are the derivatives of the force-field correction with respect to the fitting coefficients).
    
    It includes also the possibility for the computation of the local density of ensembles with plain Dirichlet
    if `which_measure = 'dirichlet'`, or with the variation of the average observables if 
    `which_measure = 'average'`.

    Since we are anyway dealing with a real-value, symmetric and semi-positive definite matrix,
    its determinant is computed through the Cholesky decomposition (which is faster for big matrices):
    `triang` is such that `metric = triang * triang.T`, so `sqrt(det metric) = det(triang)`.

    Parameters
    -----------
    
    variab : numpy.ndarray, dict or tuple
        For Ensemble Refinement, `variab` is either the dictionary `data.mol[name_mol].g` to be unwrapped
        or directly the numpy array with the observables defined in each frame.
        
        For Force-Field Fitting and `which_measure == 'jeffreys' or 'dirichlet'`, `variab` is the tuple `(fun_forces, pars, f)` where:
            - `fun_forces` is the function for the gradient of the force-field correction with respect to `pars`
            (defined through Jax as `fun_forces = jax.jacfwd(ff_correction, argnums=0)` where `ff_correction = data.mol[name_mol].ff_correction`;
            you can compute it just once at the beginning of the MC sampling);
            - `pars` is the numpy.ndarray of parameters for the force-field correction;
            - `f` is the numpy.ndarray `data.mol[name_mol].f` with the terms required to compute the force-field correction.
        If `which_measure = 'average'`, then the observables are required, too, and `variab` is the tuple `(fun_forces, pars, f, g)`.

        See documentation of `MDRefine` at https://www.bussilab.org/doc-MDRefine/MDRefine/index.html for further details
        about the `data` object.

    weights : numpy.ndarray
        Numpy array with the normalized weights of each frame; this is the probability distribution
        at which you want to compute the Jeffreys prior, corresponding to the local density of ensembles.

    which_measure: str
        String variable, chosen among: `jeffreys`, `dirichlet` or `average`, indicating the prescription
        for the local density of ensembles (Jeffreys prior, plain Dirichlet, average observables).

    -----------

    Returns
    -----------

    measure : float
        The local density of ensembles at the given distribution `weights`, computed as specified by `which_measure`
        (Jeffreys prior by default).
    
    cov : numpy.ndarray
        The metric tensor for the chosen metrics defined by `which_measure` if `which_measure = 'jeffreys'` or `'dirichlet'`;
        the covariance matrix if `which_measure = 'average'`.
    """

    if which_measure == 'jeffreys' or (which_measure == 'average' and type(variab) is not tuple):
        # in this case, the density is given by computing the variance-covariance matrix of values
        # (either forces or observables)

        if type(variab) is tuple: values = variab[0](variab[1], variab[2])
        else:
            if type(variab) is dict: values = np.hstack([variab[s] for s in variab.keys()])
            elif type(variab) is np.ndarray and len(variab.shape) == 1: values = np.array([variab]).T
            else: values = variab

        av_values = np.einsum('ti,t->i', values, weights)
        cov = np.einsum('ti,tj,t->ij', values, values, weights) - np.outer(av_values, av_values)

        # exploit the Cholesky decomposition:
        # metric = triang*triang.T, so sqrt(det metric) = det(triang)
        try:  # it may happen: `Matrix is not positive definite` (zero due to round-off errors)
            triang = np.linalg.cholesky(cov)
            density = np.prod(np.diag(triang))
        except:
            # density = np.sqrt(np.linalg.det(cov))

            cov = _make_sym_pos_def(cov)
            triang = np.linalg.cholesky(cov)
            density = np.prod(np.diag(triang))

        if which_measure == 'average': density = density**2

        return density, cov

    elif which_measure == 'average' and type(variab) is tuple:
        # in this case, we are sampling Force-Field Fitting with 'average' measure of ensembles
        # so we have to compute the covariance matrix of observables and forces;
        # then, since it is not a square matrix in general, you cannot compute its det,
        # but you have to compute the sqrt of det (C.T C)
        
        assert len(variab) == 4

        forces = variab[0](variab[1], variab[2])
        
        if type(variab[3]) is dict: g = np.hstack([variab[s] for s in variab.keys()])
        else: g = variab[3]

        av_forces = np.einsum('ti,t->i', forces, weights)
        av_g = np.einsum('ti,t->i', g, weights)
        cov = np.einsum('ti,tj,t->ij', forces, g, weights) - np.outer(av_forces, av_g)
        
        metric = np.einsum('ji,ki->jk', cov, cov)
        triang = np.linalg.cholesky(metric)
        density = np.prod(np.diag(triang))

        return density, cov

    else:
        assert which_measure == 'dirichlet', 'error on `which_measure`'

        if type(variab) is tuple: values = variab[0](variab[1], variab[2])
        else:
            if type(variab) is dict: values = np.hstack([variab[s] for s in variab.keys()])
            else: values = variab

        av_values = np.einsum('ti,t->i', values, weights)
        metric = np.einsum('ti,tj,t->ij', values, values, weights**2) + np.sum(weights**2)*np.outer(av_values, av_values)
        met = np.einsum('i,tj,t->ij', av_values, values, weights**2)
        metric -= met + met.T

        # metric = _make_sym_pos_def(metric)

        # assert np.linalg.cholesky(metric), metric
        # if not (np.isnan(metric).any() or np.isnan(metric).any()) and (np.all(np.linalg.eigvalsh(metric) > 0)):
        try:
            triang = np.linalg.cholesky(metric)
            density = np.prod(np.diag(triang))
        except:  # np.linalg.LinAlgError as e:  # it may happen: `Matrix is not positive definite` (zero due to round-off errors)
            metric = _make_sym_pos_def(metric, epsilon=1e-8)
            triang = np.linalg.cholesky(metric)
            density = np.prod(np.diag(triang))

            # my_det = np.linalg.det(metric)
            # assert type(my_det) is np.float64, 'error on my_det: %f, type %s' % (my_det, str(type(my_det)))
            # if my_det == 0 : my_det = 1e-8
            # density = np.sqrt(my_det)

        return density, metric

#%% proposal move

class Proposal_onebyone:
    """ Class for a proposal move which updates one coordinate per time (it includes the attribute `index`
        to take in memory which coordinate to update) """
    def __init__(self, step_width = 1., index = 0, rng = None):
        self.step_width = step_width
        self.index = index
        
        if rng is None: self.rng = np.random.default_rng(np.random.randint(1000))
        else: self.rng = rng
    
    def __call__(self, x0):
        
        x_new = + x0
        x_new[self.index] = x0[self.index] + self.step_width*self.rng.normal()

        self.index += 1
        self.index = int(np.mod(self.index, len(x0)))
        
        return x_new

#%% Metropolis algorithm (general)

class Saving_function():
    def __init__(self, values : dict={}, t0 : float=0., date : str='', path : str='.', i_save : int=10000):
        self.values = values
        self.date = date

        if t0 == 0: self.t0 = time.time()
        else: self.t0 = t0
        
        self.path = path
        self.i_save = i_save

    def __call__(self, av_acceptance, traj, energy, qs = None):

        self.values['av. acceptance'] = av_acceptance
        self.values['time'] = time.time() - self.t0

        temp = pandas.DataFrame(list(self.values.values()), index=list(self.values.keys()), columns=[self.date]).T
        temp.to_csv(self.path + '/par_values')

        np.save(self.path + '/trajectory', traj)
        np.save(self.path + '/energy', energy)

        # if type(sampling[2]) is not float:  # if float, it is the average acceptance
        if qs is not None:
            np.save(self.path + '/quantities', qs)

class Result_run_Metropolis(Result):
    '''Result of a `run_Metropolis` calculation.'''
    def __init__(self, traj, ene, av_acceptance, quantities = None):
        self.traj = traj
        """ Trajectory """
        self.ene = ene
        """ Energy """
        self.av_acceptance = av_acceptance
        """ Float value for the average acceptance """
        if quantities is not None:
            self.quantities = quantities
            """ Computed quantities """

def run_Metropolis(x0, proposal, energy_function, quantity_function = None, *, kT = 1.,
    n_steps = 100, seed = 1, i_print = 10000, if_tqdm = True, saving = None):
    """
    This function runs a Metropolis sampling algorithm.
    
    Parameters
    -----------

    x0 : numpy.ndarray
        Numpy array for the initial configuration.
    
    proposal : function or float or tuple
        Function for the proposal move, which takes as input variables just the starting configuration `x0`
        and returns the new proposed configuration (trial move of Metropolis algorithm).
        Alternatively, float variable for the standard deviation of a (zero-mean) multi-variate Gaussian variable
        representing the proposed step (namely, the stride).
        Another possibility is the tuple `('one-by-one', step)` where `step` is a float or int variable;
        in this case, the proposal is done on each coordinate one at a time, following a cycle.

    energy_function : function
        Function for the energy, which takes as input variables just a configuration (`x0` for instance)
        and returns its energy; `energy_function` can return also some quantities of interest,
        defined on the input configuration.
        If your energy function `energy_fun` has more than one input variables, just redefine it as
        `energy_function = lambda x : energy_fun(x, simple_model, 'dirichlet')` before passing `energy_function`
        to `run_Metropolis`.
    
    quantity_function : function
        Function used to compute some quantities of interest on the initial configuration.
        If `energy_function` has more than one output, `quantity_function` is ignored and the quantities
        of interest are the 2nd output of `energy_function` (in this way, they are computed together with
        the energy, avoiding the need for running twice the same function).
        Notice that `quantity_function` does not support other input parameters beyond the configuration;
        otherwise, you can use `energy_function`.

    kT : float
        Temperature of the Metropolis sampling algorithm.

    n_steps : int
        Number of steps of Metropolis.
    
    seed : int
        Seed for the random generation.
    
    i_print : int
        How many steps to print an indicator of the running algorithm (current n. of steps).

    if_tqdm : Bool
        Boolean variable, if `True` then use `tqdm`.

    saving : None or float or Saving_function
        An instance of the `Saving_function` class, used to save the results during Metropolis run (or in the end).
        If `saving is None` do not save, if it is `'yes'` use default object of class `Saving_function`.

    -----------

    Returns
    -----------

    obj_result : Result_run_Metropolis
        An instance of the `Result_run_Metropolis` class with trajectory, energy, average acceptance
        and computed quantities.
    """

    rng = np.random.default_rng(seed)

    if saving == 'yes': saving = Saving_function()
    if saving is None: i_save = n_steps - 1
    else: i_save = saving.i_save

    if energy_function is None:
        # energy_function = {'fun': lambda x : 0, 'args': ()}
        energy_function = lambda x : 0

    if type(proposal) is float:
        
        proposal_stride = proposal
        # def fun_proposal(x0, dx = 0.01):
        #     x_new = x0 + dx*np.random.normal(size=len(x0))
        #     return x_new

        # proposal = {'fun': fun_proposal, 'args': ([proposal])}

        def proposal_fun(x0):
            x_new = x0 + proposal_stride*rng.normal(size=len(x0))
            return x_new
    
    elif (proposal == 'one-by-one') or ((type(proposal) is tuple) and (proposal[0] == 'one-by-one')):
        
        if type(proposal) is tuple:
            assert (type(proposal[1]) is int) or (type(proposal[1]) is float), 'error on proposal'
            step_width = proposal[1]
        else: step_width = 1.

        proposal_fun = Proposal_onebyone(step_width=step_width, rng=rng)

    else:
        assert callable(proposal), 'error on proposal'
        proposal_fun = proposal

    x0_ = +x0  # in order TO AVOID OVERWRITING!
    
    traj = []
    ene = []
    quantities = []
    sum_alpha = 0

    traj.append([])
    traj[-1] = +x0_

    # energy_function may have more than one output
    # out = energy_function['fun'](x0_, *energy_function['args'])
    out = energy_function(x0_)
    u0 = out[0]

    if len(out) == 2:
        print('Warning: the quantities of interest are given by energy_function')  #  and not by quantity_function')
        q0 = out[1]  # if `energy_function` has more than one output, the second one is the quantity of interest
    else: q0 = quantity_function(x0_)
    
    ene.append([])
    ene[-1] = +u0

    quantities.append([])
    quantities[-1] = q0

    counter = range(n_steps)
    if if_tqdm: counter = tqdm(counter)

    for i_step in counter:

        x_try = +proposal_fun(x0_)  # proposal['fun'](x0_, *proposal['args'])

        out = energy_function(x_try)  # energy_function['fun'](x_try, *energy_function['args'])
        u_try = out[0]

        alpha = np.exp(-(u_try - u0)/kT)
        
        if alpha > 1: alpha = 1
        if alpha > rng.random():  # move accepted!
            sum_alpha += 1
            x0_ = +x_try
            u0 = +u_try

            if len(out) == 2: q0 = out[1]
            else: q0 = quantity_function(x0_)
        
        # traj.append(x0_)
        # to avoid overwriting!
        traj.append([])
        traj[-1] = +x0_
        
        ene.append([])
        ene[-1] = +u0

        quantities.append([])
        quantities[-1] = q0

        if (not if_tqdm) and (np.mod(i_step, i_print) == 0): print(i_step)

        if (np.mod(i_step, i_save) == 0) or (i_step == (n_steps - 1)):
            av_acceptance = sum_alpha/(i_step + 1)
            if saving is not None:
                if quantities[0] is not None:
                    qs = np.array(quantities)
                    saving(av_acceptance, np.array(traj), np.array(ene), qs)
                else:
                    saving(av_acceptance, np.array(traj), np.array(ene))
    
    if quantities[0] is None: obj_result = Result_run_Metropolis(np.array(traj), np.array(ene), av_acceptance)
    else: obj_result = Result_run_Metropolis(np.array(traj), np.array(ene), av_acceptance, np.array(quantities))
    return obj_result

    # if quantities[0] is None: return np.array(traj), np.array(ene), av_acceptance
    # else: return np.array(traj), np.array(ene), av_acceptance, np.array(quantities)

def langevin_sampling(energy_fun, starting_x, n_iter : int = 10000, gamma : float = 1e-1,
    dt : float = 5e-3, kT : float = 1., seed : int = 1, if_tqdm: bool = True):
    """
    A function to perform a Langevin sampling of `energy_fun` at temperature `kT` (with the Euler-Maruyama scheme).
    
    Parameters
    ----------

    energy_fun : function
        The energy function, written with `jax.numpy` in order to do automatic differentiation
        through `jax.grad` (this requires `energy_fun` to return a scalar value and not an array,
        otherwise you should use `jax.jacfwd` for example; to this aim, you can do 
        `jnp.sum(energy_fun(x))`).
    
    starting_x : numpy.ndarray
        The starting configuration of the Langevin sampling.
    
    n_iter : int
        Number of iterations.
    
    gamma : float
        Friction coefficient.
    
    dt : float
        Time step.
    
    kT : float
        The temperature.
    
    seed : int
        Integer value for the seed.

    if_tqdm : Bool
        Boolean variable, if `True` use `tqdm` (default choice).

    -----------

    Returns
    ----------

    traj : np.ndarray
        Numpy array with the trajectory.

    ene : np.ndarray
        Numpy array with the energies.

    force_list : list
        List with the forces.

    check : dict
        Dictionary with `'dif'` for `np.ediff1d(traj)`, together with its mean and standard deviation.
    """

    jax_energy_fun = lambda x : jnp.sum(energy_fun(x))  # to use jax.grad rather than jax.jacfwd

    rng = np.random.default_rng(seed)
    grad = jax.grad(jax_energy_fun)

    sigma = np.sqrt(2*kT*gamma)
    step_width = sigma*np.sqrt(dt)

    traj = []
    ene_list = []
    force_list = []

    # x = jnp.array(starting_x)
    x = +starting_x
    force = -grad(x)

    traj.append(x)
    ene_list.append(jax_energy_fun(x))
    force_list.append(force)

    counter = range(n_iter)
    if if_tqdm: counter = tqdm(counter)
    
    for i in counter:
        r = rng.normal(size=len(x))
        x += gamma*force*dt + step_width*r
        force = -grad(x)

        traj.append(x)
        ene_list.append(jax_energy_fun(x))
        force_list.append(force)

    # check: steps not too big!!
    dif = np.ediff1d(traj)
    mean = np.mean(dif)
    std = np.std(dif)
    check = {'dif': dif, 'mean': mean, 'std': std}

    traj = np.array(traj)
    if len(x) == 1: traj = traj[:, 0]

    ene = np.array(ene_list)

    return traj, ene, force_list, check

#%% Metropolis algorithm (specific for sampling the posterior of ER or FFR)

class MyQuantities():
    """Class with the evaluated quantities for each step of the MCMC sampling, beyond energy and trajectory."""
    def __init__(self, loss, reg, avs):
        self.loss = loss
        """`float` with the loss value (excluding the entropic contribution)."""
        self.reg = reg
        """`float` with the regularization value."""
        self.avs = avs
        """`float` with the average values."""

    @classmethod
    def merge(cls, instances):
        """
        Function to merge multiple instances of `MyQuantities` in a single one
        (to be run in the end of the MCMC sampling to collect quantities).
        """
        # get all attribute names
        attrs = vars(instances[0]).keys()

        merged = {}
        for attr in attrs:
            arrays = [getattr(obj, attr) for obj in instances]
            arrays = np.stack(arrays).T
            if arrays.shape[0] == 1: arrays = arrays[0]
            
            merged[attr] = arrays
            # try: merged[attr] = np.concatenate(arrays, axis=0)  # join arrays
            # except: merged[attr] = np.stack(arrays)  # for 0-dim arrays
        return Result_MyQuantities(cls(**merged))
        # cls(**merged) requires attributes of MyQuantities to be equal to the input variables __init__(self, ...)

class Result_MyQuantities(Result):
    """Class with the merged quantities from `MyQuantities` (`MyQuantities.merge`)."""
    def __init__(self, my_quantities_concat : MyQuantities):
        super().__init__(**my_quantities_concat.__dict__)

def energy_fun(x, data, regularization, alpha = np.inf, beta = np.inf, which_measure = 'uniform'):
    """
    This is the energy function defined for running the usual sampling algorithms, corresponding to -log of the
    posterior distribution (a part from a normalization factor and with the optional inclusion of the entropic
    contribution, as prescribed by `which_measure`). Depending on which hyperparameter is infinite (`alpha` or
    `beta`), it corresponds either to ensemble refinement or force-field fitting.

    Parameters
    -----------

    x : numpy.ndarray
        Numpy array with the lambda coefficients (for ensemble refinement) or the force-field correction coefficients
        (for force-field refinement), in the same order required by `loss_and_minimizer.loss_function`.

    data : data_loading.my_data
        An instance of the class `data_loading.my_data` class, with all the data for the molecules of interest.

    regularization : dict
        Dictionary for the regularization (`None` for ensemble refinement), as described for `MDRefinement`.

    alpha, beta : float
        Values of the hyperparameters `alpha` (ensemble refinement) or `beta` (force-field fitting):
        either one of them must be infinite (the sampling has been implemented either for ensemble
        or force-field refinement).

    which_measure : dict
        Dictionary indicating the measure used for sampling the posterior
        (choose among: `'uniform'`, `'jeffreys'`, `'average'`, `'dirichlet'`).

    -----------

    Returns
    -----------

    energy : float
        Float value for the energy used in the sampling, as defined by the input variables.

    qs : MyQuantities
        An instance of the `'MyQuantities` class containing loss, average observables and regularization values.
    """

    # vars(out).keys() = ['loss', 'loss_explicit', 'D_KL_alpha', 'abs_difference', 'av_g', 'chi2',
    #    'logZ_new', 'weights_new'] """

    a_fin, b_fin = _assert_one_finite_one_infinite(alpha, beta)

    if a_fin:
        # `if_save = True` and the correct value is `out.loss_explicit`, otherwise you are wrong because
        # it would compute the value given by the Gamma function rather than the loss itself
        # (these two values are equal only in the optimal solution!!)
        out = loss_function(None, data, regularization=None, alpha=alpha, fixed_lambdas=x, if_save=True)
        
        energy = out.loss_explicit

        qs = MyQuantities(energy, list(out.D_KL_alpha.values()), unwrap_2dict(out.av_g)[0])
        # qs = [energy] + list(out.D_KL_alpha.values()) + unwrap_2dict(out.av_g)[0]
    
    else:
        # here alpha is infinite so you could keep `if_save=False` and evaluate `energy = out.loss`
        # (no issue with the Gamma function); put anyway `if_save=True` to get also the average observables values
        out = loss_function(x, data, regularization=regularization, beta=beta, if_save=True)
        
        energy = out.loss

        qs = MyQuantities(energy, list(out.reg_ff.values()), unwrap_2dict(out.av_g)[0])
        # qs = [energy] + list(out.reg_ff.values()) + unwrap_2dict(out.av_g)[0]

    if which_measure != 'uniform':
        name_mol = list(out.weights_new.keys())[0]
        measure, cov = local_density(data.mol[name_mol].g, out.weights_new[name_mol], which_measure)
        energy -= np.log(measure)
    
    return energy, qs

def _energy_fun_mute(x0, data, regularization, alpha, beta, which_measure):
    "The same as `energy_fun` but without internal printing."
    with _suppress_stdout():
        return energy_fun(x0, data, regularization, alpha, beta, which_measure)

def posterior_sampling(starting_point, data, regularization = None, alpha : float = np.inf, beta : float = np.inf,
                       which_measure = Which_measure, proposal_move = 'default', n_steps_MC : int = int(1e4),
                       seed : int = 1):
    """
    Main function of the `bayesian` module, it is the algorithm that samples from the posterior distribution
    exp(-L(P)) with the specified uninformative prior, either in the case of ensemble refinement or force-field refinement.

    Parameters
    -----------

    starting_point : 

    data : data_loading.my_data
        An instance of the class `data_loading.my_data` class, with all the data for the molecules of interest.

    regularization : dict
        Dictionary for the regularization to the force-field correction.

    alpha, beta : float
        Float values for the hyperparameters (either one of them must be infinite or None).
    
    which_measure : Which_measure
        An instance of the `Which_measure` class, to specify the entropic measure used in the sampling
        (chosen among `FLAT = 'uniform'`, `JEFFREYS = 'jeffreys'`, `AVERAGE = 'average'`, `DIRICHLET = 'dirichlet'`).

    proposal_move : str or function or float or tuple
        Variable used to specify the move employed in the Metropolis algorithm, as indicated in `run_Metropolis`;
        if it is `'default'`, then a Gaussian move is used with standard deviation `proposal_move = 0.1`.

    n_steps_MC : int
        Integer for the number of steps in the Metropolis algorithm.

    seed : int
        Integer for the random state (seed) used in the Metropolis algorithm.

    -----------
    
    Returns
    -----------
    
    sampling : Result_MyQuantities
        An instance of the `Result_MyQuantities` class, which merges the quantities returned by each step
        of the MCMC sampling (as indicated in `MyQuantities`).
    """
    
    assert not ((not np.isinf(beta)) and (regularization is None)), 'regularization is None even if beta is not infinite'
    
    a_fin, b_fin = _assert_one_finite_one_infinite(alpha, beta)

    energy_function = lambda x0 : _energy_fun_mute(x0, data, regularization, alpha, beta, which_measure.value)
    
    if proposal_move == 'default': proposal_move = 0.1
    # then, `run_Metropolis` will take a random move given by a normal distribution of given std

    sampling = run_Metropolis(starting_point, proposal_move, energy_function, n_steps=n_steps_MC, seed=seed)

    sampling.quantities = MyQuantities.merge(sampling.quantities)

    return sampling

#%% block analysis

class Block_analysis_Result(Result):
    """Result of a `block_analysis` calculation."""
    def __init__(self, mean : float, std : float, opt_epsilon : float, epsilons : np.ndarray,
            epsilons_smooth : np.ndarray, n_blocks : np.ndarray, size_blocks : np.ndarray):
        super().__init__()
        self.mean = mean
        """`float` with the mean value of the time series."""
        self.std = std
        """`float` with the standard deviation of the time series (assuming independent frames)."""
        self.opt_epsilon = opt_epsilon
        """`float` with the optimal estimate of the associated error `epsilon`."""
        self.epsilons = epsilons
        """`list` with the associated error `epsilon` for each block size."""
        self.epsilons_smooth = epsilons_smooth
        """`list` with the associated error `epsilon` for each block size (smooth time series)."""
        self.n_blocks = n_blocks
        """`list` with the number of blocks in the time series, for each analysed block size."""
        self.size_blocks = size_blocks
        """`list` with the block sizes initially defined."""

def block_analysis(x, size_blocks = None, n_conv = 50):
    """
    This function performs the block analysis of a (correlated) time series `x`, cycling over different block sizes.
    It includes also a numerical search of the optimal estimated error `epsilon`, by smoothing `epsilon` and searching
    for the first time it decreases, which should correspond to a plateau region.

    It returns an instance of the `Block_analysis_Result` class.

    Parameters
    -----------

    x : numpy.ndarray
        Numpy array with the time series of which you do block analysis.

    size_blocks : list, int or None
        The list with the block sizes used in the analysis; you can either pass an integer value,
        in this case the list of sizes is given by `np.arange(1, np.int64(size/2) + size_blocks, size_blocks)`;
        further, if `size_blocks` is `None`, the list of sizes is `np.arange(1, np.int64(size/2) + 1, 1)`.

    n_conv : int
        Length (as number of elements in the block-size list) of the kernel used to smooth the epsilon function
        (estimated error vs. block size) in order to search for the optimal epsilon, corresponding to the plateau.
    """

    size = len(x)
    mean = np.mean(x)
    std = np.std(x)/np.sqrt(size)

    if size_blocks is None: size_blocks = np.arange(1, np.int64(size/2) + 1, 1)
    elif type(size_blocks) is int: size_blocks = np.arange(1, np.int64(size/2) + size_blocks, size_blocks)
    else: assert type(size_blocks) is list, 'incorrect size_blocks'

    n_blocks = []
    epsilon = []

    for size_block in size_blocks:

        n_block = np.int64(size/size_block)
        
        # a = 0 
        # for i in range(n_block):
        #     a += (np.mean(x[(size_block*i):(size_block*(i+1))]))**2
        # 
        # epsilon.append(np.sqrt((a/n_blocks[-1] - mean**2)/n_blocks[-1]))

        block_averages = []
        for i in range(n_block): block_averages.append(np.mean(x[(size_block*i):(size_block*(i+1))]))
        block_averages = np.array(block_averages)

        n_blocks.append(n_block)
        epsilon.append(np.sqrt((np.mean(block_averages**2) - np.mean(block_averages)**2)/n_block))

    # find the optimal epsilon: smooth the epsilon function and find the first time it decreases
    kernel = np.ones(n_conv)/n_conv
    smooth = np.convolve(epsilon, kernel, mode='same')
    diff = np.ediff1d(smooth)
    wh = np.where(diff < 0)
    opt_epsilon = smooth[wh[0][0]]
    
    return Block_analysis_Result(mean, std, opt_epsilon, epsilon, smooth, n_blocks, size_blocks)

