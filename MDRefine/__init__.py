"""A package to perform refinement of MD simulation trajectories.

TODO
"""

from ._version import __version__
from .Functions import check_and_skip, load_data, compute_js, compute_new_weights, gamma_function, normalize_observables, compute_D_KL
from .Functions import l2_regularization, compute_chi2, compute_DeltaDeltaG_terms, compute_details_ER, loss_function, loss_function_and_grad, deconvolve_lambdas
from .Functions import minimizer, select_traintest, validation, compute_hyperderivatives, compute_chi2_tot, put_together, compute_hypergradient
from .Functions import mini_and_chi2_and_grad, hyper_function, hyper_minimizer, MDRefinement, unwrap_2dict, save_txt

# required packages:
_required_ = [
    'numpy',
    'pandas',
    'jax',
    'jaxlib',
    'joblib'
]


def get_version():
    return __version__
