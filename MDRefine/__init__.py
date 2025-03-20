"""A package to perform refinement of MD simulation trajectories.

Source code is on [GitHub](https://github.com/bussilab/MDRefine).
A test pdf manual is available [here](../MDRefine.pdf).

### Examples

In the [examples](../examples) directory you can find a number of notebooks
that can be used as a source of inspiration.
"""

from ._version import __version__

from .data_loading import check_and_skip, load_data
from .loss_and_minimizer import compute_js, compute_new_weights, gamma_function, normalize_observables, compute_D_KL
from .loss_and_minimizer import l2_regularization, compute_chi2, compute_DeltaDeltaG_terms, compute_details_ER, loss_function, loss_function_and_grad, deconvolve_lambdas
from .loss_and_minimizer import minimizer, split_dataset, validation
from .hyperminimizer import compute_hyperderivatives, compute_chi2_tot, put_together, compute_hypergradient
from .hyperminimizer import mini_and_chi2_and_grad, hyper_function, hyper_minimizer
from .MDRefinement import MDRefinement, unwrap_2dict, save_txt, unwrap_dict, compute_chi2_test

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
