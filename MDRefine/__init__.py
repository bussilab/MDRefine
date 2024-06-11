"""A package to perform refinement of MD simulation trajectories.

TODO
"""

from ._version import __version__
from .Functions import *

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
