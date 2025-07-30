# MDRefine

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/bussilab/MDRefine)](https://github.com/bussilab/MDRefine/tags)
[![PyPI](https://img.shields.io/pypi/v/MDRefine)](https://pypi.org/project/MDRefine/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/MDRefine)](https://pypi.org/project/MDRefine/)
[![CI](https://github.com/bussilab/MDRefine/workflows/CI/badge.svg)](https://github.com/bussilab/MDRefine/actions?query=workflow%3ACI)

---

A package to perform refinement of MD simulation trajectories.  
📚 **Documentation** is available at the [project page](https://bussilab.github.io/doc-MDRefine/).

---

## 🚀 Installation

To install the package and dependencies from [PyPI](https://pypi.org/project/MDRefine/):

```bash
pip install MDRefine
```

To install the **development version** from source:

```
git clone https://github.com/bussilab/MDRefine.git
cd MDRefine
pip install -e .
```

We recommend installing in a dedicated environment to avoid polluting your base Python setup:

```
python3 -m venv env
source env/bin/activate
pip install MDRefine
```

To activate this environment in a new terminal:

```
source /path/to/env/bin/activate
```

## 📘 Getting Started

We suggest running the tutorials in the [`Examples/`](./Examples) folder to gain insights into MDRefine's usage for:

- Refinement of molecular ensembles  
- Force field correction  
- Forward model optimization  

The `Examples/` include Jupyter notebooks and scripts demonstrating practical workflows.

---

## 🧪 Continuous Integration

This repository uses [GitHub Actions](https://github.com/bussilab/MDRefine/actions) for automatic testing across Python versions and operating systems.  
CI runs are skipped for documentation-only commits (e.g., `README.md` changes).

---

## 📚 References

MDRefine builds on recent advances in ensemble refinement. If you use this package, please cite:

- Gilardoni, I., Piomponi, V., Fröhlking, T., & Bussi, G. (2025).  
  **MDRefine: A Python package for refining molecular dynamics trajectories with experimental data**.  
  *The Journal of Chemical Physics*, 162(19).  
  [https://doi.org/10.1063/5.0256841](https://doi.org/10.1063/5.0256841)

---

## 🙋‍♀️ Questions or Feedback?

Feel free to:

- Open an [issue](https://github.com/bussilab/MDRefine/issues)  
- Contact the authors via email: `igilardo@sissa.it`