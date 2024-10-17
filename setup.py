import ast
from setuptools import setup
from pathlib import Path

def readme():
    return Path('README.md').read_text()

def extract_variable(file_path, variable_name):
    with open(file_path, 'r') as f:
        file_content = f.read()
    module = ast.parse(file_content)
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"Variable '{variable_name}' not found in {file_path}")

def version():
    return extract_variable('MDRefine/_version.py', '__version__')

def deps():
    return extract_variable('MDRefine/__init__.py', '_required_')

def description():
    with open('MDRefine/__init__.py', 'r') as f:
        file_content = f.read()
    module = ast.parse(file_content)
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            return node.value.s.split('\n')[0]
    return ""

setup(
    name="MDRefine",
    version=version(),
    author='Ivan Gilardoni',
    description=description(),
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/bussilab/MDRefine',
    packages=['MDRefine'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering"
        ],
    install_requires=deps(),
    python_requires='>=3.8',
#    scripts=['bin/mdrefine'] # command line?
)
