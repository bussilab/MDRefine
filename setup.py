from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

def version():
    from MDRefine import __version__ as version
    return version

def deps():
    from MDRefine import _required_ as required
    return required

def description():
    from MDRefine import __doc__ as doc
    return doc.partition('\n')[0]

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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering"
        ],
    install_requires=deps(),
    python_requires='>=3.9',
#    scripts=['bin/mdrefine'] # command line?
)
