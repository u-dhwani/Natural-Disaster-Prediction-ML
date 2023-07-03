from setuptools import setup

DEPENDENCIES = [
    "joblib",
    "numpy",
    "pandas==0.23.4",
    "scipy",
    "scikit-learn==0.21.2",
    "tqdm",
    "rfpimp==1.3.4",
    "catboost==0.15",
    "deap==1.2.2",
    "librosa==0.6.3",
    "numba==0.48",
    "tsfresh==0.11.2",
]

setup(
    name='earthquake',
    packages=['earthquake'],
    install_requires=DEPENDENCIES,
    python_requires="==3.7.9"
)