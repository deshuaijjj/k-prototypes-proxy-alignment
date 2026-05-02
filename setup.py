from setuptools import find_packages, setup

setup(
    name="dmkpo-repro-package",
    version="1.0.0",
    description="Reproducible experiment package for DMKPO K-Prototypes benchmarks",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "kmodes>=0.12.0",
        "pandas>=1.3.0",
        "pyarrow>=8.0.0",
        "ucimlrepo>=0.0.3",
        "optuna>=3.0.0",
        "matplotlib>=3.4.0",
    ],
)
