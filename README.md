# MCMC-MEMS

## Overview

MCMC-MEMS is a comprehensive library designed for parameter identification in Micro-Electro-Mechanical Systems (MEMS) accelerometer devices, employing a computational Bayesian approach. This project aims to bridge the gap between theoretical mechanics and practical applications in MEMS by providing tools for high-fidelity simulations, real-time digital twins, and data-driven predictions.

### Key Features

- **Mechanics of Micro Systems**: Develop high-fidelity benchmarks for in-depth analysis and understanding of MEMS dynamics.
- **Reduced Order Modelling**: Facilitate the creation of real-time digital twins of MEMS devices, enabling efficient simulations and analyses.
- **Computational Statistics**: Leverage advanced statistical methods to make accurate data-driven predictions using the digital twin framework.

## Repository Structure

### `DOC`
The `DOC` directory houses the project documentation. It is currently under development and will include user guides, API references, and examples.

### `src`
Contains the core codebase of the project, organized into the following subdirectories:

- **InverseProblems**: Includes utilities for parameter identification, featuring least square optimization methods, CUQIpy wrappers (`utils.py`), and visualization tools (`postprocessing.py`).
- **SurrogateModeling**: Contains utilities for model definition (`model.py`), model training (`training.py`), and generating plots (`postprocessing.py`).
- **utils**: Provides preprocessing utilities (`preprocessing.py`).

### `tests`
Contains test cases and examples demonstrating the usage of the repository in various scenarios.

## Installation

To set up the MCMC-MEMS environment, you will need the following dependencies:

```bash
pip install cuqipy==0.8.0 jupyter==1.0.0 matplotlib==3.5.3 numpy==1.24.4 \
    pandas==2.1.4 scipy==1.8.1 scikit-learn==1.0.2 sphinx==7.2.6 \
    sqlite==3.44.2 tensorflow==2.7.1
```

Ensure you have python 3.9.15 version.