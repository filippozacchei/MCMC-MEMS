import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde
import time
import tensorflow as tf

# Adjust system path for local module imports
sys.path.append('../SurrogateModeling/')
sys.path.append('../utils/preprocessing')

def least_squares_optimization(y_observed, forward_model, start_point, bounds, l2_coeff = 0.0):
    """
    Performs least squares optimization to fit the model to the observed data.
    """
    result = least_squares(objective_function, start_point, args=([y_observed, forward_model, l2_coeff]), bounds=bounds, jac='3-point')
    return result.x, np.linalg.inv(result.jac.T @ result.jac)

def objective_function(input, exact_outputs, forward_model, l2_coeff = 0.0):
    predicted_outputs = forward_model(input).reshape(exact_outputs.shape)
    residuals = (predicted_outputs - exact_outputs)**2 + l2_coeff* np.linalg.norm(input,2)**2
    return np.sqrt(residuals)


