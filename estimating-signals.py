import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import random as rnd
import matplotlib.patches as mpatches
import json
import pandas as pd
import signal-detection as sd
from scipy.signal import savgol_filter
from scipy.stats import norm
from tqdm import tqdm

#TODO: implement method to estimate mu and epsilon
def esimate_signals(x, mu_type):
    """Function that estimates mu and epsilon of a given array of data points"""
    n = len(x)

    # Estimate mu
    abs_x = np.abs(x)
    if mu_type == 2:
        mu_values = np.where(abs_x > norm.ppf(1-1/n*np.log(2)), abs_x, 0)
    elif mu_type == 3:
        mu_values = np.where(abs_x > np.sqrt(2*np.log(n)) - (np.log(np.log(n)))/(2*np.sqrt(2*np.log(n))), abs_x, 0)
    elif mu_type == 4:
        # Fred's empirical version
        mu_values = np.where(abs_x > (np.log(2*np.log(n))), abs_x, 0)

    # Esimate epsilon
    j_vec = np.arange(1,2*np.log(n)+1)
    tvec =

    return mu_values

def emp_cdf(x,t):
    """Returns the empirical cdf for the array x and argument t"""
    n = len(x)
    x = np.sort(x)
    x = np.where(x <= t, x, 0) # set all values of x larger than t to zero
    no_of_nonzero = len(np.nonzero(x))
    return no_of_nonzero/n

######### Main ###########

def main():
    n = 1e2
    beta = 0.55
    r = 0.9

    null_samples, alt_samples = sd.gaussian_mixture(n,beta,r)


main()
