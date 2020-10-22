import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import random as rnd
import matplotlib.patches as mpatches
import json
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from signal_detection import *
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.stats import uniform
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


    return mu_values

def emp_cdf(x,t):
    """Returns the empirical cdf for the array x and argument t"""
    n = len(x)
    x = np.sort(x)
    if hasattr(t, "__len__"):
        ecdf = ECDF(x)
        return ecdf(t)
        # no_of_nonzero = [None] * len(t)
        # for i in tqdm(range(0,len(t))):
        #     x_dummy = np.where(x <= t[i], x, 0)
        #     no_of_nonzero[i] = len(np.nonzero(x_dummy)[0])/n
        # return np.asarray(no_of_nonzero)
    else:
        x = np.where(x <= t, x, 0) # set all values of x larger than t to zero
        no_of_nonzero = len(np.nonzero(x)[0])
        return no_of_nonzero/n

def a_value(n,alpha):
    """Returns empirical value for a_n as specified in Cai, Jin and Low (2007)"""
    t = np.linspace(0.5,norm.cdf(np.sqrt(2*np.log(n))))
    iter = 1000
    W_values = [None] * iter
    for i in tqdm(range(0,iter)):
        unif = np.random.uniform(size = n)
        V = emp_cdf(unif,t)
        U_vec = np.sqrt(n)*(V - t)
        W = np.abs(U_vec)/np.sqrt(t*(1-t))
        W_values[i] = np.max(W)

    a_n = np.percentile(W_values,(1-alpha)*100)
    return a_n

def F_conf(x,t,a_n):
    """Returns F_plus as defined in Cai, Jin and Low (2007)

    Parameters:
    x (np.array): Datapoints of mixture we want to esimate
    t (np.array): Grid points used for estimation
    a_n (float): parameter based on significance level alpha

    Returns:
    F_plus (np.array): Function value of F_plus

    """
    n = len(x)
    F = emp_cdf(x,t)

    F_plus = (2*F + a_n**2/n + np.sqrt(a_n**2/n + (4*F - 4*np.power(F,2)))*(a_n/np.sqrt(n)))/(2*(1+a_n**2/n))
    return F_plus

######### Main ###########

def main():
    n = int(1e2)
    beta = 0.55
    r = 0.9
    alpha = 0.05 # significance level
    null_samples, alt_samples = gaussian_mixture(n,beta,r)


    a_n = a_value(n,alpha)
    print(a_n)
    #a_n = 2.938844666846772 # for n = 1e2
    # a_n = 3.4200701843822388 # for n = 1e6
    t = np.linspace(0,np.sqrt(2*np.log(n)), num = n)

    mu = esimate_signals(alt_samples,2)

    F_plus = F_conf(alt_samples,t,a_n)
    cdfs = norm.cdf(t)
    eps_vec = (cdfs - F_plus)/(cdfs - norm.cdf(t - mu))
    eps_vec = eps_vec[np.nonzero(mu)]
    estimated_eps = np.max(eps_vec)
    print("True eps = " + str(n**(-beta)))
    print("Estimated eps = " + str(estimated_eps))

main()
