import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import random as rnd
import matplotlib.patches as mpatches
import json
import pandas as pd
import tikzplotlib
from statsmodels.distributions.empirical_distribution import ECDF
from signal_detection import *
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.stats import uniform
from tqdm import tqdm

def estimate_mu(x, mu_type):
    """Function that estimates mu of a given array of data points.
    Based on procedure by DasGupta et al. (2014)"""
    n = len(x)
    # Estimate mu
    abs_x = np.abs(x)
    if mu_type == 2:
        mu_values = np.where(abs_x > norm.ppf(1-1/n*np.log(2)), abs_x, 0)
    elif mu_type == 3:
        mu_values = np.where(abs_x > np.sqrt(2*np.log(n)) - (np.log(np.log(n)))/(2*np.sqrt(2*np.log(n))), abs_x, 0)
    return mu_values

def emp_cdf(x,t):
    """Returns the empirical cdf for the array x and argument t"""
    n = len(x)
    x = np.sort(x)
    if hasattr(t, "__len__"):
        ecdf = ECDF(x)
        return ecdf(t)
    else:
        x = np.where(x <= t, x, 0) # set all values of x larger than t to zero
        no_of_nonzero = len(np.nonzero(x)[0])
        return no_of_nonzero/n

def a_value(n,alpha):
    """Returns empirical value for a_n as specified in Cai, Jin and Low (2007)"""
    t = np.linspace(0.5,norm.cdf(np.sqrt(2*np.log(n))))
    iter = 100000
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

def estimate_epsilon(x,t,a_n,est_mu):
    """ Returns estimate of epsilon (fraction of non-null signals) for given dataset x """
    F_plus = F_conf(x,t,a_n)
    cdfs = norm.cdf(t)
    eps_vec = (cdfs - F_plus)/(cdfs - norm.cdf(t - est_mu))
    eps_vec = eps_vec[np.nonzero(est_mu)] # Vector of possible values for epsilon
    if len(eps_vec) == 0:
        # If no non-zero mu's are detected then epsilon clearly is zero
        estimated_eps = 0
    else:
        estimated_eps = np.max(eps_vec)
    return estimated_eps

######### Main ###########

def main():
    n = int(1e2)
    beta = 0.7
    r = 5
    alpha = 0.05 # significance level

    #a_n = a_value(n,alpha) # Calculate value for a_n
    if n == 1e2:
        a_n = 2.8979985601707705 # for n = 1e2, 100000 iterations
    elif n == 1e4:
        a_n = 3.062694335688579 # for n = 1e4, 100000 iterations
    elif n == 1e6:
        a_n = 3.4200701843822388 # for n = 1e6, 100000 iterations
    else:
        a_n = a_value(n,alpha)


    t = np.linspace(0,np.sqrt(2*np.log(n)), num = n) #grid points

    ###### Estimate eps and mu 'iter' number of times ######
    iter = 10000
    eps_est_vec = [None] * iter
    easyeps_est_vec = [None] * iter
    mu_est_vec = np.empty((n,iter))
    altmu_est_vec = [None] * iter
    for i in tqdm(range(iter)):
        null_samples, alt_samples = gaussian_mixture(n,beta,r)
        est_mu = estimate_mu(alt_samples,3)
        mu_est_vec[:,i] = est_mu
        altmu_est_vec[i] = np.mean(est_mu[np.nonzero(est_mu)])
        est_eps = estimate_epsilon(alt_samples,t,a_n,est_mu)
        eps_est_vec[i] = est_eps
        easyeps_est_vec[i] = len(np.nonzero(est_mu)[0])/n
    ########################################################

    ########### Code block for generating plots of estimated values of mu and epsilon
    no_bins = 100
    # plt.figure(1)
    # plt.hist(mu_est_vec[np.nonzero(mu_est_vec)], bins = no_bins)
    # plt.axvline(np.sqrt(2*r*np.log(n)), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\mu$")
    # plt.title(r'Spread of all estimated $\hat{\mu}$, $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ') ')
    # plt.xlabel(r'$\hat{\mu}$')
    # plt.legend()
    # figname1 = "figures/est_mu_n_" + str(n) + "_r_" + str(r).replace('.','_') + ".tex"
    # tikzplotlib.save(figname1)
    #
    # plt.figure(2)
    # plt.hist(eps_est_vec, bins = no_bins)
    # plt.axvline(n**(-beta), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\varepsilon$")
    # plt.title(r'Spread of all estimated $\hat{\varepsilon}$ using method by Cai et al., $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ') ')
    # min_xlim, max_xlim = plt.xlim()
    # plt.xlabel(r'$\hat{\varepsilon}$')
    # plt.legend()
    # figname2 = "figures/est_eps_n_" + str(n) + "_r_" + str(r).replace('.','_') + ".tex"
    # tikzplotlib.save(figname2)
    #
    # plt.figure(3)
    # plt.hist(easyeps_est_vec, bins = 50, range = [min_xlim, max_xlim])
    # plt.axvline(n**(-beta), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\varepsilon$")
    # plt.title(r'Spread of all estimated $\hat{\varepsilon}$ using easy method, $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ') ')
    # plt.xlabel(r'$\hat{\varepsilon}$')
    # plt.legend()
    # figname3 = "figures/easy_est_eps_n_" + str(n) + "_r_" + str(r).replace('.','_') + ".tex"
    # tikzplotlib.save(figname3)
    #
    # plt.figure(4)
    # plt.hist(altmu_est_vec, bins = no_bins)
    # plt.axvline(np.sqrt(2*r*np.log(n)), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\mu$")
    # plt.title(r'Mean of $\hat{\mu}$ for each iteration, $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ')')
    # plt.xlabel(r'$\hat{\mu}$')
    # plt.legend()
    # figname4 = "figures/alt_est_mu_n_" + str(n) + "_r_" + str(r).replace('.','_') + ".tex"
    # tikzplotlib.save(figname4)

    #plt.figure(5)
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(r'Estimates of $\varepsilon$')
    axs[0].hist(eps_est_vec, bins = no_bins)
    axs[0].axvline(n**(-beta), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\varepsilon$")
    axs[0].set_title(r'Spread of all estimated $\hat{\varepsilon}$ using method by Cai et al., $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ') ')
    #axs[0].xlabel(r'$\hat{\varepsilon}$')
    axs[0].legend()
    axs[1].hist(easyeps_est_vec, bins = 50)
    axs[1].axvline(n**(-beta), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\varepsilon$")
    axs[1].set_title(r'Spread of all estimated $\hat{\varepsilon}$ using easy method, $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ') ')
    #axs[1].xlabel(r'$\hat{\varepsilon}$')
    axs[1].legend()
    figname5 = "figures/all_eps_est" + str(n) + "_r_" + str(r).replace('.','_') + ".tex"
    tikzplotlib.save(figname5)

    fig2, axs2 = plt.subplots(2, sharex=True)
    fig.suptitle(r'Estimates of $\varepsilon$')
    axs2[0].hist(mu_est_vec[np.nonzero(mu_est_vec)], bins = no_bins)
    axs2[0].axvline(np.sqrt(2*r*np.log(n)), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\mu$")
    axs2[0].set_title(r'Spread of all estimated $\hat{\mu}$, $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ') ')
    #axs[0].xlabel(r'$\hat{\varepsilon}$')
    axs2[0].legend()
    axs2[1].hist(altmu_est_vec, bins = no_bins)
    axs2[1].axvline(np.sqrt(2*r*np.log(n)), color='k', linestyle='dashed', linewidth=1, label=r"True value of $\hat{\mu}$")
    axs2[1].set_title(r'Mean of $\hat{\mu}$ for each iteration, $(\beta,r,n) = $(' + str(beta) + ', '+ str(r) + ', '+str(n) + ')')
    axs2[1].set(xlabel = r'$\hat{\mu}$')
    axs2[1].legend()
    figname6 = "figures/all_mu_est" + str(n) + "_r_" + str(r).replace('.','_') + ".tex"
    tikzplotlib.save(figname6)

    plt.show()







main()
