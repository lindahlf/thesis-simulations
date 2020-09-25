import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import random as rnd
from scipy.stats import norm
import matplotlib.patches as mpatches
from tqdm import tqdm

plt.style.use("cyberpunk")

def HC(p_values):
    """ Compute higher criticism-statistic based on version by Donoho and Jin (2004) """
    p_values = np.sort(p_values) # Make sure p-values are sorted
    n = len(p_values) # Number of data points
    ivalues = np.arange(1,int(round(n/2)) + 1)
    p_values = p_values[0:int(round(n/2))] # Cut-off half of the values
    HC_vec = np.sqrt(n)*(ivalues/n - p_values)/np.sqrt(p_values - p_values**2) # Calculate scores for all datapoints
    return np.max(HC_vec)

def phi_test(p_values,s):
    """ Computes Jager and Wellner's (2007) phi-divergence based test.
    s = 2 corresponds to standard HC. """
    p_values = np.sort(p_values)
    v = p_values # use notation of Jager and Wellner
    n = len(v) # number of datapoints
    u = np.arange(1,n+1)/n
    K = v*_phi(s,u/v) + (1-v)*_phi(s,(1-u)/(1-v)) #Calculate score for all datapoints
    return np.max(K)

def _phi(s,x):
    """Helper function for phi-divergence test"""
    if s == 1:
        return x*np.log(x) - x + 1
    elif s == 0:
        return -np.log(x) + x - 1
    else:
        return (1-s+s*x-x**s)/(s*(1-s))

# TODO: implement weighted Kolmogorov-Smirnoff statistic

def weight_ks(p_values):
    """Computes CsCsHM statistic based on results by Stepanova and Pavlenko (2018)"""
    u = np.sort(p_values) # Assert p-values are sorted, translate to notation of Stepanova and Pavlenko
    n = len(u) # Number of data points
    ivalues = np.arange(1,n+1)
    T = np.sqrt(n)*(ivalues/n - u)/np.sqrt(u*(1-u)*np.log(np.log(1/(u*(1-u)))))
    return np.max(T)

def gaussian_mixture(n,beta,r):
    """ Returns a sparse Gaussian mixture and a standard Gaussian of size n.
    Standard corresponds to the null hypothesis,
    the mixture corresponds to alternative hypothesis. See Donoho and Jin (2004)"""
    epsilon = n**(-beta)
    if beta >= 0.5:
        mu = np.sqrt(2*r*np.log(n))
    else:
        mu = n**(-r)
    null_samples = np.random.normal(0, 1, n)
    alt_samples1 = null_samples[0:round(n*(1-epsilon))]
    alt_samples2 = np.random.normal(mu, 1, round(n*epsilon))
    alt_samples = np.concatenate([alt_samples1,alt_samples2])
    assert len(null_samples) == len(alt_samples)
    return null_samples, alt_samples

def calc_pvalue(data):
    """ Returns one-sided p-value of input array of data """
    return (1 - norm.cdf(data))

def simulate_scores(n,beta,r,repetitions,func,*args):
    """ Computes scores for a given detection method for a number of repetitions. """
    null_scores = [None] * repetitions
    alt_scores = [None] * repetitions
    for i in tqdm(range(repetitions)): # Printing progress bar
        null_samples, alt_samples = gaussian_mixture(n,beta,r)
        null_p = calc_pvalue(null_samples)
        alt_p = calc_pvalue(alt_samples)
        null_scores[i] = func(null_p,*args)
        alt_scores[i] = func(alt_p,*args)
    return null_scores, alt_scores

# TODO: write function to empirically determine critical value for an alpha-level test

################
# Main program #
################

# Example use

def main():
    n = int(1e4)
    beta = 0.6
    r = 0.8
    null_scores, alt_scores = simulate_scores(n,beta,r,100,weight_ks)
    null_HC, alt_HC = simulate_scores(n,beta,r,100,HC)
    fig, axs = plt.subplots(2)
    fig.suptitle('Simulated scores')
    axs[0].hist(null_scores, bins = 100, range = [0,100])
    axs[0].hist(alt_scores, bins = 100, range = [0,100])
    axs[1].hist(null_HC, bins = 100, range = [0,100])
    axs[1].hist(alt_HC, bins = 100, range = [0,100])
    mplcyberpunk.add_glow_effects()
    plt.show()

main()
