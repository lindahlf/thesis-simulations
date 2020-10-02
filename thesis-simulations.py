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
    p_values = np.sort(p_values) # Make sure p-values are sorted in ascending order
    n = len(p_values) # Number of data points
    ivalues = np.arange(1,n + 1)
    #p_values = p_values[0:int(round(n/2))] # Cut-off half of the values
    HC_vec = np.sqrt(n)*(ivalues/(n+1) - p_values)/np.sqrt(p_values - p_values**2) # Calculate scores for all datapoints
    return np.max(HC_vec)

def phi_test(p_values,s):
    """ Computes Jager and Wellner's (2007) phi-divergence based test.
    s = 2 corresponds to standard HC.
    """
    p_values = np.sort(p_values) #sorting in ascending order
    v = p_values # use notation of Jager and Wellner
    n = len(v) # number of datapoints
    u = np.arange(1,n+1)/(n+1)
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

def weight_ks(p_values):
    """Computes CsCsHM statistic based on results by Stepanova and Pavlenko (2018)"""
    u = np.sort(p_values) # Assert p-values are sorted, translate to notation of Stepanova and Pavlenko
    n = len(u) # Number of data points
    ivalues = np.arange(1,n+1)
    T = np.sqrt(n)*(ivalues/(n+1) - u)/np.sqrt(u*(1-u)*np.log(np.log(1/(u*(1-u)))))
    return np.max(T)

def gaussian_mixture(n,beta,r):
    """ Returns a sparse Gaussian mixture and a standard Gaussian of size n.
    Standard corresponds to the null hypothesis,
    the mixture corresponds to alternative hypothesis. See Donoho and Jin (2004)
    """
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

def calc_pvalue(data,two_side = False):
    """ Returns one-sided (default) or two-sided p-values of input array of data """
    if two_side:
        return 2*(1-norm.cdf(np.abs(data)))
    else:
        return (1 - norm.cdf(data))

def simulate_scores(n,beta,r,repetitions,method,*args):
    """ Computes scores for a given detection method for a number of repetitions. """
    null_scores = [None] * repetitions
    alt_scores = [None] * repetitions
    for i in tqdm(range(repetitions)): # Printing progress bar
        null_samples, alt_samples = gaussian_mixture(n,beta,r)
        null_p = calc_pvalue(null_samples)
        alt_p = calc_pvalue(alt_samples)
        null_scores[i] = method(null_p,*args)
        alt_scores[i] = method(alt_p,*args)
    return null_scores, alt_scores

def critical_value(n,alpha,method,*args):
    """ Simulates a threshold above which to reject the null hypothesis for a significance level alpha
        for a given method
    """
    N = int(round(50/alpha))
    iter = 100
    thresh_vec = [None] * iter
    for i in tqdm(range(iter)):
        scores = [None] * N
        for j in range(N): # N replicates averaged over 100 iterations
            null_samples = np.random.normal(0, 1, n)
            p_values = calc_pvalue(null_samples)
            scores[j] = method(p_values,*args)
        thresh_vec[i] = np.percentile(scores,100*(1-alpha))
    threshold = np.mean(thresh_vec)
    print('Method: ', str(method.__name__))
    if len(args) != 0:
        print('Parameter = ', str(args[0]))
    print('Significance level: ', str(alpha))
    print('Number of samples: ', str(n))
    print('Threshold: ', str(threshold))
    return threshold # Returns top alpha percentile of simulated scores

#TODO: write function to show sum of type I and II errors
def error_sums(n,r,beta,repetitions,threshold,verbose,method, *args):
    #check whether r or beta is array of values, then loop over that
    print(verbose)
    if hasattr(r, "__len__") and hasattr(beta, "__len__") == False:
        errors = [None] * len(r)
        for i in tqdm(range(len(r))):
            null_scores, alt_scores = simulate_scores(n,beta,r[i],repetitions,method,*args)
            # 1 if null is rejected,zero otherwise
            null_scores = np.asarray(null_scores)
            alt_scores = np.asarray(alt_scores)
            null_scores = np.where(null_scores <= threshold, null_scores, 0)
            null_scores = np.where(null_scores > threshold, null_scores, 1)
            alt_scores = np.where(alt_scores <= threshold, alt_scores, 0)
            alt_scores = np.where(alt_scores > threshold, alt_scores, 1)
            typeI = np.sum(null_scores)/repetitions
            typeII = (repetitions-np.sum(alt_scores))/(repetitions)
            print('Type I = ', str(typeI))
            print('Type II = ', str(typeII))
            errors[i] = typeI + typeII
        print(errors)
        if verbose:
            plt.plot(r,errors,'*')
            plt.show()
    elif hasattr(beta, "__len__") and hasattr(r, "__len__") == False:
        errors = [None] * len(beta)
        for i in tqdm(range(len(beta))):
            null_scores, alt_scores = simulate_scores(n,beta[i],r,repetitions,method,*args)
            null_scores = np.where(null_scores <= threshold, null_scores, 0)
            null_scores = np.where(null_scores > threshold, null_scores, 1)
            alt_scores = np.where(alt_scores <= threshold, alt_scores, 0)
            alt_scores = np.where(alt_scores > threshold, alt_scores, 1)
            errors[i] = np.sum(null_scores)/repetitions + (repetitions-np.sum(alt_scores))/(repetitions)
        if verbose:
            plt.plot(beta, errors)
            plt.show()
    return

#TODO: implement get_threshold function that returns threshold given method and sample size


################
# Main program #
################

# Example use

def main():
    n = int(1e2)
    beta = 0.8
    r = np.linspace(0.8,0.9,20)
    #critical_value(n,0.05,phi_test,2)
    error_sums(n,r,beta,100,4.68,False,HC)
    # null_scores, alt_scores = simulate_scores(n,beta,r,100,weight_ks)
    # #  null_HC, alt_HC = simulate_scores(n,beta,r,100,HC)
    # fig, axs = plt.subplots(2)
    # fig.suptitle('Simulated scores')
    # axs[0].hist(null_scores, bins = 100, range = [0,1000])
    # axs[1].hist(alt_scores, bins = 100, range = [0,1000])
    # # #axs[1].hist(null_HC, bins = 100, range = [0,100])
    # # #axs[1].hist(alt_HC, bins = 100, range = [0,100])
    # mplcyberpunk.add_glow_effects()
    # plt.show()



main()
