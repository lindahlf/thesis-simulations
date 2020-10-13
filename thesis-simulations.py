import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import random as rnd
import matplotlib.patches as mpatches
import json
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import norm
from tqdm import tqdm

plt.style.use("seaborn")
#plt.style.use("cyberpunk")

def HC(p_values):
    """ Compute higher criticism-statistic based on version by Donoho and Jin (2004) """
    p_values = np.sort(p_values) # Make sure p-values are sorted in ascending order
    n = len(p_values) # Number of data points
    ivalues = np.arange(1, n + 1)
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

def CsCsHM(p_values):
    """Computes  statistic based on results by Stepanova and Pavlenko (2018)"""
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

def sim_critval(n,alpha,method,*args):
    """ Simulates a critical value above which to reject the null hypothesis for a significance level alpha
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
    crit_value = np.mean(thresh_vec)
    print('Method: ', str(method.__name__))
    if len(args) != 0:
        print('Parameter = ', str(args[0]))
    print('Significance level: ', str(alpha))
    print('Number of samples: ', str(n))
    print('Critical value: ', str(crit_value))
    return crit_value # Returns top alpha percentile of simulated scores

def plot_errorsum(n,r,beta,repetitions,verbose,method, *args):
    """Plots sum of type I and II errors as a function of either r or beta"""
    if len(args) != 0:
        print("Running simulation for " + str(method.__name__) + " with n = " + str(n) + " and parameter = " + str(args[0]))
    else:
        print("Running simulation for " + str(method.__name__) + " with n = " + str(n))
    critical_value = get_critval(n,method,*args)
    if hasattr(r, "__len__") and hasattr(beta, "__len__") == False:
        errors = [None] * len(r)
        for i in tqdm(range(len(r))):
            null_scores, alt_scores = simulate_scores(n,beta,r[i],repetitions,method,*args)
            null_scores = np.asarray(null_scores)
            alt_scores = np.asarray(alt_scores)
            # 1 if null is rejected,zero otherwise
            null_scores = np.where(null_scores > critical_value, null_scores, 0)
            null_scores = np.where(null_scores <= critical_value, null_scores, 1)
            alt_scores = np.where(alt_scores > critical_value, alt_scores, 0)
            alt_scores = np.where(alt_scores <= critical_value, alt_scores, 1)
            typeI = np.sum(null_scores)/repetitions
            typeII = (repetitions-np.sum(alt_scores))/(repetitions)
            errors[i] = typeI + typeII
        if verbose:
            plt.plot(r,errors)
            plt.xlabel(r'$r$')
            plt.ylabel('Error sum')
            if len(args) != 0:
                plt.title('Sum of type I and II error for ' + str(method.__name__) + ', s = ' + str(args[0]) )
            else:
                plt.title('Sum of type I and II error for ' + str(method.__name__))
            plt.legend((r'$n = $' + str(n) + r' $\beta = $' + str(beta),), loc = 'best')
            plt.show()
    elif hasattr(beta, "__len__") and hasattr(r, "__len__") == False:
        errors = [None] * len(beta)
        for i in tqdm(range(len(beta))):
            null_scores, alt_scores = simulate_scores(n,beta[i],r,repetitions,method,*args)
            null_scores = np.asarray(null_scores)
            alt_scores = np.asarray(alt_scores)
            null_scores = np.where(null_scores > critical_value, null_scores, 0)
            null_scores = np.where(null_scores <= critical_value, null_scores, 1)
            alt_scores = np.where(alt_scores > critical_value, alt_scores, 0)
            alt_scores = np.where(alt_scores <= critical_value, alt_scores, 1)
            errors[i] = np.sum(null_scores)/repetitions + (repetitions-np.sum(alt_scores))/(repetitions)
        if verbose:
            plt.plot(beta, errors)
            plt.xlabel(r'$\beta$')
            plt.ylabel('Error sum')
            if len(args) != 0:
                plt.title('Sum of type I and II error for ' + str(method.__name__) + ' s = ' + str(args[0]))
            else:
                plt.title('Sum of type I and II error for ' + str(method.__name__))
            plt.legend((r'$n = $' + str(n) + r' $r = $' + str(beta),), loc = 'best')
            plt.show()
    elif hasattr(beta, "__len__") and hasattr(r, "__len__"):
        # TODO: add additional elif, when both r and beta are arrays
        assert len(beta) == len(r)
        errors = [None] * len(beta)
        for i in tqdm(range(len(beta))):
            null_scores, alt_scores = simulate_scores(n,beta[i],r[i],repetitions,method,*args)
            null_scores = np.asarray(null_scores)
            alt_scores = np.asarray(alt_scores)
            null_scores = np.where(null_scores > critical_value, null_scores, 0)
            null_scores = np.where(null_scores <= critical_value, null_scores, 1)
            alt_scores = np.where(alt_scores > critical_value, alt_scores, 0)
            alt_scores = np.where(alt_scores <= critical_value, alt_scores, 1)
            errors[i] = np.sum(null_scores)/repetitions + (repetitions-np.sum(alt_scores))/(repetitions)
        if verbose:
            fig, axs = plt.subplots(2)
            fig.suptitle('Sum of type I and II errors along detection boundary')
            axs[0].plot(beta,errors)
            axs[1].plot(r,errors)
            plt.show()
    if len(args) != 0:
        print("Finished simulation for " + str(method.__name__) + " with n = " + str(n) + " and parameter = " + str(args[0]))
    else:
        print("Finished simulation for " + str(method.__name__) + " with n = " + str(n))
    return errors

def get_critval(n,method,*args):
    """ Method for extracting critical value for given method and number of samples """
    with open('critical_values.json') as file:
        data = json.load(file)
    if len(args) != 0:
        critical_value = data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]
    else:
        critical_value = data[str(method.__name__)]["n"][str(n)]
    return critical_value

def log_errorsim(filename, n, beta, r, result, method, *args):
    """Logs error simulation data for the given parameters"""
    with open(filename) as file:
        data = json.load(file)
    if len(args) != 0:
        if hasattr(r, "__len__") and hasattr(beta, "__len__") == False:
            r = r.tolist()
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["beta_fix"]["result"] = result
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["beta_fix"]["beta"] = beta
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["beta_fix"]["r"] = r
        elif hasattr(beta, "__len__") and hasattr(r, "__len__") == False:
            beta = beta.tolist()
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["r_fix"]["result"] = result
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["r_fix"]["beta"] = beta
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["r_fix"]["r"] = r
        elif hasattr(beta, "__len__") and hasattr(r, "__len__"):
            beta = beta.tolist()
            r = r.tolist()
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["result"] = result
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["beta"] = beta
            data[str(method.__name__)]["n"][str(n)]["parameter"][str(args[0])]["r"] = r
    else:
        if hasattr(r, "__len__") and hasattr(beta, "__len__") == False:
            r = r.tolist()
            data[str(method.__name__)]["n"][str(n)]["beta_fix"]["result"] = result
            data[str(method.__name__)]["n"][str(n)]["beta_fix"]["beta"] = beta
            data[str(method.__name__)]["n"][str(n)]["beta_fix"]["r"] = r
        elif hasattr(beta, "__len__") and hasattr(r, "__len__") == False:
            beta = beta.tolist()
            data[str(method.__name__)]["n"][str(n)]["r_fix"]["result"] = result
            data[str(method.__name__)]["n"][str(n)]["r_fix"]["beta"] = beta
            data[str(method.__name__)]["n"][str(n)]["r_fix"]["r"] = r
        elif hasattr(beta, "__len__") and hasattr(r, "__len__"):
            beta = beta.tolist()
            r = r.tolist()
            data[str(method.__name__)]["n"][str(n)]["result"] = result
            data[str(method.__name__)]["n"][str(n)]["beta"] = beta
            data[str(method.__name__)]["n"][str(n)]["r"] = r

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
    return

################
# Main program #
################

# Example use

def main():
    n = int(1e2)
    ######## Plot with beta fixed ######
    beta = 0.8
    r = np.linspace(0.01,0.9,100)

    ######## Plot on detection boundary ######
    # beta1 = np.linspace(0.5,0.75,100)
    # beta2 = np.linspace(0.75,1,100)
    # r1 =  beta1 - 0.5
    # r2 = (1-np.sqrt(1-beta2))**2
    # beta = np.concatenate([beta1,beta2])
    # r = np.concatenate([r1,r2])

    method = phi_test
    param = 2
    # result = plot_errorsum(n,r,beta,100,False,method,param)
    # log_errorsim('r_and_beta.json',n,beta,r,result,method,param)

    #plot_errorsum(n,r,beta,100,True,phi_test,2)

    width = 1.2 # set linewidth
    # Extract results and plot them
    with open('beta0_8.json') as file:
        data = json.load(file)
    HC_results = data["HC"]["n"][str(n)]["beta_fix"]["result"]
    cscshm_results = data["CsCsHM"]["n"][str(n)]["beta_fix"]["result"]
    phi0_results = data["phi_test"]["n"][str(n)]["parameter"]["0"]["beta_fix"]["result"]
    phi1_results = data["phi_test"]["n"][str(n)]["parameter"]["1"]["beta_fix"]["result"]
    phi2_results = data["phi_test"]["n"][str(n)]["parameter"]["2"]["beta_fix"]["result"]

    plt.plot(r,HC_results, linewidth=width)
    plt.plot(r,cscshm_results, linewidth=width)
    # plt.plot(r,phi0_results)
    # plt.plot(r,phi1_results)
    plt.plot(r,phi2_results, linewidth=width)
    plt.xlabel(r'$r$')
    plt.ylabel('Error sum')
    plt.legend(("HC","CsCsHM",r"$\varphi$, s = 2"))
    #plt.legend(("HC","CsCsHM",r"$\varphi$, s = 0", r"$\varphi$, s = 1", r"$\varphi$, s = 2"))
    plt.title(r'Sum of type I and II errors as a function of $r$, $\beta$ = ' + str(beta) + r", $n$ = " + str(n))
    #figname = "beta0_8"  + "n" + str(n) + ".eps"
    #plt.savefig(figname, format = 'eps', dpi = 1200)
    plt.show()

    # #idx = range(0,len(r))
    # idx = range(0,round(len(r)/4))
    # idx = 4*np.asarray(idx)
    # # Extract detection boundary results and plot them
    # with open('r_and_beta.json') as file:
    #     data = json.load(file)
    # HC_results = np.asarray(data["HC"]["n"][str(n)]["result"])
    # cscshm_results = np.asarray(data["CsCsHM"]["n"][str(n)]["result"])
    # # phi0_results = data["phi_test"]["n"][str(n)]["parameter"]["0"]["result"]
    # # phi1_results = data["phi_test"]["n"][str(n)]["parameter"]["1"]["result"]
    # phi2_results = np.asarray(data["phi_test"]["n"][str(n)]["parameter"]["2"]["result"])
    # plt.figure(1)
    # plt.plot(r[idx],HC_results[idx], linewidth=width)
    # plt.plot(r[idx],cscshm_results[idx], linewidth=width)
    # # plt.plot(r,phi0_results)
    # # plt.plot(r,phi1_results)
    # plt.plot(r[idx],phi2_results[idx], linewidth=width)
    # plt.xlabel(r'$r$')
    # plt.ylabel('Error sum')
    # #plt.legend(("HC","CsCsHM",r"$\varphi$, s = 2"))
    # #plt.legend(("HC","CsCsHM",r"$\varphi$, s = 0", r"$\varphi$, s = 1", r"$\varphi$, s = 2"))
    # plt.title(r'Sum of type I and II errors as a function of $r$ along detection boundary' + r", $n$ = " + str(n))
    # fig1 = 'r_errorsum_detbound_n' + str(n) + '.eps'
    # #plt.savefig(fig1, format = 'eps', dpi = 1200)
    # plt.figure(2)
    # plt.plot(beta[idx],HC_results[idx], linewidth=width)
    # plt.plot(beta[idx],cscshm_results[idx], linewidth=width)
    # # plt.plot(r,phi0_results)
    # # plt.plot(r,phi1_results)
    # plt.plot(beta[idx],phi2_results[idx], linewidth=width)
    # plt.xlabel(r'$\beta$')
    # plt.ylabel('Error sum')
    # plt.legend(("HC","CsCsHM",r"$\varphi$, s = 2"))
    # #plt.legend(("HC","CsCsHM",r"$\varphi$, s = 0", r"$\varphi$, s = 1", r"$\varphi$, s = 2"))
    # plt.title(r'Sum of type I and II errors as a function of $\beta$ along detection boundary' + r", $n$ = " + str(n))
    # fig2 = 'beta_errorsum_detbound_n' + str(n) + '.eps'
    # #plt.savefig(fig2, format = 'eps', dpi = 1200)
    # plt.show()



    # null_scores, alt_scores = simulate_scores(n,beta,r,100,CsCsHM)
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
