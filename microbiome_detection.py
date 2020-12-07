from biom import load_table
from signal_detection import *
from signal_estimation import *
from scipy.stats import norm
from scipy.stats import mstats
from scipy import stats
from fitter import Fitter
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm #sm.qqplot(trans, line='s')
import seaborn as sns



#df_new.to_csv('reduced_df.csv', sep='\t')
#np.savetxt("trans_var.csv", new_var, delimiter=",")

def HC_update(p_values, alpha):
    """ Compute higher criticism-statistic based on version by Donoho and Jin (2004) """
    p_values = np.sort(p_values) # Make sure p-values are sorted in ascending order
    n = len(p_values) # Number of data points
    ivalues = np.arange(1, n + 1)
    #p_values = p_values[0:int(round(n/2))] # Cut-off half of the values
    HC_vec = np.sqrt(n)*(ivalues/(n+1) - p_values)/np.sqrt(p_values - p_values**2) # Calculate scores for all datapoints
    HC_vec_reduced = HC_vec[0:int(alpha*(len(HC_vec)-1))]
    max_idx = np.argmax(HC_vec_reduced)
    return HC_vec_reduced[max_idx], max_idx, HC_vec_reduced

def CsCsHM_update(p_values, alpha):
    """Computes CsCsHM statistic based on results by Stepanova and Pavlenko (2018)"""
    u = np.sort(p_values) # Assert p-values are sorted, translate to notation of Stepanova and Pavlenko
    n = len(u) # Number of data points
    ivalues = np.arange(1,n+1)
    T = np.sqrt(n)*(ivalues/(n+1) - u)/np.sqrt(u*(1-u)*np.log(np.log(1/(u*(1-u)))))
    T_red = T[0:int(alpha*(len(T)-1))]
    max_idx = np.argmax(T_red)
    return T_red[max_idx], max_idx, T_red

def indep_transform(data):
    """ Transforms a data matrix with dirichlet distributed data and returns a set
    of mutually independent standard \beta-distributed random variables """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    n,p = np.shape(data)
    for i in range(p):
        foo_sum = np.sum(data[:,i:p],axis = 1)
        data[:,i] = np.divide(data[:,i],foo_sum)
    return data

def remove_zero_features(df,no_zeros = 1):
    """ Removes columns from pandas dataframe with given amount of nonzero elements"""
    thing = df.astype(bool).sum(axis=0) # number of nonzeros in each column
    idx = pd.Index(thing) #Index format
    location = idx.get_loc(no_zeros) # Set all elements that are 1.0 to True, rest to False.
    loc_of_one = np.asarray(np.nonzero(location)) #Array of columns with only one nonzero element
    loc_of_one = loc_of_one[0]
    df_new = df.drop(df.columns[loc_of_one], axis = 1) # New reduced dataframe
    return df_new

def norm_data(df):
    """ Normalizes each sample in dataframe by number of counts for respective sample """
    cols = df.columns
    sum = df.sum(axis=1)
    df_new = df.loc[:,cols[1]:cols[-1]].div(sum, axis=0)
    return df_new

def clr_trans(data):
    """Performs center log-ratio transformation on dataframe of count data.
     Replaces zeros with small constant"""
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    data[data == 0] = 1/len(data[0,:])**2 # replace zeros with small constant
    geometric_mean = mstats.gmean(data,axis=1)
    clr_data = np.log(data / geometric_mean[:,None])
    return clr_data

def ks_pval(data):
    """Returns p-values based on Kolmogorov-Smirnoff test for beta distribution"""
    n,p = np.shape(data)
    pvals = [None] * p
    for i in range(p):
        foo, pvals[i] = stats.kstest(data[:,i], "beta", args = (1,p-i))
    return pvals

def clt_norm(data):
    """Normalizes 1D vector of data according to central limit theorem using mean and std"""
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    mean = np.mean(data)
    std = np.sqrt(np.var(data,ddof = 1))
    return (data - mean)/std

def adaptive_selection(p_values, data, alpha):
    """Runs HC and CsCsHM adaptive procedures for given p-values."""
    HC_stat, max_idx_HC, HC_vec = HC_update(p_values, alpha)
    cscshm_stat, max_idx_cscshm, cscshm_vec = CsCsHM_update(p_values, alpha)
    thresh_HC = data[max_idx_HC]
    thresh_cscshm = data[max_idx_cscshm]
    nonnull_HC = np.where(data < thresh_HC, 0, data)
    nonnull_cscshm = np.where(data < thresh_cscshm, 0, data)
    return nonnull_HC, nonnull_cscshm

def dirichlet_variance(alpha):
    """Computes variance of dirichlet distribution given parameter vector alpha"""
    sum = np.sum(alpha)
    var = (alpha*(sum-alpha))/((1+sum)*sum**2)
    return var

def import_data(file,remove_zero = True):
    """Imports microbiome dataset, with an option to remove features with mainly zeros"""
    df = pd.read_csv(file, sep='\t')

    if remove_zero == True:
        df = remove_zero_features(df)
    headers = df.columns
    df = df.loc[:,headers[1]:headers[-1]]
    return df, headers[1:len(headers)+1]


def main():
    df, headers = import_data('parkinsons_table.tsv')
    df_clr = clr_trans(df)

    # df_foo = df_red.to_numpy()
    # n,p = np.shape(df_foo)
    # df_red = df_red.replace(0,1/p**2) # replace zeros with small constants
    #df_norm = norm_data(df_clr)

    var = np.var(df_clr, axis = 0, ddof = 1)
    p = len(var)
    var_norm = clt_norm(var)
    var_pval = calc_pvalue(var_norm)

    nonnull_HC, nonnull_cscshm = adaptive_selection(var_pval,var_norm, 0.1)
    #ks_pvalues = ks_pval(indep_data)
    #np.savetxt("norm_data.csv", df_norm.to_numpy(), delimiter=",")

    diff_abundant_features = headers[np.nonzero(nonnull_cscshm)]
    np_diff_abun = diff_abundant_features.to_numpy()
    np.savetxt('diff_abundant_features.csv', np_diff_abun, fmt = '%s', delimiter='\n')
    # fig, axs = plt.subplots(2, sharex=True)
    # axs[0].hist(np.delete(var_norm,np.nonzero(nonnull_cscshm)), bins = 100)
    # axs[1].hist(var_norm[np.nonzero(nonnull_cscshm)])
    # plt.show()
    return

main()
