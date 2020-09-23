import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import random as rnd
from scipy.stats import norm
import matplotlib.patches as mpatches

def HC(pvalues):
    """Compute HC-statistic based on method by Donoho and Jin (2004)"""
    p_values = np.sort(p_values) # Make sure p-values are sorted
    n = len(p_values) # Number of data points

    ivalues = np.arange(1,int(round(n/2)) + 1)
    p_values = p_values[0:int(round(n/2))] # Cut-off half of the values

    HC_vec = np.sqrt(n)*(ivalues/n - p_values)/np.sqrt(p_values - p_values**2) #Compute for all

    return np.max(HC_vec)
