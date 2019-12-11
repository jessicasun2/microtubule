import numpy as np
import pandas as pd
import math

def ecdf_vals(data):
    '''Return the ECDF values for values of x in a given data in the form of an array'''
    # Find total length of the data
    n = len(data)
    
    # Initialize an array to store the x and y values we get from the data
    x_y_values = np.zeros((n, 2))
    
    # loop through the data and store the value as an x value
    # find the fraction of data points that are less than or equal and add it to the array
    for i in range(n):
        x_y_values[i, 0] = data[i]
        y = (len(data[(data <= data[i])]))/n
        x_y_values[i, 1] = y
        
    return x_y_values

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

def draw_bs_reps_mean(data, size):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

def conf_int_mean(data):
    '''Find the confidence interval of the mean for a sample by drawing bootstrapped samples'''
    bs_reps_data = draw_bs_reps_mean(data, size=10000)
    mean_data_conf_int = np.percentile(bs_reps_data, [2.5, 97.5])
    return mean_data_conf_int

def test_stat_mean(data1, data2):
    '''Use mean as the test statistic to see how related two datasets are by returning a p-value'''
    n = len(data1)
    m = len(data2)
    count = 0
    # join all the data
    all_data = np.concatenate((data1, data2), axis=0)
    
    data1_mean = []
    data2_mean = []
    for i in range(10000):
        shuffle = np.random.permutation(all_data)
        data1_mean.append(np.mean(shuffle[:n]))
        data2_mean.append(np.mean(shuffle[len(all_data)-m:]))
        
    actual_diff = np.abs(np.mean(data1) - np.mean(data2))
    new_diffs = []
    for i in range(len(data1_mean)):
        new_diff = np.abs(np.mean(data1_mean[i]) - np.mean(data2_mean[i]))
        new_diffs.append(new_diff)
        if new_diff >= actual_diff:
            count +=1 
    p_value = count/10000
    print('The p-value of the mean test statistic is {}'.format(p_value))
    return p_value

def conf_int_CLT(data):
    '''Calculate the confidence intervals using the CLT'''
    z = 1.96
    data_mean = np.mean(data)
    data_var = np.var(data)/(len(data) -1)
    lower_data = data_mean - z*np.sqrt(data_var)
    upper_data = data_mean + z*np.sqrt(data_var)
    conf_int = [lower_data, upper_data]
    return conf_int

def ecdf(x, data):
    '''Computes the value of the ECDF built from a 1D array, data at arbitrary points x'''
    n = len(data)
    
    # Initialize an array to store the y values we get from the data
    y_values = np.zeros(n)

    # find the fraction of data points that are less than or equal to the x value and add it to the array
    for i in range(len(x)):
        num = data[data <= x[i]]
        value = len(num) / n 
        y_values[i] = value
    
    return y_values

# get epsilon
def get_epsilon(alpha, data):
    '''Finds the epsilon for the DKW inequality given alpha and the data set'''
    n = len(data)
    epsilon = np.sqrt((np.log(2/alpha))/(2*n))
    return epsilon

def DKW_inequality(alpha, x, data):
    '''Finds the lower and upper bounds of the confidence interval using the DKW inequality'''
    epsilon = get_epsilon(alpha, data)
    
    lower = []
    upper = []
    
    ecdf_vals = ecdf(x, data)
    
    for i in range(len(ecdf_vals)):
        low = np.maximum(0, (ecdf_vals[i] - epsilon))
        high = np.minimum(1, (ecdf_vals[i] + epsilon))
        lower.append(low)
        upper.append(high)
    
    return lower, upper