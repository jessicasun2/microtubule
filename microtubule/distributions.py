import itertools
import warnings

import numpy as np
import pandas as pd
import math
import itertools
import warnings

import scipy.optimize
import scipy.stats as st

import bebi103

import tqdm


beta_2 = [0.25, 1, 2, 5, 10]
beta_1 = 1

def simulate_s_poisson(beta1, beta2):
    # random number generator
    rg = np.random.default_rng(seed=93)
    process_1 = rg.exponential(1/beta_1, size=150)

    process = []
    for i in range(len(beta_2)):
        pro_2 = rg.exponential(1/beta_2[i], size=150)
        process.append(process_1 + pro_2)
    return process

def cdf_func(time, beta_1, beta_2):
    '''Calculate the CDF of the successive Poisson distribution'''
    cdf = []
    for i in range(len(time)):
        t = time[i]
        coeff = beta_1*beta_2/(beta_2 - beta_1)
        exp = (1/beta_1)*(1 - math.exp(-beta_1 * t)) - (1/beta_2)*(1 - math.exp(-beta_2 * t))
        value = coeff * exp
        cdf.append(value)
    return cdf

def log_like_gamma(params, t):
    '''Log-likelihood function for gamma distribution'''

    alpha, beta = params
    
    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(t, alpha, 0, 1/beta))

def log_like_gamma_log_params(log_params, n):
    """Log likelihood for Gamma distribution with input being logarithm of parameters."""
    log_a, log_b = log_params

    alpha = np.exp(log_a)
    beta = np.exp(log_b)

    return np.sum(st.gamma.logpdf(n, alpha, 0, 1/beta))

def guess_params(n):
    n_mean = np.mean(n)
    n_var = np.var(n)/(len(n) - 1)
    beta_i = n_mean/n_var
    alpha_i = ((n)**2)/n_var
    return (alpha_i, beta_i)

def mle_gamma(n):
    '''Function to compute the MLE'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    alpha_i, beta_i = guess_params(n)

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_gamma(params, n),
            x0=np.array([alpha_i, beta_i]),
            args=(n,),
            method='Powell'
        )
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

def mle_log_gamma(n):
    '''Function to compute the MLE'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    alpha_i, beta_i = guess_params(n)

        res2 = scipy.optimize.minimize(
            fun=lambda log_params, labeled: -log_like_gamma_log_params(log_params, labeled),
            x0=np.array([np.log(alpha_i), np.log(beta_i)]),
            args=(labeled,),
            method='BFGS'
        )
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def log_like_poisson_2(params, t):
    """
    Log likelihood function for successive Poisson processes
    
    params: beta1, beta2 which are rates of arrivals of those processes
    
    As beta1 ~= beta2, the distribution is a gamma distribution
    """

    beta_1, beta_2 = params
    
    # ensure the rates are not zero or negative and that beta_1 > beta_2 for our log sum exp trick to work
    if beta_1 <= 0 or beta_2 <= 0 or beta_1 > beta_2:
        return -np.inf
    
    n = len(t)
    
    # if they are close, it's the gamma distribution
    if abs(beta_1 - beta_2) < 1e-6:
        alpha = 2
        fun = log_like_gamma((alpha, beta_1), t)
    
    # if not, it is the log liklihood function of successive Poisson processes
    else:
        x = -beta_1*t
        y = -beta_2*t
        fun = np.log((beta_1*beta_2)/(beta_2 - beta_1)) + x + np.log((1 - np.exp(y-x)))
        fun = fun.sum()

    return fun

def mle_poisson_2(t):
    '''Function to compute the MLE'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like_poisson_2(params, t),
            x0=np.array([0.003, 0.006]),
            args=(t,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def AIC_df(data)       
    # make a dataframe to store all the data
    df_mle = pd.DataFrame(index=['alpha', 'beta', 'b1', 'b2'])

    # data set
    t = data

    # put in the paramters found from the MLE of each distribution
    # Gamma
    alpha, beta = mle_gamma(t)
    # Successive Poisson
    b1, b2 = mle_poisson_2(t)

    # Store results in data frame
    df_mle['12 uM'] = [alpha, beta, b1, b2]

    # Find the log liklihood using function written above for each distribution
    ell = log_like_gamma((alpha, beta), t)
    df_mle.loc["log_like_gamma"] = ell

    ell_p = log_like_poisson_2((b1, b2), t)
    df_mle.loc["log_like_poisson_2"] = ell_p

    # AIC for each distribution based on definition of AIC
    df_mle.loc['AIC_gamma'] = -2 * (df_mle.loc['log_like_gamma'] - 2)
    df_mle.loc['AIC_poisson_2'] = -2 * (df_mle.loc['log_like_poisson_2'] - 2)

    # Akaike weight for each distribution based on definition
    AIC_max = max(df_mle.loc['AIC_gamma'].values, df_mle.loc['AIC_poisson_2'].values)
    numerator = np.exp(-(df_mle.loc['AIC_gamma'] - AIC_max)/2)
    denominator = numerator + np.exp(-(df_mle.loc['AIC_poisson_2'] - AIC_max)/2)
    df_mle.loc['w_gamma'] = numerator / denominator
    df_mle.loc['w_poisson_2'] = 1- (numerator / denominator)

    # Take a look
    return df_mle