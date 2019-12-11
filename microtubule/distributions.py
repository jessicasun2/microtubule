import numpy as np


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