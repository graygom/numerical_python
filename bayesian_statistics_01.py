#
# TITLE: Bayesian statistics
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: 赤石雅典, Pythonでスラスラわかるベイズ推論「超」入門, 講談社, 2023-11-24
#
#

import numpy as np
import scipy.special as scsp
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

#-----------------------------------------------------
# 1.1 discrete distribution
#-----------------------------------------------------

if False:
    n = 5
    p = 0.5
    x = range(n+1)
  
    # distribution function > binomial distribution, distribution of binary data
    y = [scsp.comb(n, i) * p**i * (1-p)**(n-i) for i in x]

    # plot
    plt.bar(x, y)
    plt.grid(ls=':')
    plt.title('visualization of probabilty distribution (n=5)')



-----------------------------------------------------
# 1.2 continuos distribution
#-----------------------------------------------------

if False:
    n = 1000
    p = 0.5
    x = range(n+1)

    # distribution function > binomial distribution, distribution of binary data
    y = [scsp.comb(n, i) * p**i * (1-p)**(n-i) for i in x]

    # plot
    plt.bar(x, y)
    plt.xlim((430, 570))
    plt.grid(ls=':')
    plt.title('visualization of probabilty distribution (n=1000)')



#-----------------------------------------------------
# 1.3 normal distribution
#-----------------------------------------------------

if False:
    # normal distribution function
    def norm(x, mu, sigma):
        return np.exp( -( ( x - mu ) / sigma )**2 / 2 ) / ( np.sqrt( 2 * np.pi ) * sigma )

    # probability distribution > binomial distribution, distribution of binary data
    n = 1000
    p = 0.5
    x = np.arange(430, 571)
    y1 = [scsp.comb(n, i) * p**i * (1-p)**(n-i) for i in x]

    # normal distribution, distribution of continuous data
    mu = n/2
    sigma = np.sqrt(mu/2)
    y2 = norm(x, mu, sigma)

    # plot
    plt.bar(x, y1)
    plt.plot(x, y2, 'k')
    plt.xlim((430, 570))
    plt.grid(ls=':')
    plt.title('probability distribution vs. normal distribution')



#-----------------------------------------------------
# 1.4 probability model & sampling
#-----------------------------------------------------

# a. preparation of data using pandas or NumPy
# b. definition of probability model using PyMC
# c. sampling using PyMC
# d. statistical analysis of results using ArviZ

if True:
    # PyMC, ArviZ version check
    print('Running on PyMC ver. %s' % pm.__version__)
    print('Running on ArviZ ver. %s' % az.__version__)

    #
    model = pm.Model()

    #
    with model:
        # binomial distribution, distribution of binary data
        x = pm.Binomial('x', p=0.5, n=5)
