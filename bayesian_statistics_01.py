#
# TITLE: Bayesian statistics
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: 赤石雅典, Pythonでスラスラわかるベイズ推論「超」入門, 講談社, 2023-11-24
#
#

import numpy as np
import scipy.stats as scs
import scipy.special as scsp
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import arviz as az
import seaborn as sns

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

        # sample prior predictive
        prior_samples = pm.sample_prior_predictive(random_seed=42)

        # analysis using numpy
        x_samples = prior_samples['prior']['x'].values
        print('type: ', type(x_samples))
        print('shape: ', x_samples.shape)
        print('values: ', x_samples, '\n')

        # analysis using pandas
        value_counts = pd.DataFrame( x_samples.reshape(-1) ).value_counts().sort_index()
        print(value_counts)

        # statistical analysis using ArviZ
        summary = az.summary(prior_samples, kind='stats')
        print(summary)

        # visualization using ArviZ
        ax = az.plot_dist(x_samples)
        ax.set_title('ArviZ visualization')



#-----------------------------------------------------
# 1.6 probability distribution & PyMC programming
#-----------------------------------------------------

if True:
    #
    model = pm.Model()

    #
    with model:
        x = pm.Binomial('x', p=0.5, n=5)
        prior_samples = pm.sample_prior_predictive(random_seed=42)

    #
    x_samples = prior_samples['prior']['x'].values

    #
    summary = az.summary(x_samples, kind='stats')
    print(summary)

    #
    ax = az.plot_dist(x_samples)



#-----------------------------------------------------
# 2.1 Bernoulli discrete probability distribution
#-----------------------------------------------------

if True:
    #
    p = 0.5

    model1 = pm.Model()

    with model1:
        x = pm.Bernoulli('x', p=p)
        prior_samples1 = pm.sample_prior_predictive(random_seed=42)

    # numpy array
    x_samples1 = prior_samples1['prior']['x'].values
    print(x_samples1)

    # arviz statistical analysis
    summary1 = az.summary(prior_samples1, kind='stats')
    display(summary1)

    ax = az.plot_dist(x_samples1)
    ax.set_title('Bernoulli distribution')



#-----------------------------------------------------
# 2.2 binomial discrete probability distribution
#-----------------------------------------------------

if True:
    #
    p = 0.5
    n = 5

    # PyMC
    model2 = pm.Model()

    with model2:
        x = pm.Binomial('x', p=p, n=n)
        prior_samples2 = pm.sample_prior_predictive(random_seed=42)

    # Numpy array
    x_samples2 = prior_samples2['prior']['x'].values
    print(x_samples2)

    # ArviZ
    summary2 = az.summary(prior_samples2, kind='stats')
    display(summary2)

    ax = az.plot_dist(x_samples2)
    ax.set_title('binomial distribution p=%.2f n=%i' % (p, n))

    #--------------------------

    #
    p = 0.5
    n = 50

    # PyMC
    model3 = pm.Model()

    with model3:
      x = pm.Binomial('x', p=p, n=n)
      prior_samples3 = pm.sample_prior_predictive(random_seed=42)

    # Numpy
    x_samples3 = prior_samples3['prior']['x'].values
    print(x_samples3)

    # ArviZ
    summary3 = az.summary(prior_samples3, kind='stats')
    display(summary3)

    ax = az.plot_dist(x_samples3)
    ax.set_title('binomial distribution p=%.2f n=%i' %  (p, n))



#-----------------------------------------------------
# 2.3 normal continuous probability distribution
#-----------------------------------------------------

if True:
    # setosa sepal length
    df = sns.load_dataset('iris')

    df1 = df.query('species == "setosa"')

    bins = np.arange(4.0, 6.2, 0.2)

    # KDE: kernel density estimation
    fig = plt.subplots(1,1)
    sns.histplot(df1, x='sepal_length', bins=bins, kde=True)
    plt.xticks(bins)

    #-----------------------------------------------------

    def norm(x, mu, sigma):
        return np.exp( -(x - mu)**2 / 2 ) / ( np.sqrt(2*np.pi) * sigma )

    #
    mu1, sigma1 = 3.0, 2.0
    mu2, sigma2 = 1.0, 3.0

    #
    x = np.arange(-8.0, 10.01, 0.01)

    xticks = np.arange(-8.0, 11.0, 1.0)

    #
    fig, ax = plt.subplots(1,1)
    ax.plot(x, norm(x, mu1, sigma1), label='$\mu$=%.1f, $\sigma$=%.1f' % (mu1, sigma1))
    ax.plot(x, norm(x, mu2, sigma2), label='$\mu$=%.1f, $\sigma$=%.1f' % (mu2, sigma2))
    plt.xticks(xticks, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(ls=':')
    ax.set_title('normal distribution function')

    #-----------------------------------------------------

    mu = 0.0
    sigma = 1.0

    model4 = pm.Model()

    with model4:
        x = pm.Normal('x', mu=mu, sigma=sigma)
        prior_samples4 = pm.sample_prior_predictive(random_seed=42)

    #
    x_samples4 = prior_samples4['prior']['x'].values
    print(x_samples4[:,:100])

    fig, ax = plt.subplots(1,1)
    ax = az.plot_dist(x_samples4)
    ax.set_title('normal distribution $mu$=%.1f 4sigma$=%.1f' % (mu, sigma))
    ax.grid(ls=':')

    #-----------------------------------------------------

    bins = np.arange(-4.0, 4.5, 0.5)

    fig, ax = plt.subplots(1, 1)
    ax = az.plot_dist(x_samples4, kind='hist', hist_kwargs={'bins': bins})
    ax.set_title('normal distribution $mu$=%.1f $sigma$%.1f' % (mu, sigma))
    ax.grid(ls=':')

    #-----------------------------------------------------

    summary4 = az.summary(prior_samples4, kind='stats')
    display(summary4)

    #-----------------------------------------------------

    norm_model = scs.norm(0.0, 1.0)
    print(norm_model.cdf(-1.88), norm_model.cdf(1.88))

    #-----------------------------------------------------

    mu = 3.0
    sigma = 2.0

    model5 = pm.Model()

    with model5:
        x = pm.Normal('x', mu=mu, sigma=sigma)
        prior_samples5 = pm.sample_prior_predictive(random_seed=42)

    x_samples5 = prior_samples5['prior']['x'].values
    print(x_samples5[:,:100])

    summary5 = az.summary(prior_samples5)
    print(summary5)

    fig, ax = plt.subplots(1, 1)
    ax = az.plot_dist(x_samples5)
    ax.set_title('normal distribution $mu$=%.1f, $sigma$=%.1f' % (mu, sigma))
    ax.grid(ls=':')



#-----------------------------------------------------
# 2.4 uniform continuous probability distribution
#-----------------------------------------------------

if True:
    #
    lower = 0.0
    upper = 1.0

    #
    model6 = pm.Model()

    with model6:
        x = pm.Uniform('x', lower=lower, upper=upper)
        prior_samples6 = pm.sample_prior_predictive(random_seed=42)

    #
    x_samples6 = prior_samples6['prior']['x'].values
    print(x_samples6[:,:100])

    #
    fig, ax = plt.subplots(1, 1)
    ax = az.plot_dist(x_samples6)
    ax.set_title('uniform probability distribution, continuous')
    ax.grid(ls=':')

    #
    bins = np.arange(0.0, 1.1, 0.1)
    fig, ax = plt.subplots(1, 1)
    ax = az.plot_dist(x_samples6, kind='hist', hist_kwargs={'bins': bins})
    ax.set_title('uniform probability distribution lower=%.1f upper=%.1f' % (lower, upper))
    ax.grid(ls=':')

    #
    summary6 = az.summary(prior_samples6, kind='stats')
    print(summary6)

    #-----------------------------------------------------

    lower = 0.1
    upper = 0.9

    model7 = pm.Model()

    with model7:
        x = pm.Uniform('x', lower=lower, upper=upper)
        prior_samples7 = pm.sample_prior_predictive(random_seed=42)

    x_samples7 = prior_samples7['prior']['x'].values
    print(x_samples7[:,:100])

    fig, ax = plt.subplots(1, 1)
    ax = az.plot_dist(x_samples7)
    ax.set_title('uniform probability distribution lower=%.1f upper=%.1f' % (lower, upper))
    ax.grid(ls=':')

    summary7 = az.summary(prior_samples7, kind='stats')
    print(summary7)



#-----------------------------------------------------
# 2.5 Beta continuous probability distribution
#-----------------------------------------------------

if True:
    #









