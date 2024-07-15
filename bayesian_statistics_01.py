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

#-----------------------------------------------------
# discrete distribution
#-----------------------------------------------------

if True:
    n = 5
    p = 0.5
    x = range(n+1)
  
    # distribution function
    y = [scsp.comb(n, i) * p**i * (1-p)**(n-i) for i in x]

    # plot
    plt.bar(x, y)
    plt.grid(ls=':')
    plt.title('visualization of probabilty distribution')

