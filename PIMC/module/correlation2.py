import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from numba import njit
import os
from matplotlib import pyplot as plt
import multiprocessing
from functools import partial

Nt_array=np.arange(200, 3800, 400)
execute=True


@njit(fastmath=True, cache=True)
def correlations_retry(x: np.array):
    x=np.abs(x[10000:])
    x_mean=x.mean()
    N=len(x)
    k_max=int(N/20)
    C=np.zeros(k_max)
    for k in range(1, k_max):
        somma=0
        factor=1/(N-k)
        for j in range(N-k):
            somma+=(x[j]-x_mean)*(x[j+k]-x_mean)
        C[k]=somma*factor
    return C

path='results/results_beta2/risultati_tailor_050'
if execute:
    tau=[]
    Corr_array=[]
    for Q_arr in os.listdir(path):
        print(f'Calcolo per {Q_arr}')
        Q_path=os.path.join(path, Q_arr)
        Q=np.loadtxt(Q_path)
        C=correlations_retry(Q)
        Corr_array.append(C)
        tau_single=np.abs(sum(Q))
        tau.append(tau_single)
   
    Corr_array=np.array(Corr_array)
    length=len(Corr_array[0])
    for i in range(len(Corr_array)-3):
        print(Corr_array[i].max())
        plt.plot(range(length), Corr_array[i])
    plt.show()
    



