import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

def connected_correlation_time(x: np.array, normalized=False):
    x_mean=x.mean()
    C_fixed=(x*x).mean()-x_mean**2
    print(C_fixed)
    max_time=int(len(x)/20)
    C=np.array([C_fixed+(x[:-k]*x[k:]).mean()-x_mean**2 for k in range(1, max_time)])
    if normalized:
        if C[0]!=0:
            return C/C[0]
        else:
            print('C0 è zero ')
    return C

def exponential(k, a, tau):
    return a*np.exp(-k/tau)

def correlation_time(x):
    C=connected_correlation_time(x)

    opt, _ =curve_fit(exponential, np.arange(len(C)), C, p0=(1, 1))
    return opt[1]
    