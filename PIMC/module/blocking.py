from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit(fastmath=True, cache=True)
def blocking(x: np.array, block_dim: int):

    num_block=int(len(x)/block_dim)
    y=np.empty(num_block)

    for i in range(num_block):
        y[i]=np.mean(x[(i*block_dim):(i*block_dim+block_dim)])
    return y

# @njit(fastmath=True, cache=True)
def blocking_errors(x: np.array, max_size: int, plot: bool):
    block_steps=20
    blocks=np.arange(1, max_size, block_steps)
    sigma=[]
    for block_size in blocks:
        x_block=blocking(x, block_size)
        sigma.append(np.std(x_block)/np.sqrt(len(x_block)))
    sigma=np.array(sigma)
    print('Saturation error: ', sigma.max())
    if plot:
        plt.scatter(blocks, sigma, s=6, c='blue')
        plt.title(r'Blocking $\sigma_Q$ Metro, $N_t=300$', fontsize=18)
        plt.xlabel('block size', fontsize=14)
        plt.ylabel(r'$\sigma_Q$', fontsize=14)
        plt.show()
    return(sigma.max())
    
