import numpy as np
from numba import njit, float64

@njit(float64[:](float64[:],float64), fastmath=True)
def ricampionamento(array_osservabile, bin):
    sample=[]
    for _ in range(round(len(array_osservabile)/bin)+1): #se bin>len è zèro!
        ii=np.random.randint(0, len(array_osservabile)+1)
        sample.extend(array_osservabile[ii:min(ii+bin, len(array_osservabile))]) 
    return np.array(sample)

def bootstrap_binning(array_osservabile, bin):
    obs_list=[]
    ritorno=[]
    array_osservabile=array_osservabile.astype(np.float64)
    for _ in range(100):
        sampler=ricampionamento(array_osservabile, bin)
        sampler=np.array(sampler)**2
        obs=np.mean(sampler)
        obs_list.append(obs)
    obs_list2=np.array(obs_list)**2
    # ritorno.append(np.sqrt(np.mean(obs_list2)-np.mean(obs_list)**2))
    ritorno.append(np.std(obs_list, ddof=1))
    return ritorno

