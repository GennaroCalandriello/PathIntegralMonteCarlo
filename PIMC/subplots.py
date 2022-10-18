import numpy as np
import matplotlib.pyplot as plt
import module.func_cerchio as fnc
from scipy.stats import norm

# Nt_analysis=[600, 1000, 2200, 3800]
Nt_analysis=[2200, 3800]
subplot=True

path_analysis='results/results_beta2/risultati_tailor_sqrt_d'
fig, axs = plt.subplots(2, 1, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.5)

Q=np.array([np.loadtxt(f'{path_analysis}/MC_Nt_{Nt_analysis[i]}.txt') for i in range(2)])
print(len(Q[0]))

for i in range(2):
    fnc.graphic_analysis(Nt_analysis[i], Q[i])

if subplot:
    for i in range(2):
        print(Q[i])
        axs[i].plot(range(len(Q[i])), Q[i], color='blue')
        axs[i].set_title(f' Tailor Nt = {Nt_analysis[i]}, ' r'$\delta=\sqrt{\eta}$, $\beta$ = 2')
        axs[i].set_xlabel('Steps')
        axs[i].set_ylabel('Q')
    plt.show()




