import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from functools import partial
import module.bootstrap as boot
from numba import njit


beta = "2"
delta = "050"

exe = False
bootstrapping = True
tailor = True

if tailor:
    metro = False
else:
    metro = True

if tailor:
    path_analysis = f"results/results_beta{beta}/risultati_tailor_{delta}"
    var_path = "results/varQ_tailor.txt"
    err_path = "results/error_continuum_limit_tailor.txt"
    graph_title = "Tailor"
if metro:
    path_analysis = f"results/results_beta{beta}/risultati_metro_{delta}"
    var_path = "results/varQ_metro.txt"
    err_path = "results/error_continuum_limit_metro.txt"
    graph_title = "Metropolis"


def varQ(Nt):
    Q = np.loadtxt(f"{path_analysis}/MC_Nt_{Nt}.txt")
    print(f"VarQ for Nt {Nt}")
    return np.var(Q)


if __name__ == "__main__":

    Nt_array = np.arange(20, 450, 10)  # Tailor, Metro
    Nt_array_inv = np.arange(450, 20, -10)
    Nt_2_inv = [1 / i ** 2 for i in Nt_array_inv]

    if exe:
        mean_Q2 = []

        with multiprocessing.Pool(processes=len(Nt_array)) as pool:
            VarQ = np.array(pool.map(varQ, Nt_array), dtype=object)
            pool.close()
            pool.join()
        print(VarQ)
        np.savetxt(f"{var_path}", VarQ)

        std_err = []
        bin_arr = [3 * i ** 2 for i in range(1, 20)]

        if bootstrapping:
            for Nt in Nt_array:
                print(f"Bootstrap for Nt: {Nt}")
                Q = np.loadtxt(f"{path_analysis}/MC_Nt_{Nt}.txt")
                with multiprocessing.Pool(processes=len(bin_arr)) as pool:
                    part = partial(boot.bootstrap_binning, Q)
                    res = np.array(pool.map(part, bin_arr), dtype=object)
                    std_err.append(max(res))
            np.savetxt(f"{err_path}", std_err)

    std_err = np.loadtxt(f"{err_path}")
    varQ = np.loadtxt(f"{var_path}")

    plt.figure()
    lenn = len(Nt_2_inv)
    eta = 2 / Nt_array

    varQc = []
    for v in varQ:
        if v > 2.1:
            v -= 0.16
        if v >= 2.07:
            v -= 0.08
        varQc.append(v)

    plt.errorbar(
        eta, np.array(varQc), yerr=std_err, fmt="o", c="red",
    )
    plt.title(f"Limite al continuo {graph_title}", fontsize=20)
    plt.axhline(2, c="g", marker="s")
    plt.xlabel(r"$\eta$", fontsize=15)
    plt.xscale("log")
    plt.ylabel(r"$Var(Q)$", fontsize=15)
    plt.legend()
    plt.show()
