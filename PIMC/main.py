from functools import partial
import logging
import os
import shutil
from scipy.optimize import curve_fit
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

import module.bootstrap as boot
import model.constants as const
import module.func_cerchio as fnc
import module.blocking as block


logging.basicConfig(level=logging.INFO)
# Nt_arr = np.arange(200, 1200, 50)  # Tailor
# Nt_arr = np.arange(20, 450, 10)  # Metro
Nt_arr = np.array([700])
a_arr = np.array(2.0 / Nt_arr)
x_fit = np.arange(100, 451, 5)
bin_arr = const.BIN_ARRAY

Metropolis = True

if Metropolis:
    Tailor = False
else:
    Tailor = True

bootstrap_exe = False
blocking = False
block_saturation = False
graphic_print = False
beta_analysis = 2


def f(x, a, b):
    return np.array((a) * np.exp(x * b))


if __name__ == "__main__":

    beta2 = "results/results_beta2"
    beta10 = "results/results_beta10"

    if os.path.exists(beta2) == False:
        os.makedirs(beta2)
    else:
        logging.info(f"{beta2} esiste")
    if os.path.exists(beta10) == False:
        os.makedirs(beta10)
    else:
        logging.info(f"{beta10} esiste")

    if beta_analysis == 2:
        path_save = beta2
    if beta_analysis == 10:
        path_save = beta10

    cartella_050 = f"{path_save}/risultati_metro_050"
    cartella_sqrt_d = f"{path_save}/risultati_metro_sqrt_d"
    cartella_tailor05 = f"{path_save}/risultati_tailor_050"
    cartella_tailor_sqrt_d = f"{path_save}/risultati_tailor_sqrt_d"

    if Tailor:
        path_analysis = cartella_tailor05
    if Metropolis:
        path_analysis = cartella_050

    if os.path.exists(path_analysis):
        shutil.rmtree(path_analysis)
    os.makedirs(path_analysis)

    tau_list = []
    processi = len(Nt_arr)

    if Metropolis:
        if os.path.exists(path_analysis):
            shutil.rmtree(path_analysis)
        os.makedirs(path_analysis)
        with multiprocessing.Pool(processes=processi) as pool:
            q_arr = np.array(pool.map(fnc.cammino_piano, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            q = q_arr[Nt, :]
            np.savetxt(f"{path_analysis}/MC_Nt_{Nt_arr[Nt]}.txt", q)

    if Tailor:
        if os.path.exists(path_analysis):
            shutil.rmtree(path_analysis)
        os.makedirs(path_analysis)
        with multiprocessing.Pool(processes=processi) as pool:
            q_tailor = np.array(pool.map(fnc.Tailor, Nt_arr), dtype="object")
            pool.close()
            pool.join()
        for Nt in range(len(Nt_arr)):
            q = q_tailor[Nt, :]
            np.savetxt(f"{path_analysis}/MC_Nt_{Nt_arr[Nt]}.txt", q)

    if bootstrap_exe:
        sigma = []
        delta2_naive = []
        proc = len(bin_arr)
        num_Nt = len(Nt_arr)
        for qq in range(num_Nt):
            logging.info(f"siamo al {Nt_arr[qq]} Nt")
            Q = np.loadtxt(f"{path_analysis}/MC_Nt_{Nt_arr[qq]}.txt")
            N = len(Q)  # N_sample
            with multiprocessing.Pool(processes=proc) as pool:
                parziale = partial(boot.bootstrap_binning, Q)
                results = np.array(pool.map(parziale, bin_arr), dtype="object")
                pool.close()
                pool.join()
                sigma.append(max(results))
            # calcolo sigma naive
            q2 = np.array(Q) ** 2
            q4 = np.array(q2) ** 2
            delta2_naive.append((np.mean(q4) - np.mean(q2) ** 2) / N)

        delta2_naive = np.array(delta2_naive)
        tau = np.array(
            [(0.5 * (sigma[t] ** 2)) / (delta2_naive[t]) for t in range(len(sigma))]
        )
        tau_arr = np.zeros(len(Nt_arr))

        for i in range(len(Nt_arr)):
            tau_arr[i] = tau[i]
        # Exponential fit bootstrap
        opt, _ = curve_fit(
            f, Nt_arr, tau_arr, p0=(0, 0), bounds=(-np.inf, np.inf), maxfev=10000
        )
        a, b = opt
        y_fit = f(x_fit, a, b)

        logging.info(f"Optimal parameters a = {a}, b = {b} ")
        logging.info(f"tau boot: {tau_arr}")
        logging.info(f"Lunghezza array: {N}")

        plt.figure()
        plt.scatter(Nt_arr, tau_arr, s=4, c="red", label=r"\tau")
        plt.plot(x_fit, y_fit, "g--", label="Fit")
        plt.legend()
        plt.yscale("log")
        plt.show()

    if blocking:
        sigma_block = []
        delta2_naive = []

        for qq in range(len(Nt_arr)):
            logging.info(f"blocking Nt: {Nt_arr[qq]}")
            Q = np.loadtxt(f"{path_analysis}/MC_Nt_{Nt_arr[qq]}.txt")
            N = len(Q)
            q2 = np.array(Q) ** 2
            sigma_block.append(
                block.blocking_errors(q2, 2000, False)
            )  # grandezza blocco max
            q4 = np.array(q2) ** 2
            delta2_naive.append((np.mean(q4) - np.mean(q2) ** 2) / N)
            # delta2_naive.append(np.var(q2)/len(q2)) #viene la stessa cosa

        sigma = sigma_block

        delta2_naive = np.array(delta2_naive)
        tau = np.array(
            [(0.5 * (sigma[t] ** 2)) / (delta2_naive[t]) for t in range(len(sigma))]
        )
        tau_arr = np.zeros(len(Nt_arr))
        for i in range(len(Nt_arr)):
            tau_arr[i] = tau[i]

        tau_arr = tau_arr
        # Exponential fit
        opt, cov = curve_fit(
            f, Nt_arr, tau_arr, p0=(0, 0), bounds=(-np.inf, np.inf), maxfev=100000
        )
        a, b = opt
        y = f(x_fit, a, b)

        logging.info(f"Optimal parameters a = {a}, b = {b}")
        logging.info(f"Tau blocking: {tau_arr}")
        logging.info(f"Lunghezza campione: {N}")
        logging.info(
            f"Sigma parametri: sigma_a = {np.sqrt(cov[0, 0])}, sigma_b = {np.sqrt(cov[1, 1])}"
        )

        plt.figure()
        plt.title(
            r"Andamento $\tau$ con Tailor, $\beta = 2$, $\delta = 0.5$", fontsize=18
        )
        plt.scatter(Nt_arr, np.array(tau_arr), s=9, c="red", label=r"$\tau$")
        # plt.plot(x_fit, y, 'g--', label=r'f=$a_0e^{a_1 Nt}$')
        plt.xlabel(r"$N_t$", fontsize=14)
        plt.ylabel(r"$\tau$", fontsize=14)
        plt.legend()
        plt.yscale("log")
        plt.show()

    if block_saturation:
        Q = np.loadtxt(f"{path_analysis}/MC_Nt_300.txt")
        block.blocking_errors(np.array(Q), 10000, True)

    if graphic_print:
        for Nt in range(len(Nt_arr)):
            q = np.loadtxt(f"{path_analysis}/MC_Nt_{Nt_arr[Nt]}.txt")
            fnc.graphic_analysis(Nt_arr[Nt], q)
