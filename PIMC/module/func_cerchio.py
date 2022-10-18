from re import A
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, float64, types
from scipy.stats import norm
from scipy.optimize import curve_fit

### costanti
cammini = 10000
term = 2000
beta = 2
idecorrel = 10


@njit()
def distanza(x, y):
    if abs(x - y) <= 0.5:
        d = x - y
    if (x - y) < -0.5:
        d = x - y + 1.0
    if (x - y) > 0.5:
        d = x - y - 1.0
    return d


@njit(fastmath=True, cache=True)
def diff_azione(i, y_new, ypp, ymm, yprova):  # i, y_new, ypp, ymm, yprova
    d1 = distanza(ypp[i], yprova)
    d2 = distanza(ypp[i], y_new[i])
    d3 = distanza(yprova, ymm[i])
    d4 = distanza(y_new[i], ymm[i])
    return (d1 * d1) + (d3 * d3) - (d2 * d2) - (d4 * d4)


@njit(fastmath=True, cache=True)
def avvolgimento(Nt, y, g):
    sum = 0
    for i in range(0, Nt):
        sum += distanza(g[i], y[i])
    return round(sum)


@njit(fastmath=True, cache=True)
def geometry(Nt):
    npp = [i + 1 for i in range(0, Nt)]
    nmm = [i - 1 for i in range(0, Nt)]
    npp[Nt - 1] = 0
    nmm[0] = Nt - 1
    return (npp, nmm)


@njit(fastmath=True, cache=True)
def metropolis(y, Nt, a, ypp, ymm):
    y_new = y.copy()
    delta = np.sqrt(a)  # DELTA!!!
    for i in range(Nt):
        r = np.random.uniform(-delta, delta)
        yprova = (y_new[i] + r) % 1
        if i == 0:
            ymm[Nt - 1] = y_new[0]
        else:
            ymm[i - 1] = y_new[i]
        if i == Nt - 1:
            ypp[0] = y_new[Nt - 1]
        else:
            ypp[i + 1] = y_new[i]
        s = diff_azione(i, y_new, ypp, ymm, yprova)
        x = np.exp(-s / (2 * a))
        if (s < 0) or (np.random.rand() < x):
            y_new[i] = yprova
        if i == 0:
            ymm[Nt - 1] = y_new[0]
        else:
            ymm[i - 1] = y_new[i]
        if i == Nt - 1:
            ypp[0] = y_new[Nt - 1]
        else:
            ypp[i + 1] = y_new[i]
    y_new[Nt - 1] = y_new[0]
    return y_new, ypp, ymm


@njit(fastmath=True, parallel=True)
def cammino_piano(Nt):
    ypp = np.zeros(Nt)
    ymm = np.zeros(Nt)
    a = beta / Nt
    q = []
    y = np.zeros(Nt)
    for cam in range(cammini):
        if cam % 10000 == 0:
            print("metro", cam, "Nt = ", Nt)
        for _ in range(idecorrel):
            y_new, ypp, ymm = metropolis(y, Nt, a, ypp, ymm)
            y = y_new
        if (cam > term) and cam % 5 == 0:
            q.append(avvolgimento(Nt, y_new, ypp))
    return np.array(q)


@njit(fastmath=True, cache=True)
def Tailor(Nt):
    a = beta / Nt
    delta = np.sqrt(a)  # DELTA !
    p_cut = 0.06
    epsilon = 0.2 * a
    y = np.zeros(Nt)  # np.random.normal(0, np.sqrt(beta), size=Nt)  # np.zeros(Nt)
    npp, nmm = geometry(Nt)
    q_list = []
    for t in range(cammini):
        dS = 0
        if t % 10000 == 0:
            print(f"gno Tailor step: {t} Nt : {Nt}")
        ypp, ymm = np.zeros(Nt), np.zeros(Nt)
        if np.random.rand() < p_cut:
            y0 = (y[0] + 0.5) % (1)
            y_new = y.copy()
            for i in range(Nt):
                if abs(distanza(y[i], y0)) <= epsilon:
                    iend = i
                continue
            yprova = (2 * y0 - y[iend]) % (1)
            dS = (
                distanza(yprova, y[iend + 1]) ** 2 - distanza(y[iend], y[iend + 1]) ** 2
            )
            if dS < 0:
                cambio = True
            else:
                if np.random.rand() < np.exp(-dS / (2 * a)):
                    cambio = True
                else:
                    cambio = False
            if cambio == True:
                for m in range(iend, Nt):
                    y_new[m] = (2 * y0 - y[m]) % (1)
                    ypp[m] = y_new[npp[m]]
                    ymm[m] = y_new[nmm[m]]
            y = y_new
            y[Nt - 1] = y[0]
        for h in range((Nt)):
            dS1 = 0
            rand = np.random.randint(0, Nt)
            r = np.random.uniform(-delta, delta)
            y_old = y[rand]
            y_bef = y[nmm[rand]]
            y_aft = y[npp[rand]]
            y_new1 = (y_old + r) % 1
            dS1 = (
                distanza(y_aft, y_new1) ** 2
                + distanza(y_new1, y_bef) ** 2
                - distanza(y_aft, y_old) ** 2
                - distanza(y_old, y_bef) ** 2
            )
            acceptance = np.exp(-dS1 / (2 * a))
            if (dS1 < 0) or (np.random.rand() < acceptance):
                y[rand] = y_new1
        y[Nt - 1] = y[0]

        for l in range(Nt):
            ypp[l] = y[npp[l]]
            ymm[l] = y[nmm[l]]

        if (t > term) and (t % 5 == 0):
            q = avvolgimento(Nt, y, ypp)
            q_list.append(q)
    return np.array(q_list)


@njit()
def Tailor2punto0(Nt):

    a = beta / Nt
    delta = np.sqrt(a)  # DELTA !
    p_cut = 0.4
    epsilon = 0.2 * a
    y = np.zeros(Nt)  # np.random.normal(0, np.sqrt(beta), size=Nt)  # np.zeros(Nt)
    npp, nmm = geometry(Nt)
    q_list = []
    for t in range(cammini):

        for h in range((Nt)):
            dS1 = 0
            rand = np.random.randint(0, Nt)
            r = np.random.uniform(-delta, delta)
            y_old = y[rand]
            y_bef = y[nmm[rand]]
            y_aft = y[npp[rand]]
            y_new1 = (y_old + r) % 1
            dS1 = (
                distanza(y_aft, y_new1) ** 2
                + distanza(y_new1, y_bef) ** 2
                - distanza(y_aft, y_old) ** 2
                - distanza(y_old, y_bef) ** 2
            )
            acceptance = np.exp(-dS1 / (2 * a))
            if (dS1 < 0) or (np.random.rand() < acceptance):
                y[rand] = y_new1
        y[Nt - 1] = y[0]


        dS = 0
        if t % 10000 == 0:
            print(f"gno Tailor step: {t} Nt : {Nt}")
        ypp, ymm = np.zeros(Nt), np.zeros(Nt)
        if np.random.rand() < p_cut:
            y0 = (y[0] + 0.5) % (1)
            y_new = y.copy()
            for i in range(Nt):
                if abs(distanza(y[i], y0)) <= epsilon:
                    iend = i
                continue
            yprova = (2 * y0 - y[iend]) % (1)
            dS = (
                distanza(yprova, y[iend + 1]) ** 2 - distanza(y[iend], y[iend + 1]) ** 2
            )
            if dS < 0:
                cambio = True
            else:
                if np.random.rand() < np.exp(-dS / (2 * a)):
                    cambio = True
                else:
                    cambio = False
            if cambio == True:
                for m in range(iend, Nt):
                    y_new[m] = (2 * y0 - y[m]) % (1)
                    ypp[m] = y_new[npp[m]]
                    ymm[m] = y_new[nmm[m]]
            y = y_new
            y[Nt - 1] = y[0]
        
        for l in range(Nt):
            ypp[l] = y[npp[l]]
            ymm[l] = y[nmm[l]]

        if (t > term) and (t % 5 == 0):
            q = avvolgimento(Nt, y, ypp)
            q_list.append(q)
    return np.array(q_list)



def graphic_analysis(Nt, q):

    bins = np.arange(q.min(), q.max() + 2)
    bins = bins - 0.5
    xlims = [-15, 15]
    x = np.linspace(*xlims, 1000)

    plt.figure()
    plt.hist(
        q, bins, density=True, histtype="step", fill=False, color="b", label=f"{Nt}"
    )

    plt.xlim(xlims)
    plt.plot(x, norm.pdf(x, 0, np.sqrt(beta)), "g--", label="Distr. teorica")
    plt.xlabel("Q")
    plt.ylabel("P(Q)")
    plt.legend()
    plt.title(f"Tailor Nt = {Nt}, " r"$\beta$=2, $\delta$=$\sqrt{\eta}$")
    plt.show()


# queste funzioni sono da ricontrollare... forse lo faremo forse no
def connected_time_correlation(x: np.array, max_time, normalized=False) -> np.array:
    if max_time > len(x) - 1:
        raise IndexError

    x_mean = x.mean()
    C = np.array(
        [(x * x).mean() - x_mean ** 2]
        + [(x[:-k] * x[k:]).mean() - x_mean ** 2 for k in range(1, max_time)]
    )
    if normalized:
        if C[0] != 0:
            return C / C[0]
        else:
            print("C[0] is zero, returning unormalized correlations")

    return C


def exponential(x, a, tau):
    return a * np.exp(-x / tau)


def correlation_time(x, label=None):
    C_LIMIT = 0.005
    T_LIMIT = 1000
    """Compute the montecarlo correlation time via an exponential fit
    """
    max_time = int(len(x) / 20)
    C = connected_time_correlation(x, max_time, normalized=True)

    stop = np.argmin(C > C_LIMIT)
    if stop == 0:
        stop = T_LIMIT
    elif stop == 1:
        stop = 3

    C_fit = C[:stop]

    opt, cov = curve_fit(exponential, np.arange(stop), C_fit, p0=(1, 1))
    print(opt[1])

    return opt[1]

