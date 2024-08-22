import numpy as np
from scipy.special import erf, ndtr, ndtri

"""
Name
    kendall

Purpose

   Compute Kendall's tau correlation coefficient following Isobe, Feigelson, and Nelson
   (1986; ApJ 306:490) accounting for left-censored data (i.e., data with upper limits).
   

Arguments
    :x (*np.ndarray*): 1xN array of independent variable, containing either
                            measured values or detection limits
    :y (*np.ndarray*): 1xN array of dependent variable, containing either
                            measured values or detection limits

Keyword Arguments:
    :censors (*np.ndarray*): 2xN array containing censors of `x` and `y` with 1
                            representing an uncensored datum and 0 representing
                            a left-censored (upper-limit) datum. Default is
                            uncensored (2xN array of 1s).

Returns
    :tau (*float*): Kendall's tau correlation coefficient, accounting for
                            upper limits on data
    :p (*float*): probability that the null hypothesis (no correlation) is true
"""


def kendall(x, y, censors=None, varcalc="simple", upper=True):
    # check that x and y are identical lengths
    if len(x) != len(y):
        print("X and Y different lengths!")
        return None
    else:
        # length of data
        n = len(x)
    # check if censors exist, and if not, assume no censoring
    if np.any(censors == None):
        censors = np.ones((2, n))
    # otherwise, check that censor has same shape as T
    elif len(censors) != 2 or len(censors[0]) != n:
        print("Censor length must match data length!")
        return
    # array of rank differences (J in A&S Eqns 1, 3, 5, and 6)
    J = np.zeros((2, n, n))
    # array of data (T in A&S Eqn 1 is for right censored)
    if upper:
        T = np.array([-x, -y])
    else:
        T = np.array([x, y])

    # Compute ararys a_ij & b_ij as defined in Eq. (27) of Isobe et al.
    # or in Sect. 2.2 of Akritas & Siebert
    # (a_ij in Isobe = -a_ij in Akritas, does not matter, but we follow Isobe)
    for j in range(n):
        for i in range(n):
            for k in range(2):
                I_kij = 0.0
                if T[k, i] < T[k, j]:
                    I_kij = 1.0

                I_kji = 0.0
                if T[k, j] < T[k, i]:
                    I_kji = 1.0

                J[k, i, j] = censors[k, j] * I_kji - censors[k, i] * I_kij

    # Kendall's tau
    tau = np.sum(np.prod(np.triu(J), axis=0))  
    tau *= 2 / (float(n) * (float(n) - 1))

    if varcalc == "simple":
        # simple p-value
        var = 2 * (2 * n + 5) / (9 * n * (n - 1))
        p = 1 - erf(abs(tau) / np.sqrt(2 * var))

    elif varcalc == "ifn":
        # calculation after Eq. (26) in Isobe et al.
        S_array = [[J[0, i, j] * J[1, i, j] for i in range(n)] for j in range(n)]
        S = np.sum(np.array(S_array))

        # calculation afer Eq. (28) in Isobe et al.
        A_1_arr = np.array(
            [
                [[J[0, i, j] * J[0, i, k] for i in range(0, n)] for j in range(0, n)]
                for k in range(0, n)
            ]
        )
        B_1_arr = np.array(
            [
                [[J[1, i, j] * J[1, i, k] for i in range(0, n)] for j in range(0, n)]
                for k in range(0, n)
            ]
        )

        A_1 = np.sum(A_1_arr)
        B_1 = np.sum(B_1_arr)
        A_2 = np.sum(J[0, :, :] ** 2)
        B_2 = np.sum(J[0, :, :] ** 2)
        A_1 -= A_2
        B_1 -= B_2
        var_1 = (4 * A_1 * B_1) / (n * (n - 1) * (n - 2))
        var_2 = (2 * A_2 * B_2) / (n * (n - 1))
        var = var_1 + var_2

        p = 1 - erf(abs(S) / np.sqrt(2 * var))

    return tau, p


"""
Name
    tau_conf

Purpose
    Determine confidence intervals on Kendall's tau with left-censored data by
    bootstrapping values of x and y. Kendall's tau calculated following
    Akritas & Siebert (1996) MNRAS 278, 919-924.

Arguments
    :x (*np.ndarray*): 1xN array of independent variable, containing either
                            measured values or detection limits
    :y (*np.ndarray*): 1xN array of dependent variable, containing either
                            measured values or detection limits

Keyword Arguments:
    :censors (*np.ndarray*): 2xN array containing censors of `x` and `y` with 1
                            representing an uncensored datum and 0 representing
                            a left-censored (upper-limit) datum. Default is
                            uncensored (2xN array of 1s).
    :p_conf (*float*): two-sided  probability interval of the desired confidence
                            interval. Default is 0.6826 (1-sigma).
    :n_samp (*int*): number of samples to draw. Default is 10^4.
    :method (*str*): 'montecarlo' for Monte Carlo sampling of uncertainties or
                            'bootstrap' for bootstrapping of measurements.
                            Default is 'montecarlo'.

"""


def tau_conf(
    x,
    y,
    x_err=None,
    y_err=None,
    censors=None,
    p_conf=0.6826,
    n_samp=int(1e4),
    method="montecarlo",
):
    # check if censors exist, and if not, assume no censoring
    if np.any(censors == None):
        censors = np.ones((2, len(x)))
    # bootstrap sampling
    if method == "bootstrap":
        tau_boot = []
        for i in range(n_samp):
            inds = np.unique(np.random.randint(0, high=len(x) - 1, size=len(x)))
            tau_i, p_i = kendall(x[inds], y[inds], censors[:, inds])
            tau_boot += [tau_i]
        quants = np.quantile(tau_boot, [0.5 - 0.5 * p_conf, 0.5, 0.5 + 0.5 * p_conf])
        tau_lower, tau_upper = np.diff(quants)
    # MC sampling
    else:
        tau_mc = []
        x_mc = np.random.normal(size=(n_samp, len(x)), loc=x, scale=x_err)
        y_mc = np.random.normal(size=(n_samp, len(y)), loc=y, scale=y_err)
        for i in range(n_samp):
            tau_i, p_i = kendall(x_mc[i, :], y_mc[i, :], censors)
            tau_mc += [tau_i]
        quants = np.quantile(tau_mc, [0.5 - 0.5 * p_conf, 0.5, 0.5 + 0.5 * p_conf])
        tau_lower, tau_upper = np.diff(quants)
    return tau_lower, tau_upper
