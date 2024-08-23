import numpy as np
from scipy.special import erf


def kendall(x, y, censors=None, varcalc="simple", upper=True):
    """Compute Kendall's tau and associated p-value for censored paired data (i.e., data
    with upper or lower limits).  Kendall's tau is a correlation measure for ordinal data.

    Data has to be either left- or right censored (upper limits or lower limits).

    Parameters
    ----------

    x : numpy.ndarray
      1D array of sample values of length N (measured values or detection limits)

    y : numpy.ndarray
      1D array of paired sample values to y (measured values or detection limits)

    censors: np.ndarray | NoneType
      2xN array containing censors of `x` and `y` with 1 representing an uncensored
      datum and 0 representing a left-censored (upper-limit) datum.  If `None` is given
      (default) all values of x and y are assumed to be uncensored uncensored (2xN array
      of 1s).

    varcalc : {'simple', 'ifn'}
      Method used to estimate the p-value (see Notes).

      * `simple`: Uses the asymptotic approximation of the variance for uncensored data.
      * `ifn`: Uses the approximation from Isobe, Feigelson, and Nelson (1986)

    upper : bool
      If `True` (default), it will

    Notes
    -----

    The calculation of tau and p follows the calculation presented in [Isobe, Feigelson,
    and Nelson (1986; ApJ 306:490)][1].  Originally this formalism was developed in a
    biostatistics context by [Brown, Holander & Korwar (1974)][3].  In the context of
    partial correlations it has also been presented in [Akritas & Seibert (1996; MNRAS
    278, 919][2], but with a more compact notation.

    [1]: https://doi.org/10.1086/164359
    [2]: https://doi.org/10.1093/mnras/278.4.919
    [3]: https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD767617.xhtml

    For the calculation of the p-value the distribution and the variance of tau under the
    null-hypothesis (no correlation) is required.  For uncensored data and large enough
    n the distribution can be appoximated by a normal distribution (e.g.,
    [Wikipedia][4]) and the resulting expression depends only on the sample size n.
    While for censored data the distribution is approximately normal as well, the
    variance depends on the distribution of censored values with respect to the sample
    proportions [(Oakes 1982; Biometrics 38, 451)][5].

    [4]: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Hypothesis_test
    [5]: https://doi.org/10.2307/2530458

    """

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


def tau_conf(
    x,
    y,
    x_err=None,
    y_err=None,
    censors=None,
    p_conf=0.6826,
    n_samp=int(1e4),
    method="montecarlo",
    varcalc="simple",
    upper=True,
):
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
    # check if censors exist, and if not, assume no censoring
    if np.any(censors == None):
        censors = np.ones((2, len(x)))

    # bootstrap sampling
    if method == "bootstrap":
        tau_boot = []
        for i in range(n_samp):
            inds = np.unique(np.random.randint(0, high=len(x) - 1, size=len(x)))
            tau_i, p_i = kendall(
                x[inds], y[inds], censors[:, inds], varcalc=varcalc, upper=upper
            )
            tau_boot += [tau_i]
        quants = np.quantile(tau_boot, [0.5 - 0.5 * p_conf, 0.5, 0.5 + 0.5 * p_conf])
        tau_lower, tau_upper = np.diff(quants)
    # MC sampling
    else:
        tau_mc = []
        x_mc = np.random.normal(size=(n_samp, len(x)), loc=x, scale=x_err)
        y_mc = np.random.normal(size=(n_samp, len(y)), loc=y, scale=y_err)
        for i in range(n_samp):
            tau_i, p_i = kendall(
                x_mc[i, :], y_mc[i, :], censors, varcalc=varcalc, upper=upper
            )
            tau_mc += [tau_i]
        quants = np.quantile(tau_mc, [0.5 - 0.5 * p_conf, 0.5, 0.5 + 0.5 * p_conf])
        tau_lower, tau_upper = np.diff(quants)

    return tau_lower, tau_upper
