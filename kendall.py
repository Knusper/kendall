# kendall.py
# Released under GPLv3 license (see LICENSE).
# Copyright: E.C. Herenz (2024), S. Flury (2023)

import numpy as np
from scipy.special import erf
from tqdm import tqdm


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
      If `True` (default), upper limits are assumed.  Otherwise, censored data are regarded as
      lower limits.

    Returns
    -------

    result : tuple
     (tau, p) -  with tau being Kendall's tau and p being the associated p-value.

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
    # (a_ij in Isobe = -a_ij in Akritas; does not matter in the calculations
    # as they are invariant to skew-symmetry, but we follow Isobe)
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
        # - sums involving a_ij a_ik & b_ij b_ik
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

        # - sums involving a_ij**2 & b_ij**2
        A_2 = np.sum(J[0, :, :] ** 2)
        B_2 = np.sum(J[0, :, :] ** 2)
        A_1 -= A_2
        B_1 -= B_2

        # final sum
        var_1 = (4 * A_1 * B_1) / (n * (n - 1) * (n - 2))
        var_2 = (2 * A_2 * B_2) / (n * (n - 1))
        var = var_1 + var_2

        # z-score -> comparison to standard normal distribution -> p-value
        p = 1 - erf(abs(S) / np.sqrt(2 * var))

    return tau, p


def tau_conf(
    x,
    y,
    x_err=None,
    y_err=None,
    censors=None,
    n_samp=1e4,
    method="bootstrap",
    upper=True,
    return_tau_dist=False,
):
    """Determine confidence intervals on Kendall's tau computed withe the function `kendall`
    either by bootstrapping values of x and y, or by a Monte-Carlo simulation using
    error-bars on x and/or y.

    Parameters
    ----------

    x : numpy.ndarray
      1D array of sample values of length N (measured values or detection limits)

    y : numpy.ndarray
      1D array of paired sample values to y (measured values or detection limits)

    x_err : NoneType | np.ndarray
      1D errors of uncertainties associated with x.

    y_err : NoneType | np.ndarray
      1D errors of uncertainties associated with y.

    censors: np.ndarray | NoneType
      2xN array containing censors of `x` and `y` with 1 representing an uncensored
      datum and 0 representing a left-censored (upper-limit) datum.  If `None` is given
      (default) all values of x and y are assumed to be uncensored uncensored (2xN array
      of 1s).

    p_conf: float
      Two-sided probability interval of the desired confidence
      interval. Default is 0.68 (~ 1-sigma).

    n_samp: int
      Number of bootstrap or Monte-Carlo samples.  Default is 1e4.

    method: {"montecarlo", "bootstrap"}
      Method to use for calculating confiendence (default: "bootstrap").  If
      "montecarlo" is used, `x_err` and `y_err` are required, for "bootstrap" `x_err`
      and `y_err` are ignored.

    return_tau_dist: bool
      Wether or not to return the full sample of tau results.

    Returns
    -------
    result : tuple

    If `return_tau_dist=False`:
      `tau_q25, tau_median, tau_q75 : float, float`
      Lower quartile, median, and upper quartile of tau distribution.

    If `return_tau_dist=True`:
      `tau_lower, tau_upper, np.ndarray`
      Lower quartile, median, and upper quartile of tau distribution, as well as array
      of all tau values drawn in the Monte-Carlo or bootstrap experiment

    Notes
    -----

    Both formalisms, i.e. bootstrapping and Monte-Carlos simulation, for estimating the
    robustness of the obtained correlation coefficient are explained in [Curan (1994;
    arXiv:1411.3816)][1].

    [1]: https://doi.org/10.48550/arXiv.1411.3816

    """
    n_samp = int(n_samp)
    
    # check if censors exist, and if not, assume no censoring
    if np.any(censors == None):
        censors = np.ones((2, len(x)))

    tau_dist = np.empty(n_samp)

    # in both methods we fix varcalc == simple, since we're only interested in tau
    if method == "bootstrap":
        # boot-strap
        print("Doing bootstraps for n_samp=" + str(n_samp))
        for i in tqdm(range(n_samp)):
            inds = np.unique(np.random.randint(0, high=len(x) - 1, size=len(x)))
            tau_i, _ = kendall(
                x[inds],
                y[inds],
                censors[:, inds],
                varcalc="simple",
                upper=upper,
            )
            tau_dist[i] = tau_i

    elif method == "montecarlo" and (np.any(x_err != None) or np.any(y_err != None)):
        # allow for only x or y errors being passed
        if np.any(x_err == None):
            x_err = np.zeros_like(x)
        if np.any(y_err == None):
            y_err = np.zeros_like(y)
        # monte-carlo simulation
        x_mc = np.random.normal(size=(n_samp, len(x)), loc=x, scale=x_err)
        y_mc = np.random.normal(size=(n_samp, len(y)), loc=y, scale=y_err)
        if np.any(censors == None):
            # do not alter the upper limits (TODO: this could be optional, as the upper limit is a
            # statistic as well)
            x_mc[~censors[0, :].astype(bool)] = x[~censors[0, :].astype(bool)]
            y_mc[~censors[1, :].astype(bool)] = y[~censors[1, :].astype(bool)]

        print("Running MC simulation for n_samp = " + str(n_samp))
        for i in tqdm(range(n_samp)):
            tau_i, _ = kendall(
                x_mc[i, :], y_mc[i, :], censors, varcalc="simple", upper=upper
            )
            tau_dist[i] = tau_i

    else:
        raise ValueError(
            "Method parameter not understood, or neither x_err nor y_err defined"
        )

    tau_q25, tau_median, tau_q75 = np.quantile(tau_dist, [0.25, 0.5, 0.75])

    if return_tau_dist == True:
        return tau_q25, tau_median, tau_q75, tau_dist
    else:
        return tau_q25, tau_median, tau_q75
