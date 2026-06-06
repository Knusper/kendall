# Kendall's tau for censored data in python

<img src="./tau_chatgpt_logo.png" alt="Tau logo" align="left" height="300">

The python function `kendall` in `kendall.py` calculates a non-parametric correlation coefficient (Kendall's τ).  Kendall's τ measures the correlation strength for a paired sample of ordinal level data.  Here, unlike in [`scipy.stats.kendalltau`][sptau], the data of *N* pairs may be partially censored (either with upper- or lower- limits, but not with mixed upper- and lower limits).  Kendall's τ can also be used as a statistical test to rule out the null-hypothesis that the two variables are uncorrelated.  This statistical test ([Kendalls' τ test][sptautest]) requires the calculation of the p₀-value.

The calculation of τ and the p₀-value follow  [Isobe, Feigelson, and Nelson (1986)][1].  Originally this formalism was developed in the context of medical science[^1] by [Brown, Holander & Korwar (1974)][3].  With respect to partial correlations the formalism is also presented in [Akritas & Seibert (1996)][2].  For now, we only provide an implementation of the Akritas & Seibert partial-correlation coefficent p₀-values for uncensored data.

The p₀-value calculation requires the distribution and the variance of τ under the null-hypothesis.  For uncensored data and a large enogh *N* the distribution can be approximated by a Normal distribution.  In this case the resulting expression depends only on the sample size[^2].  We thus advise against using the p₀-values calculated with our routine for small samples with uncensored data.  Please use [`scipy.stats.kendalltau`][sptau] instead.

For censored data and large *N* the distribution of τ under the null-hypothesis is approximately normal as well, but the variance depends on the distribution of censored values with respect to the sample proportions [(Oakes 1982)][5].  Thus, in practice, an estimate of the variance from the data is required.  This code follows the approach of Isobe et al. and Brown et al., but more refined approaches exists in the literature.  An example developed with astronomical data in mind is given by [Akritas, Murphy, and LaValley (1995)][6].  This formalism also support simultaneously left- and right- censored data.   The computation of p-values with a sample dependent variance estimator is implemented in R as part of the package [NADA][NADA] (routine `cenken`).  As of yet, an implementation in python remains desireable[^3].

Additional functionality is included with the function `tau_conf`. This function determine the robustness of the correlation coefficient due to each individual datum (done by bootstrapping) or uncertainties in the data (done by Monte Carlo sampling).  A description of the idea beyond these procedures can be found in [Curan (2015, arXiv:1411.3816)][Curan]. 

For the calculation of partial correlleations (τ(12.3); here currently implemented only for uncensored data) we follow Akritas & Seibert (1996).  They provide an recipe for estimating the expectation value of the variance under the null-hypothesis that τ(12) = τ(13)×τ(23) from the data, and then assume that the sample is large enough such that Var(τ(12.3)) equals the variance of the Normal distribution.

[^1]: Survival time comparison between patients receiving a heart transplant with patients not receiving such treatment.
[^2]: For small samples and uncensored data the distribution can not be written down in closed form.  It requires evaluation of all possible permutations of the N pairs under the null hypothesis.  Then the calculation of the p-value requires the the calulation of all |τ| values for these permutations.  While some trickery can simplify this calculation, it is not yet implemented here; `scipy` provides it since ~2019 and conservatively assumes n<50 as small (R uses n<60) -- see the [resolved issue at github][gh].  In practice, I found that for n = 15 the critical |τ| values for the threshold p=0.05 differ by ≈ 10⁻².  Critical values for |τ| for given p-values are found also tabulated in the statistical literature. 
[^3]: PRs are very welcome. The existing code in NADA is spagghetti-like confusion and the acompanying book does not shed any light on the issue. 


[1]: https://doi.org/10.1086/164359
[2]: https://doi.org/10.1093/mnras/278.4.919
[3]: https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD767617.xhtml
[4]: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Hypothesis_test
[5]: https://doi.org/10.2307/2530458
[6]: https://doi.org/10.1080/01621459.1995.10476499
[NADA]: https://www.rdocumentation.org/packages/NADA/
[sptau]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
[sptautest]: https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_kendalltau.html
[gh]: https://github.com/scipy/scipy/issues/8456
[Curan]: https://arxiv.org/abs/1411.3816

## Provided functions

`kendall(x, y, censors=None, varcalc="simple", upper=True)` 

`tau_conf(x, y, x_err=None, y_err=None, censors=None, p_conf=0.6826, n_samp=int(1e4), method="montecarlo", varcalc="simple", upper=True)`

`partial_corr(T1, T2, T3)`

See online help of those function (or the source code) for notes on their usage.

# History of this code 

A python implementation of the Isobe et al. algorithm was initially written by S. Flury
for work presented in [Flury et al. (2022)][7].  This code assumed the theoretical value for
the variance in the case of uncensored data and large n.  E.C. Herenz modified the code
to use the empirical variance calculation as described in [Isobe et al. (1986)][1] for work
presented in [Herenz et al. (2025)][8].

[7]: https://doi.org/10.3847/1538-4357/ac61e4
[8]: https://doi.org/10.1051/0004-6361/202451012

# Requirements

- numpy - https://numpy.org/
- scipy - https://scipy.org/
- tqdm - https://tqdm.github.io/

# Acknowledging the use of the code

If your research benifits from this code, please cite Isobe et al. (1986) or Akritas &
Seibert (1996).  A link to this repository in your paper would be appreciated.

## Copyright

The code is released under GPLv3 license (see LICENSE).
Copyright: E.C. Herenz (2024), S. Flury (2023)

## State on the use of generative LLM's

OpenAI's ChatGPT was used for creating the logo.  Code and Readme are written by hand. 
