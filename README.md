# Kendall's tau for censored data in python

The python functions in `kendall.py` calculate a non-parametric correlation coefficient
(Kendall's τ), that measures the strength of correlation for a paired sample of ordinal
level data, that may be partially censored (either with upper- or lower- limits, but not
with mixed upper- and lower limits).  It can also be used as a statistical test to rule
out the null-hypothesis that the two variables are uncorrelated.
  
The calculation of tau and the p-value follow the calculation of [Isobe, Feigelson, and
Nelson (1986; ApJ 306:490)][1].  Originally this formalism was developed in the context
of medical science[^1] by [Brown, Holander & Korwar (1974)][3].  With respect to
partial correlations the formalism is also given in [Akritas & Seibert (1996; MNRAS
278, 919)][2], but with a more compact notation compared to Isobe et al (1986).

The p-value calculation requires the distribution and the variance of τ under the null-hypothesis.  For uncensored data and large enough n the distribution can be approximated by a normal distribution (e.g., [Wikipedia][4]).  In this case the resulting expression depends only on the sample size[^2].  (For small samples and uncensored data the distribution depends on n and C, where C is the number of concordant pairs, but the somewhat cumbersome computation is not yet implemented here; `scipy` provides it since ~2019 and conservatively assumes n<50 as small -- see the [resolved issue at github][gh].)
That being said, when using this code for small samples with uncensored data the here provided p-values should be treated with caution.  Use `scipy.stats.kendalltau` instead.

For censored data and large n the distribution of τ under the null-hypothesis is approximately normal as well, but the variance depends on the distribution of censored values with respect to the sample proportions [(Oakes 1982; Biometrics 38, 451)][5].  Thus, in practice, an estimate of the variance from the data is required.  This code follows the approach of Isobe et al. and Brown et al., but more refined approaches exists in the literature.  An example developed with astronomical data in mind is given by [Akritas, Murphy, and LaValley (1995; J Am Stat Assoc 429, 170)][6]; as of yet the computation of p-values with this variance estimator is only implemented in R as part of the package [NADA][NADA] (routine `cenken`).  This formalism also support simultaneously left- and right- censored data[^3].

[1]: https://doi.org/10.1086/164359
[2]: https://doi.org/10.1093/mnras/278.4.919
[3]: https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD767617.xhtml
[4]: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Hypothesis_test
[5]: https://doi.org/10.2307/2530458
[6]: https://doi.org/10.1080/01621459.1995.10476499
[gh]: https://github.com/scipy/scipy/issues/8456
[NADA]: https://www.rdocumentation.org/packages/NADA/

[^1]: Survival time comparison between patients receiving a heart transplant with patients not receiving such treatment.
[^2]: In the case of censored data and small n, you are advised to be very cautios with the p-values computed here.  In scipy.stats.kendalltau, perhaps conservatively, small n means n < 50.
[^3]: If someone is clever enough to understand this and to implement it here, PRs are very welcome.  From numerical experiments with my sample (Herenz et al. 2024), I find that the calculation of the p-values with this routine are slightly larger and, likely, more robust.

Additional functionality is included to determine the robustness of the correlation coefficient to individual datum (done by bootstrapping) or uncertainties in the data (done by Monte Carlo sampling).  A description of the idea beyond these procedures can be found in [Curan (2015, arXiv:1411.3816)][Curan].

[Curan]: https://arxiv.org/abs/1411.3816

## Provided functions

`kendall(x, y, censors=None, varcalc="simple", upper=True)` 

`tau_conf( x, y, x_err=None, y_err=None, censors=None, p_conf=0.6826, n_samp=int(1e4), method="montecarlo", varcalc="simple", upper=True, )`

See online help of those function (or source code) for notes on their usage.

# History of this code

A python implementation of the Isobe et al. algorithm was initially written by S. Flury
for work presented in [Flury et al. (2022)][7].  This code assumed the theoretical value for
the variance in the case of uncensored data and large n.  E.C. Herenz modified the code
to use the empirical variance calculation as described in [Isobe et al. (1986)][1] for work
presented in [Herenz et al. (2024)][8].

[7]: https://doi.org/10.3847/1538-4357/ac61e4
[8]: https://ui.adsabs.harvard.edu/abs/2024arXiv240603956H/abstract

# Acknowledging the use of the code

If your research benifits from this code, please cite Isobe et al. (1986) and include a
link to this github repository. 

