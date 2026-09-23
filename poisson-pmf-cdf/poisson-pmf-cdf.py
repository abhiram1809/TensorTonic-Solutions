import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    def calc_specific_pmf(lam, k):
        return ((lam**k) * math.exp(-1*lam))/math.factorial(k)
    pmf = calc_specific_pmf(lam, k)
    cdf = sum([calc_specific_pmf(lam, i) for i in range(k+1)])

    return {"pmf":pmf, "cdf":cdf}