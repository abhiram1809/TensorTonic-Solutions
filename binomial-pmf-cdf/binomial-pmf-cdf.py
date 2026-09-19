import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    def comb(num, times):
        return math.factorial(num)/(math.factorial(times)*math.factorial(num-times))
    def comp_pmf(bn_c, n, p, k):
        return bn_c*(p**k)*((1-p)**(n-k))
    bn_c = comb(n, k)
    pmf = comp_pmf(bn_c, n, p, k)
    cumulatives = [comp_pmf(comb(n, iter), n, p, iter) for iter in range(k+1)]
    cdf = sum(cumulatives)
    return {"pmf": pmf, "cdf": cdf}
    