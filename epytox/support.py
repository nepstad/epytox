import numpy as np

def mortality_lognormal(r, s):
    '''Calculate mortality from cumulative log-normal distribution

    Keyword arguments:
    :param r: ratio of body burdens to cbr, summed (dimensionless)
    :param s: dose-response slope (dimensionless)
    :returns: mortality fraction (fraction)
    '''
    if r>0:
        mean = 0.0
        x = (np.log10(r) - mean) / (s * np.sqrt(2))
        return 0.5 * (1 + erf(x))
    else:
        return 0.0

def mortality_loglogistic(conc, alpha, beta):
    '''Calculate mortality from cumulative log-logistic distribution

    Keyword arguments:
    :param conc: internal concentration ()
    :param alpha: threshold level ()
    :param beta: shape parameter ()
    :returns: F, cumulative log-logistic
    '''
    if conc>0:
        return 1.0 / (1 + (conc/alpha)**-beta)
    else:
        return 0.0

