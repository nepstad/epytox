# -*- coding: utf-8 -*-
'''
Tests for GUTS module.
'''
import numpy as np
import pandas as pd
from epytox import guts


def test_guts_sic():
    #Test model intialization
    guts_sic = guts.SIC()
    assert guts_sic.params['ke'].value > 0

    #Test model predict (forward solve)
    cd = lambda t: [0.1]
    y, r = guts_sic._solve(guts_sic.params, y0=[0.0], 
                           times=np.linspace(0, 50, 10), cd=cd)
    assert np.abs(y[-1] - 0.1) < 1e-7


def test_guts_sic_sd():
    #Test model intialization
    m = guts.SIC_SD()
    ke0 = m.params['ke'].value
    assert m.params['ke'].value > 0
    assert m.params['b'].value > 0
    assert m.params['c0s'].value > 0
    assert m.params['hb'].value > 0

    #Test model predict (forward solve)
    cd = lambda t: [0.1]
    y, r = m._solve(m.params, y0=[0.0, 1.0], 
                           times=np.linspace(0, 50, 10), cd=cd)
    assert np.abs(y[-1, 0] - 0.1) < 1e-7

    #Test model fitting (to some fake data)
    datasets = [[lambda t: [0.1], pd.Series(index=[0,1,2,3], data=[1., .9, .8, .7])], ]

    res = m.fit(datasets, False, False)
    assert res.params['ke'].value != ke0
    print(res.params)
