# -*- coding: utf-8 -*-
'''
Exposure profile functions.
'''
import numpy as np
from scipy import interpolate


def step_treatment_generator(c0, tswitch):
    def step_treatment(t):
        '''Step function treatment, constant up to tswitch, then zero'''
        return t>tswitch and 0.0 or c0
    return step_treatment


def multi_constant_treatment_generator(treatments):
    '''Create functions of constant exposure from list'''
    treatment_funcs = {}
    for treat in treatments.columns:
        f = step_treatment_generator(treatments[treat], -np.inf)
        f.__name__ = 'treatment_{0}'.format(treat.lower())
        treatment_funcs[treat] = f
    return treatment_funcs


def multi_interp_treatment_generator(treatments):
    '''Create interpolating treatment functions from data'''
    treatment_funcs = {}
    for treat in treatments:
        t = treatments[treat].index
        #Get concentration values
        conc = treatments[treat].values
        
        #Create interpolationg spline function
        s = interpolate.InterpolatedUnivariateSpline(t, conc, k=1)
        s.__name__ = 'treatment_{0}'.format(treat.lower())
        treatment_funcs[treat] = s
    return treatment_funcs
