# -*- coding: utf-8 -*-
'''
Support function for likelihood function computation and analysis
'''
import sys
import numpy as np
import pandas as pd
import scipy.integrate as sid
import lmfit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def log_likelihood_multinom(model, data):
    '''
    Multinomial log-likelihood function for survival data model fitting.
    
    This implements equation (10) in the GUTS paper [1].

    Survival as t->\infty goes to zero. This is handled by making
    a bin for the time interval  [tend, \infty) in which
    all the remaining animals are dead. Thus the probability
    adds up to one (over all possibilities)
    '''
    # Tail regularization parameter
    eps = 1e-9
    #model[model<0] = model[model>0].min()
    model[model<eps] = eps

    #Calculate time difference of model and data
    data_diff = np.diff(data[::-1])
    model_diff = np.diff(model[::-1])

    #Tail regularization: avoid increasing survival over time due to numerical
    #precision limits.
    model_diff[model_diff<eps] = eps

    #Any surviving animals at last time point?
    last_bin = 0
    if model[-1] > 0.0:
       # Then add contributions to likelihood function
       last_bin = data[-1] * np.log(model[-1])

    # Likelihood function from GUTS paper
    lik = np.sum(data_diff * np.log(model_diff)) + last_bin

    return lik


def log_likelihood_normal(model, data, sigma):
    '''
    Normal log-likelihood function for model fitting to data
    '''
    #Normal log likelihood
    lik = -0.5 * ((data - model)**2 / sigma**2 + np.ln(2*np.pi*sigma**2))

    return lik


def get_varying_params(params):
    return [var for var in params.keys() if params[var].vary]


def scan_likelihood_1d(params, objective, *likargs):
    '''Scan through values of each optimized parameter, keeping others fixed'''

    # Determine the parameters that have been fitted
    varypars = [p for p in params.keys() if params[p].vary]

    n = 2
    fig, axes = plt.subplots(n, n, figsize=(n*6, n*5))
    ax = axes.flatten()
    for i, p in enumerate(varypars):
        print('  Processing {0}'.format(p))
        # Optimized parameter value
        popt = params[p].value
        pmin = params[p].min

        vals = np.linspace(max(pmin, 0.1*popt), 10*popt, 30)
        curp = params.copy()
        likes = []
        for v in vals:
            curp[p].set(value=v)
            likes.append(objective(curp, *likargs))

        ax[i].plot(vals, likes, marker='.')
        ax[i].set_title(p)
        ax[i].axvline(popt, c='k')

    return fig, ax


def scan_likelihood_2d(params, objective, *likargs):
    '''Scan through values of pairs of optimized parameter, keeping others fixed'''

    # Determine the parameters that have been fitted
    varypars = [p for p in params.keys() if params[p].vary]
    n = len(varypars)
    N = 30

    # Set up required number of subplots
    #m = (n**2 - n)//2
    fig, ax = plt.subplots(n, n, figsize=(n*4, n*4))

    for i, p in enumerate(varypars):
        # Optimized parameter value
        popt = params[p].value
        pmin = params[p].min
        vals = np.linspace(max(pmin, 0.1*popt), 10*popt, N)
        for j, p2 in enumerate(varypars[i:]):
            #print('  Processing {0},{1}'.format(p, p2))
            popt2 = params[p2].value
            pmin2 = params[p2].min
            vals2 = np.linspace(max(pmin2, 0.2*popt2), 1.8*popt2, N)
            curp = params.copy()
            likes = np.zeros((N, N), dtype=np.double)
            pbar = tqdm(total=N+1, desc='Processing {0},{1}'.format(p, p2))
            likes1d = []
            for ii, v in enumerate(vals):
                curp[p].set(value=v)
                if (p != p2):
                    for jj, v2 in enumerate(vals2):
                        curp[p2].set(value=v2)
                        likes[ii, jj] = objective(curp, *likargs)
                else:
                    likes1d.append(objective(curp, *likargs))
                pbar.update(1)
            pbar.close()

            if p != p2:
                ax[j+i, i].pcolormesh(vals, vals2, likes, cmap=plt.cm.viridis)
                ax[j+i, i].set_xlabel(p)
                ax[j+i, i].set_ylabel(p2)
                curax = ax[j+i, i]
            else:
                ax[i, i].plot(vals, likes1d, marker='.')
                ax[i, i].set_xlabel(p)
                curax = ax[i, i]
            #curax.set_xticklabels(curax.get_xticklabels(), rotation=90)

    plt.tight_layout()
    return fig, ax


def profile_likelihood(params, fitfunc, objective, profpar, *likargs):
    maxiter = 40
    chisqrcrit = 3.8415 # From scipy.stats.chi2.isf(0.05, 1) [alpha, ddof]

    p = params.copy()

    # Calculate likelihood at parameters value, assumed to be the
    # maximum (from fit)
    Lmax = -objective(params, *likargs)

    # Optimal value for the profiled parameter
    popt = params[profpar].value

    # This is the initial step size to change the profiled parameter
    delta = 1e-2 * popt

    ll = []
    parvals = []
    prevpar = p
    done = False
    switch_time = False
    iters = 0
    curlL = Lmax
    prevL = Lmax
    while not done and iters<maxiter:
        if np.abs(delta) < (1e-4 * popt):
            print('  Delta too small, stopping')
            break
        # Next profile param value to try
        v = prevpar[profpar].value + delta

        # If we now are below the minimum value, decrease delta and try
        # again
        if v < params[profpar].min:
            #done = True
            print('  {0} below minimum value!'.format(profpar))
            delta = delta/3
            continue

        # If we now are above the maximum value, decrease delta and try
        # again
        if v > params[profpar].max:
            #done = True
            print('  {0} above maximum value!'.format(profpar))
            delta = delta/3
            continue

        print('  {2}: Now evaluating {0} = {1:.4g} (delta={3:.4g})'.format(profpar, v, iters, delta))

        # Fix parameter being profiled at next value
        prevpar[profpar].set(value=v, vary=False)

        # Maximize likelihood for reduced parameter set
        res = fitfunc(objective, prevpar, *likargs, printres=False,
                      progressbar=False)
        #print(lmfit.report_fit(res))
        print('    ' + res.message)

        # Calculate likelihood at found optimum
        curL = -objective(res.params, *likargs)

        # Calculate likelihood ratio criterion
        # This is equation 36 in the TKTD course refresher
        likrat = 2*(Lmax - curL)

        print('    L = {0:.4g}, dL = {1:.4g}'.format(curL, likrat))

        # Negative change?
        #if (likrat < 0):
        #    # Then try smaller delta
        #    print('    Negative change! Trying smaller delta')
        #    # Reset value to previous
        #    prevpar[profpar].set(value=v-delta, vary=False)
        #    delta = delta/2
        #    continue

        # Is the change bigger than threshold (1% of previous)?
        #if np.abs(likrat) > np.abs(0.01*Lmax):
        if np.abs(curL - prevL) > np.abs(0.01 * prevL):
            # Change too big (doubles), halve delta and discard current iteration
            #if np.abs(likrat)>2:
            if np.abs(curL - prevL) > np.abs(prevL):
                delta = delta/2

                # If the new delta is too small, we keep this value and
                # move to the negative branch, or stop.
                if np.abs(delta) < (1e-4 * popt):
                    if delta < 0:
                        done = True
                        continue
                    else:
                        # Keep this iteration, switch to negative branch
                        switch_time = True
                else:
                    print('    Change to big, decreasing delta (to {0:.4g})'.format(delta))

                    # Reset value to previous
                    prevpar[profpar].set(value=v-delta, vary=False)
                    #ll.append(likrat)
                    #parvals.append(v)
                    continue
        else:
            # Change to small, increase step size, but keep iteration results
            delta = delta*2
            print('    Change to small, increasing delta (to {0:.4g})'.format(delta))

        ll.append(likrat)
        parvals.append(v)

        # Keep the parameters from this fit, use as starting point for
        # next iteration
        prevpar = res.params
        prevL = curL

        # Are we done?
        #if np.abs(max(ll)-min(ll)) > 2:
        if np.abs(likrat) > chisqrcrit:
            if delta < 0:
                done = True
            else:
                switch_time = True

        # Change to negative delta at halfway to maxiter
        if (iters*2 == maxiter) or (switch_time):
            print('    -->Switching to lower range for {0}<--'.format(profpar))
            switch_time = False
            # Reset to optimal parameter values
            prevpar = params.copy()
            delta = -1e-3 * popt
            prevL = Lmax

        iters += 1

    print('\n done!')

    return parvals, ll


def profile_likelihood_simple(params, fitfunc, objective, profpar, *likargs):
    '''Profile using a log-spaced grid of values'''
    prevpar = params.copy()

    # Calculate likelihood at parameters value, assumed to be the
    # maximum (from fit)
    Lmax = -objective(params, *likargs)

    # Optimal value for the profiled parameter
    popt = params[profpar].value

    # This is the initial step size to change the profiled parameter
    delta = 1e-2 * popt
    valsl = [popt - delta*i**2 for i in range(0, 11)]
    valsu = [popt + delta*i**2 for i in range(1, 11)]

    # Rescale the lower one
    pmin = params[profpar].min
    #vasls / max(valsljj

    ll = []
    iters = 0
    for v in valsu:
        print('  Now evaluating {0} = {1} (iteration #{2})'.format(profpar, v, iters))

        # Fix parameter being profiled at next value
        prevpar[profpar].set(value=v, vary=False)

        # Maximize likelihood for reduced parameter set
        res = fitfunc(objective, prevpar, *likargs, printres=False)

        print('    ' + res.message)

        # Calculate likelihood at found optimum
        curL = -objective(res.params, *likargs)

        # Calculate likelihood ratio criterion
        # This is equation 36 in the TKTD course refresher
        likrat = 2*(Lmax - curL)
        ll.append(likrat)

        print('    L = {0}, dL = {1}'.format(curL, likrat))

        # Keep the parameters from this fit, use as starting point for
        # next iteration
        prevpar = res.params

        iters += 1

    print('  Switching to lower range')
    prevpar = params.copy()

    for i, v in enumerate(valsl):
        print('  Now evaluating {0} = {1} (iteration #{2})'.format(profpar, v, iters))
        if v<pmin:
            valsl = valsl[:i]
            print('  Smallest allowed value of parameter reached, stopping')
            break

        # Fix parameter being profiled at next value
        prevpar[profpar].set(value=v, vary=False)

        # Maximize likelihood for reduced parameter set
        res = fitfunc(objective, prevpar, *likargs, printres=False)

        print('    ' + res.message)

        # Calculate likelihood at found optimum
        curL = -objective(res.params, *likargs)

        # Calculate likelihood ratio criterion
        # This is equation 36 in the TKTD course refresher
        likrat = 2*(Lmax - curL)
        ll.append(likrat)

        print('    L = {0}, dL = {1}'.format(curL, likrat))

        # Keep the parameters from this fit, use as starting point for
        # next iteration
        prevpar = res.params
        iters += 1


    print('  Done!')

    vals = np.array(valsu + valsl)
    idx = np.argsort(vals)

    return vals[idx], np.array(ll)[idx]


def profile_likelihood_all(neglnlike, fitfunc, params, data, exposure):
    prof_vars = [var for var in params.keys() if params[var].vary]
    profiles = dict()
    for curvar in prof_vars:
        print('Now profiling with {0} fixed'.format(curvar))
        vals, plik = profile_likelihood(params, fitfunc,
                                               neglnlike, curvar,
                                               *(data, exposure))
        profiles[curvar] = [vals, plik]

    return profiles


def plot_likelihood_profiles(profiles, res):
    '''Plot profiles for all parameters'''
    n = len(profiles)
    fig, ax = plt.subplots(1, n, figsize=(6*n, 6))
    chisqrcrit = 3.8415 # From scipy.stats.chi2.isf(0.05, 1) [alpha, ddof]
    pvarnames, pvals = zip(*profiles.items())
    varorder = np.argsort(pvarnames)
    #for i, (k, v) in enumerate(zip(pvarnames[varorder], pvals[varorder])):
    for i, j in enumerate(varorder):
        k = pvarnames[j]
        v = pvals[j]
        idx = np.argsort(v[0])
        ax[i].plot(np.array(v[0])[idx], np.array(v[1])[idx], '-', marker='o')
        ax[i].set_xlabel(k)
        ax[i].axvline(res.params[k].value, color='k', linestyle='--')
        ax[i].axhline(chisqrcrit, color='k', linestyle='--')
        ax[i].set_ylim(0, 4)
    plt.tight_layout()
    return fig, ax
