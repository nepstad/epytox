# -*- coding: utf-8 -*-
'''
Survival (toxico-dynamics) models, forward simulation and model fitting.

References
----------
    [1] Jager, T. et al. (2011). General unified threshold model of survival -
        a toxicokinetic-toxicodynamic framework for ecotoxicology.  
        Environmental Science & Technology, 45(7), 2529-2540.
'''

import sys
import numpy as np
import pandas as pd
import scipy.integrate as sid
from scipy.special import erf
import lmfit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import corner


#ODE solver settings
ATOL = 1e-9
MXSTEP = 1000


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


def guts_sic(y, t, ke, cd):
    '''One-compartment scaled internal concentration ODE (rhs)'''

    # One-compartment kinetics model for body residues
    dy = ke*(cd(t) - y)

    return dy


def guts_sic_sd(y, t, params, cd, dy):
    '''GUTS-SIC-SD: Scaled internal concentration + hazard rate survival ODE (rhs)'''
    v = params
    n = y.size - 1

    # One-compartment kinetics model for body residues
    dcv = guts_sic(y[:n], t, v['ke'], cd)

    #Dose metric
    cstot = np.sum(y[:n])

    #Hazard rate
    hc = v['b'] * max(0, cstot - v['c0s'])
    h = hc + v['hb']
    ds = -h * y[n]

    dy[:n] = dcv
    dy[n] = ds

    return dy


def solve_guts_sic_sd(params, y0, times, cd):
    '''Solve the GUTS-SIC-SD ODE.'''
    v = params.valuesdict()
    dy = y0.copy()
    rhs = guts_sic_sd
    y = sid.odeint(rhs, y0, times, args=(v, cd, dy), atol=ATOL, mxstep=MXSTEP)
    return y


def solve_guts_sic_it(params, y0, times, cd):
    '''Solve the GUTS-SIC-IT ODE.

    Scaled internal concentration, individual tolerance
    '''
    v = params.valuesdict()

    #Solve uptake kinetics for internal concentrations
    y = sid.odeint(guts_sic, y0, times, args=(v['ke'], cd), atol=ATOL,
                   mxstep=MXSTEP)

    #Number of body residues
    n = ystep.shape[1] - 1
    for i, ystep in enumerate(y):
        if i == 0:
            continue

        #Total internal concentration
        cstot = np.sum(ystep[:n])

        #Calculate survival from ratio of internal concentration to 
        #tolerance threshold.
        surv = y[0, n] * (1.0 - mortality_lognormal(cstot/v['cbr'], v['s']))

        #Survival cannot increase with time
        y[i, n] = min(y[i-1, n], surv)
    return y


def get_model_prediction(times, params, exposure, s0=1,
                         solver=solve_guts_sic_sd):
    v = params.valuesdict()
    n = exposure.shape[0]

    #Evaluate model for each exposure concentration
    model_pred = []
    for col in exposure.columns:
        cd = lambda t: exposure[col].values

        # Initial conditions: zero internal concentration, 100% survival
        y0 = np.array([0.0]*n  + [s0])

        # Evaluate model at data time points with present parameter set
        y = solver(params, y0, times, cd)

        model_pred.append(y)

    return model_pred


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


def objective(params, data, exposures):
    '''Negative log-likelihood objective function for survival.
    
    Loop over exposures in the data, and calculate negative log likelihoods.
    '''
    #n = exposures.shape[0]
    n = 1 #@todo Infer number of components here
    y0 = np.zeros(n+1, dtype=np.double)
    times = data.index.values
    negloglik = 0
    for treatment in exposures:
        #Define exposure function
        #cd = lambda t: np.array([np.double(curexp)])
        cd = exposures[treatment]

        #Evaluate model, keep only survival value
        y0[:] = [0.0, 1.0]
        y = solve_guts_sic_sd(params, y0, times, cd)[:, 1]
        
        negloglik -= log_likelihood_multinom(y, data[treatment].values)

    return negloglik


def fit(objective, params, data, exposure, printres=True, progressbar=True):
    '''Fit model to data via likelihood function (objective)'''
    maxiter = 3000
    if progressbar:
        pbar = tqdm(total=maxiter)
        def objective_wrapper(*args, **kwargs):
            pbar.update(1)
            return objective(*args, **kwargs)
    else:
        objective_wrapper = objective

    # Minimize the objective function using the simplex method
    mini = lmfit.Minimizer(objective_wrapper, params,
                           fcn_args=(data, exposure))
    result = mini.minimize(method='nelder', params=params, tol=1e-8,
                           options=dict(maxfev=maxiter, maxiter=maxiter, 
                                        xatol=1e-8, fatol=1e-8))

    if progressbar:
        pbar.close()

    # Print result of fit
    if printres:
        print(result.message)
        lmfit.report_fit(result)

    return result


def mcmc(objective, params, data, exposure, nsteps=10000, nwalkers=None,
        progressbar=True):
    if not nwalkers:
        nwalkers = 2*len([n for n, p in params.items() if p.vary])
    print('Commencing MCMC with {0} walkers, {1} steps'.format(nwalkers, nsteps))

    maxiter = nsteps * nwalkers
    if progressbar:
        pbar = tqdm(total=maxiter)
        def objective_wrapper(*args, **kwargs):
            pbar.update(1)
            return objective(*args, **kwargs)
    else:
        objective_wrapper = objective

    mini = lmfit.Minimizer(objective_wrapper, params, 
                           fcn_args=(data, exposure))
    res = mini.emcee(burn=0, steps=nsteps, thin=1, nwalkers=nwalkers, 
                     params=params)

    if progressbar:
        pbar.close()

    return res


def plot_mcmc(mcmc_res, res, burn=0):
    # Make corner plot of 1D and 2D projection of the samled values
    c = corner.corner(mcmc_res.flatchain, labels=mcmc_res.var_names,
              truths=list(res.params.valuesdict().values()), verbose=True,
              show_titles=True)

    # Plot trajectory of each walker in parameter space
    n = mcmc_res.chain.shape[2]
    fig, ax = plt.subplots(n, 1, figsize=(8, 4*n))
    for i in range(n):
        _ = ax[i].plot(mcmc_res.chain[:, burn:, i].T, 'k-', alpha=0.1)
        ax[i].set_ylabel(mcmc_res.var_names[i])
    ax[-1].set_xlabel('Step number')

    return fig, ax, c


def plot_fit(params, data, exposure, subplots=True):
    '''Plot the data and calculated model fit'''
    colors = sns.color_palette('Set2', data.shape[1])
    if subplots:
        n = data.shape[1]
        fig, ax = plt.subplots(n, 2, figsize=(9, 4*n))
    else:
        n = 1
        fig, ax = plt.subplots(n, 2, figsize=(17, 6*n))
        ax = ax.reshape((n, 2))

    times = np.linspace(0, data.index[-1], 100)
    for i, treatment in enumerate(data.columns):
        if subplots:
            idx = i
        else:
            idx = 0
        cur_color = colors.pop()

        #Exposure profile in this treatment
        cd = exposure[treatment]

        ax[idx, 1].plot(data.index, data[treatment], 'k', ls=':', marker='o',
                   mfc=cur_color, mec='k', mew=1)

        # Initial conditions: zero internal concentration, 100% dataival
        y0 = [0] * 1 + [data[treatment].iloc[0]]

        # Evaluate model at data time points with present parameter set
        y = solve_guts_sic_sd(params, y0, times, cd)
        ax[idx, 1].plot(times, y[:, -1], color=cur_color)
        ax[idx, 0].plot(times, np.sum(y[:, :-1], axis=1), color=cur_color,
                label=treatment)

        if subplots:
            ax[idx, 0].set_title(treatment)
            ax[idx, 1].set_title(treatment)
        else:
            ax[idx, 0].legend(loc='best')
            ax[idx, 0].set_title('Scaled internal concentration')
            ax[idx, 1].set_title('Survival')


        dT = times[-1] - times[0]
        dy = data.max().max()
        ax[idx, 1].set_ylim(-0.05*dy, 1.05*dy)
        ax[idx, 1].set_xlim(-0.05*dT, times[-1] + 0.05*dT)

        ax[idx, 0].set_ylabel('Scaled internal concentration')
        ax[idx, 1].set_xlabel('Survival')

    ax[-1, 1].set_xlabel('Time [h]')
    ax[-1, 0].set_xlabel('Time [h]')
    plt.tight_layout()
    if not subplots:
        fig.suptitle(', '.join(['{0}={1:.2E}'.format(k, params[k].value) for k in params.keys()]))
        fig.subplots_adjust(top=0.85)
    return fig, ax


def generate_survival_data(ke=0.5, b=0.2, c0s=4.0, hb=0.01, exposure_values=None):
    '''Generate survival data set using GUTS-SIC-SD forward model'''
    if not exposure_values:
        exposure_values = [0.0, 10.0, 25.0, 50.0]
    
    #Generate exposures dataframe, single compound 
    exposure_names = [str(el) for el in exposure_values]
    exposures = pd.DataFrame(index=['CompoundX'], columns=exposure_names, dtype=np.double)
    exposures.loc['CompoundX', :].values[:] = exposure_values[:]
    
    #Partition coefficient
    #bcf = pd.Series(index=['CompoundX'], data=[1.0])
    
    #Create Dataframe for survival data, set initial number of surviving individuals
    times = np.linspace(0.0, 10.0, 10)
    data = pd.DataFrame(index=times, columns=exposure_names)
    data.iloc[0, :] = [50.0] * len(exposures)

    #Generate survival data using forward GUTS-SD-SIC model
    params = lmfit.Parameters()
    params.add_many(('ke', ke), ('b', b), ('c0s', c0s), ('hb', hb))
    solutions = get_model_prediction(params, data, exposures)
    
    #Add some gaussian noise to survival data.
    surv = np.array(solutions)[:, :, 1].T
    data.loc[:, :] = np.floor(np.random.normal(loc=surv, scale=np.maximum(.01*surv, 1e-12)))
    data[data>50] = 50.0
    
    return data, exposures
