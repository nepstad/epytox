# -*- coding: utf-8 -*-
'''
Uptake (TK) and survival (TD) models, forward simulation and model fitting.

References
----------
    [1] Jager, T. et al. (2011). General unified threshold model of survival -
        a toxicokinetic-toxicodynamic framework for ecotoxicology.  
        Environmental Science & Technology, 45(7), 2529-2540.
'''
import numpy as np
import scipy.integrate as sid
import lmfit
import matplotlib.pyplot as plt
import seaborn as sns
import corner
from tqdm import tqdm
from epytox.likelihood import log_likelihood_multinom, log_likelihood_normal
from epytox.support import mortality_loglogistic


class Model:
    def __init__(self, num_comps=1):
        #Number of internal concentration components
        self.num_comps = num_comps

        #Model parameters, these are set on deriving/implemeting classes
        self.params = lmfit.Parameters()
        self.initial_params = self.params.copy()

        # Parameters for ODE solver and data fit
        self._ode_nsteps = 3000
        self._fit_maxiter = 3000

    def _rhs(self, t, y, params, cd, *args, **kwargs):
        '''Evaluate right hand side of model ODE'''
        raise NotImplementedError('Must be implement in derived class')

    def _solve(self, params, y0, times, cd):
        '''Solver for model ODE.
        
        Returns solution to model ODE at times <times>, given model parameters
        <params> and the time-dependent exposure profile function <cd>, for
        given initial conditions <y0>.
        '''

        # Trying explicit bdf for stiff equations, since lsoda complains
        #r = sid.ode(cls.currhs).set_integrator('vode', nsteps=1000, method='bdf')
        r = sid.ode(self._rhs).set_integrator('lsoda', nsteps=self._ode_nsteps)
        r.set_initial_value(y0)
        r.set_f_params(params.valuesdict(), cd)
        sols = [y0]
        for t in times[1:]:
            r.integrate(t)
            sols.append(r.y)

        return np.array(sols), r

    def negloglike(self, params, datasets):
        '''Negative log-likelihood objective function (normal).'''
        raise NotImplementedError('Please implement in derived class')

    def fit(self, datasets, printres=True, progressbar=True):
        '''Fit model to data via negative log likelihood function'''
        if progressbar:
            pbar = tqdm(total=self._fit_maxiter)
            def objective_wrapper(*args, **kwargs):
                pbar.update(1)
                return self.negloglike(*args, **kwargs)
        else:
            objective_wrapper = self.negloglike

        #Settings for Nelder-Mead/simplex minimization
        nelder_opts = dict(maxfev=self._fit_maxiter, maxiter=self._fit_maxiter, 
                           xatol=1e-8, fatol=1e-8)

        # Minimize the objective function using the simplex method
        mini = lmfit.Minimizer(objective_wrapper, self.params, fcn_args=(datasets,))
        result = mini.minimize(method='nelder', params=self.params, tol=1e-7, 
                               options=nelder_opts)
        self.mle_result = result

        if progressbar:
            pbar.close()

        # Print result of fit
        if printres:
            print(result.message)
            lmfit.report_fit(result)

        return result

    def mcmc(self, datasets, nsteps=10000, thin=1, ntemps=1, nwalkers=None, progressbar=True):
        def objective(*args, **kwargs):
            return -self.negloglike(*args, **kwargs)

        if not nwalkers:
            nwalkers = 2*len([n for n, p in self.params.items() if p.vary])
        infostr = 'Commencing MCMC with {0} walkers, {1} steps, {2} temperatures.'
        print(infostr.format(nwalkers, nsteps, ntemps))

        maxiter = nsteps * nwalkers
        if progressbar:
            pbar = tqdm(total=maxiter)
            def objective_wrapper(*args, **kwargs):
                pbar.update(1)
                return objective(*args, **kwargs)
        else:
            objective_wrapper = objective

        mini = lmfit.Minimizer(objective_wrapper, self.params, 
                               fcn_args=(datasets,))
        res = mini.emcee(burn=0, steps=nsteps, thin=thin, nwalkers=nwalkers, 
                         ntemps=ntemps, params=self.params)

        if progressbar:
            pbar.close()

        return res


    def profile_likelihood(self, datasets):
        #parvals, ll = likelihood.profile_likelihood(self.params, fitfunc, objective, profpar, *likargs):
        pass


class SurvivalModel(Model):
    def negloglike(self, params, datasets):
        '''Negative multinomial log-likelihood objective function for survival.
        
        Loop over treatments and replicates in the data, and calculate negative
        log likelihoods.
        '''
        #Using single scaled internal concentration component
        y0 = np.zeros(2, dtype=np.double)

        #Loop over treatments and survival data
        negloglik = 0
        for cd, data in datasets:
            times = data.index.values
            #@todo improve: Initial cond.: zero internal conc., 100% survival
            y0[:] = [0.0, 1.0]

            #Evaluate model, keep only survival value
            y, _ = self._solve(params, y0, times, cd)
            
            #Calculate negative log likelihood for current treatment
            negloglik -= log_likelihood_multinom(y[:, 1], data.values)

        return negloglik


class InternalConcentrationModel(Model):
    def negloglike(self, params, datasets):
        '''Negative normal log-likelihood objective function.
        
        Loop over treatments and replicates in the data, and calculate negative
        log likelihoods.
        '''
        n = self.num_comps
        y0 = np.zeros(n+1, dtype=np.double)
        times = data.index.values
        negloglik = 0
        sigma = 1.0 #@todo this should be a nuisance parameter
        for treatment, data in datasets:
            #Initial condition: zero internal concentration
            y0[:] = [0.0]

            #Evaluate model
            y, _ = self._solve(params, y0, times, cd)
            
            #Calculate negative log likelihood for current treatment
            negloglik -= log_likelihood_normal(y, data[curtreat].values, sigma)

        return negloglik


#------------------------------------------------------------------------------
# Here follows specific implementations of different GUTS flavors.
#------------------------------------------------------------------------------
class SIC_SD(SurvivalModel):
    '''GUTS-SIC-SD: Scaled internal concentration + stochastic death survival'''

    def __init__(self):
        super().__init__()
        self.params.add('ke', min=1.0e-10, max=100.0, value=1.0, vary=True)
        self.params.add('b', min=1e-10, max=100, value=0.4, vary=True)
        self.params.add('c0s', min=1e-10, max=100, value=2.0, vary=True)
        self.params.add('hb', min=0, max=10, value=0.01, vary=True)

    def _rhs(self, t, y, params, cd):
        v = params
        dy = y.copy()

        # One-compartment kinetics model for body residues
        dy[0] = v['ke']*(cd(t) - y[0])

        #Dose metric
        cstot = y[0]

        #Hazard rate
        hc = v['b'] * max(0, cstot - v['c0s'])
        h = hc + v['hb']
        dy[1] = -h * y[1]

        return dy


class SIC_IT(SurvivalModel):
    '''GUTS-SIC-IT: Scaled internal concentration + individual tolerance survival'''

    def __init__(self):
        super().__init__()
        self.params.add('ke', min=1.0e-10, max=100.0, value=1.0, vary=True)
        self.params.add('alpha', min=1e-10, max=100, value=1.0, vary=True)
        self.params.add('beta', min=1e-10, max=100, value=1.0, vary=True)
        self.params.add('hb', min=0, max=10, value=0.01, vary=True)

    def _rhs(self, t, y, params, cd):
        v = params
        dy = y.copy()

        # One-compartment kinetics model for body residues
        dy[0] = v['ke']*(cd(t) - y[0])

        #Dose metric
        cstot = y[0]

        y[1] = 0.0

        return dy

    def _solve(self, params, y0, times, cd):
        '''
        Solver for model ODE.
        
        Returns solution to model ODE at times <times>, given model parameters
        <params> and the time-dependent exposure profile function <cd>, for
        given initial conditions <y0>.
        '''
        v = params.valuesdict()
 
        #Solve uptake equation
        sols, r = super()._solve(params, y0,  times, cd)
        y = sols

        #Calculate individual tolerance survival
        #@todo Need to do this at each integration time step to resolve
        #internal concentration dynamics and peaks in cstot?
        for i, ystep in enumerate(y):
            if i == 0:
                continue

            t = times[i]

            #Total internal concentration
            cstot = ystep[0]

            #Calculate survival from ratio of internal concentration to 
            #tolerance threshold.
            F = mortality_loglogistic(cstot, v['alpha'], v['beta'])
            surv = y[0, -1] * (1.0 - F) * np.exp(-v['hb']*t)

            #Survival cannot increase with time
            y[i, -1] = min(y[i-1, -1], surv)

        return y, r


class IC_SID_SD:
    pass


class SIC(InternalConcentrationModel):
    '''One-compartment scaled internal concentration'''
    def __init__(self):
        super().__init__()
        self.params.add('ke', min=1.0e-10, max=100.0, value=1.0, vary=True)

    def _rhs(self, t, y, params, cd):
        v = params
        ke = v['ke']
        dy = ke*(cd(t) - y)
        return dy


class IC_QSAR(InternalConcentrationModel):
    '''Multi-component single-compartment internal concentration model'''

    def __init__(self, num_comps=1, qsar='kow', kows=None):
        '''
        qsar: 'kow' or custom function for Piw
            kow-qsar: Piw = a/log_10(kow)
        '''
        super().__init__(num_comps=num_comps)
        self.params.add('ke', min=1.0e-10, max=100.0, value=1.0, vary=True)
        self.params.add('ki', min=1.0e-10, max=100.0, value=1.0, vary=True)

        if qsar == 'kow':
            self.kows = kows
            self.Piw = lambda a: a/self.kows
            self.params.add('qsar_a', min=1.0e-10, max=100.0, value=1.0, vary=True)


    def _rhs(self, t, y, params, cd):
        v = params
        ki = v['ki']
        ke = v['ke']
        qsar_a = v['qsar_a']
        Piw = self.Piw(qsar_a)
        dy = ki*Piw*cd(t) - ke*y
        return dy


class IC(InternalConcentrationModel): 
    '''One-compartment internal concentration'''
    def __init__(self):
        super().__init__()
        self.params.add('ke', min=1.0e-10, max=100.0, value=1.0, vary=True)
        self.params.add('ki', min=1.0e-10, max=100.0, value=1.0, vary=True)

    def _rhs(self, t, y, params, cd):
        v = params
        ki = v['ki']
        ke = v['ke']
        dy = ki*cd(t) - ke*y
        return dy


class IC_N_Kow(Model): 
    '''One-compartment internal concentration for N compounds'''
    def __init__(self):
        super().__init__()
        self.params.add('ke', min=1.0e-10, max=100.0, value=1.0, vary=True)
        self.params.add('Ps', min=1.0e-10, max=100, value=1.0, vary=True)

    def _rhs(self, t, y, params, cd, kows):
        v = params
        ke = v['ke']
        Ps = v['Ps']
        Piw = Ps/kow
        dy = ke*(Psw*cd(t) - y)
        return dy


#------------------------------------------------------------------------------
# Visualization support functions
#------------------------------------------------------------------------------

def plot_fit(model, data, exposure, subplots=True):
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

        # Initial conditions: zero internal concentration, 100% survival
        y0 = [0, data[treatment].iloc[0]]

        # Evaluate model at data time points with present parameter set
        y, r = model._solve(model.params, y0, times, cd)
        ax[idx, 1].plot(times, y[:, 1], color=cur_color)
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
        fig.suptitle(', '.join(['{0}={1:.2E}'.format(k, model.params[k].value) for k in model.params.keys()]))
        fig.subplots_adjust(top=0.85)
    return fig, ax


def plot_mcmc(mcmc_res, res, burn=0, quantiles=[0.025, 0.5, 0.975]):
    # Make corner plot of 1D and 2D projection of the samled values
    c = corner.corner(mcmc_res.flatchain.iloc[burn:, :], labels=mcmc_res.var_names,
              truths=list(res.params.valuesdict().values()), verbose=True,
              show_titles=True, quantiles=quantiles)

    # Plot trajectory of each walker in parameter space
    n = mcmc_res.chain.shape[2]
    fig, ax = plt.subplots(n, 1, figsize=(8, 4*n))
    for i in range(n):
        _ = ax[i].plot(mcmc_res.chain[:, burn:, i].T, 'k-', alpha=0.1)
        ax[i].set_ylabel(mcmc_res.var_names[i])
    ax[-1].set_xlabel('Step number')

    return fig, ax, c
