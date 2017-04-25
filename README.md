ePyTox
======

A Python toolkit for environmental toxicology model fitting and simulations. At
present, mainly a demonstration of using Python for numerical ecotox modelling.

* Supports the GUTS-IT and GUTS-SD models [1]
* Find optimal model parameters by maximizing the likelihood
* Profile the likelihood function
* Sample from the posterior using Markov Chain Monte Carlo

The heavy lifting is done by the scipy, lmfit and emcee packages. Additionally,
matplotlib, pandas, corner, tqdm and seaborn is used.


Installing
----------

Clone the repository, and then run setup.py (python setup.py install).


Quick start
-----------

See example notebook in the notebook/ folder.


References
----------
    [1] Jager, T., Albert, C., Preuss, T. G., & Ashauer, R. (2011). General
        unified threshold model of survival - A toxicokinetic-toxicodynamic
        framework for ecotoxicology. Environmental Science and Technology, 45(7),
        2529–2540. 
