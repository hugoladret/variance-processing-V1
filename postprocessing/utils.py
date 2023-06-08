#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import numpy as np 
from lmfit import Model, Parameters


def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))   

def norm_minus(a) :
    return 2 * ((a - np.min(a)) / (np.max(a) - np.min(a))) -1

# this helps with the confusion matrices that suffers from kfold structures (!= cv)
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# --------------------------------------------------------------
# The functions below are re-used for the population TC
# --------------------------------------------------------------
def tuning_function(x, mu, kappa, fmax, bsl):
    # Von Mises, with kappa the concentration, mu the location, I0 Bessel order 0
    # fmax the firing rate at pref ori, bsl the min firing rate (not the baseline, which was substracted) 
    tf = np.exp((kappa)*np.cos((x-mu)))#/(2*np.pi*I0(kappa))
    tf = norm_data(tf)
    tf *= fmax
    tf += bsl
    return tf

def fit_tc(array, init_kappa):
    x = np.linspace(-np.pi, np.pi, len(array))
    y = array
    
    mod = Model(tuning_function)
    pars = Parameters()
    pars.add_many(('mu', 0, False, 0., np.pi),
                  ('kappa', init_kappa, True,  .1, 60.),
                  ('fmax', np.max(array), False, 0.01, np.max(array)),
                 ('bsl', np.min(y), False, -0.0001, np.max(array)))

    out = mod.fit(y, pars, x=x, nan_policy='omit')

    return out.best_values

def cirvar(arr) :
    cv_thetas = np.linspace(-np.pi, np.pi, len(arr))
    R = np.sum(arr* np.exp(2j*cv_thetas) / np.sum(arr))
    cv = 1 - np.abs(np.real(R))
    return cv