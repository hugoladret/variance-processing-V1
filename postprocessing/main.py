#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
Main file, which runs analysis in a serial way.
"""

import matplotlib.pyplot as plt

import params as prm
import utils

import run_introduction

import run_sn_tuning_curves
import run_sn_psth
import run_sn_dynamics
import run_sn_NKR

import run_clustering

import run_decoding_theta
import run_decoding_btheta
import run_decoding_bthetatheta
import run_decoding_continuous

print('#################################')
print('#  Making introduction figures  #')
print('#################################\n')
run_introduction.make_intro_img()
run_introduction.make_mc()
run_introduction.make_cv()
plt.close('all')
print('Done with the introduction figures !')


print('\n##################################')
print('#  Making single neuron figures  #')
print('##################################\n')
run_sn_tuning_curves.make_stats()
run_sn_tuning_curves.make_tc()
plt.close('all')
print('Done with the tuning curves figures !')

run_sn_psth.make_psth()
plt.close('all')
print('Done with the PSTH figures !')

run_sn_dynamics.make_early_late_ratio()
run_sn_dynamics.make_dynamical_tc()
plt.close('all')
print('Done with the dynamical figures !')

run_sn_dynamics.make_correlation_array()
run_sn_dynamics.make_optimal_delays_plot()
print('Done with extracting data for correlation array')
plt.close('all')

for dtype in ['CV', 'HWHH', 'Rmax'] : 
    print('Doing NKR for %s' % dtype)
    run_sn_NKR.make_NKR(dtype = dtype)
    run_sn_NKR.make_NKR_histogram(dtype = dtype)
    run_sn_NKR.make_NKR_BIC(dtype = dtype)
    plt.close('all')
print('Done with the NKR figures !')

print('##############################')
print('#  Making clustering figures #')
print('##############################\n')
run_clustering.make_clustering()
import importlib
importlib.reload(prm) # once clustering has been done, re re-get the vul/res neurons
run_clustering.make_direction_selectivity()
run_clustering.make_depth()
plt.close('all')
print('Done with the clustering figures !')

if prm.clustering_available : # this is just a sanity check, clustering should have made everything available
    run_sn_dynamics.make_early_late_ratio_plot()
    run_sn_dynamics.make_optimal_delay_ratio_plot()
    run_sn_tuning_curves.make_last_btheta_tuned()
    run_sn_tuning_curves.make_baseline()
    for dtype in ['CV', 'HWHH', 'Rmax'] :
        run_sn_NKR.make_CV_plot(dtype = dtype)
        run_sn_NKR.make_params_plot(dtype = dtype)
    plt.close('all')
    print('Done with the post-clustering figures !')

print('\n############################')
print('#  Making decoding figures #')
print('############################\n')
run_decoding_theta.make_theta_decoding_all()
run_decoding_theta.make_theta_population_tc_all()
run_decoding_theta.make_theta_decoding_groups()
run_decoding_theta.make_theta_population_tc_groups()
plt.close('all')
print('Done with the decoding theta figures !')

run_decoding_btheta.make_btheta_decoding_all()
run_decoding_btheta.make_btheta_population_tc_all()
run_decoding_btheta.make_btheta_decoding_groups()
run_decoding_btheta.make_btheta_population_tc_groups()
plt.close('all')
print('Done with the decoding Btheta figures !')

run_decoding_bthetatheta.make_bthetatheta_decoding_all()
run_decoding_bthetatheta.make_bthetatheta_population_tc_all()
run_decoding_bthetatheta.make_bthetatheta_decoding_groups()
run_decoding_bthetatheta.make_bthetatheta_population_tc_groups()
run_decoding_bthetatheta.make_bthetatheta_population_marginalization_groups()
run_decoding_bthetatheta.make_coeffmaps()
plt.close('all')
print('Done with the decoding Btheta x Theta figures !')

print('\n#####################################')
print('#  Making continuous decoding figures #')
print('#######################################\n')
run_decoding_continuous.make_continuous_theta_decoding()
run_decoding_continuous.make_continuous_btheta_decoding()
run_decoding_continuous.make_continuous_btheta_theta_decoding()
plt.close('all')
print('Done with the decoding with continuous scoring !')

print('\n##############################################################')
print('#  All done ! Jump to the ../model folder for the final figures#')
print('################################################################\n')
'''