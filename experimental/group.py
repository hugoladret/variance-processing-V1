#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import group_params as prm
from analysis import group_exclusion, group_analysis, group_plotting
from utils import pipeline_utils


print('#################################')
print('#  Starting group analysis      #')
print('#################################\n')       

# Moving clusters
if prm.do_export : 
    pipeline_utils.regroup_clusters(prm.group_name, prm.folders_list)

# Doing exclusion
if prm.do_exclusion :
    group_exclusion.exclude()


print('\n###############')
print('#  Analyzing  #')
print('###############\n')   
      
# Spearman correlation of NKR
if prm.do_analysis : 
    group_analysis.spearman_nkr(var = 'Phi')
    group_analysis.spearman_nkr(var = 'CirVar')
    
    # Re-running waveform clustering
    group_analysis.waveform_analysis()
    
    # Fisher information 
    group_analysis.pop_fisher_info()
    
    # Decoding
    group_analysis.jazayeri_orientation_decoding()
    #group_analysis.jazayeri_orientation_fitting()

print('\n##############')
print('#  Plotting  #')
print('##############\n')   
      
# Doing recap
if prm.do_plotting :
    group_plotting.recap()
    
    # Plotting the groups and their values
    group_plotting.plot_nkr_correlation(var = 'Phi')
    group_plotting.plot_nkr_correlation(var = 'CirVar')
    
    # Plotting the waveform clustering
    group_plotting.plot_waveform_KMeans()
    
    # Plotting the PSTH fits
    #group_plotting.plot_psth_fits()
    
    # Plotting the fisher information of the population
    group_plotting.plot_fisher_info()
    
    # Plotting the log-likelihoods 
    group_plotting.plot_ori_decoding()
    group_plotting.plot_ori_decoding_errors()


