#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:22:19 2020

@author: hugo
Exclusion functions for the group analysis
"""

import numpy as np
import os
import shutil 
import group_params as prm
from tqdm import tqdm


def exclude() :
    '''
    Excludes clusters based on criterion defined in group_params
    '''
    
    print('# Excluding clusters... #')
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    total = len(clusters_list) # total clusters before cleaning
    excluded = np.zeros(7) # we have 5 criterions of exclusion
    
    for cluster in tqdm(clusters_list) :
        if not os.path.isdir(clusters_path + cluster) :
            continue 
        spiketimes = np.load(clusters_path + cluster + '/spiketimes.npy')
        tc_mean = np.load(clusters_path + cluster + '/0.000_plot_MC_TC_nonmerged_means.npy')
        r_squared = np.load(clusters_path + cluster + '/0.000_plot_neurometric_merged_fit_reports.npy')
        nkr_params = np.load(clusters_path + cluster + '/0.000_phi_fit.npy', allow_pickle = True)
        nkr_params = nkr_params.item()
        phi_c50 = nkr_params['c50']
        nkr_params = np.load(clusters_path + cluster + '/0.000_cirvar_fit.npy', allow_pickle = True)
        nkr_params = nkr_params.item()
        cv_c50 = nkr_params['c50']
        cv_rmax = nkr_params['rmax']
        waveform_mean = np.load(clusters_path + cluster + '/waveform_mean.npy')
        waveform_std = np.load(clusters_path + cluster + '/waveform_std.npy')
        
        # Waveform exclusion, -------------------------------------------------
        # this removes all kind of non-neuron artifacts and MUA left 
        if np.abs(np.mean(waveform_mean - waveform_std)) > prm.waveform_value :
            excluded[0] +=1 
            shutil.rmtree(clusters_path + cluster)
            continue
        
        # Firing rate dropping exclusion --------------------------------------
        # Receipe at https://stackoverflow.com/questions/57712650/
        hist = np.histogram(spiketimes, bins = 1800)[0]
        below_mask = np.r_[False, hist < prm.drop_value, False]
        idx = np.flatnonzero(below_mask[:-1]!=below_mask[1:])
        lens = idx[1::2]-idx[::2]
        drops = lens >= prm.drop_duration
        if np.any(drops) :
            excluded[1] +=1 
            shutil.rmtree(clusters_path + cluster)
            continue
        
        
        # Evoked firing rate exclusion --------------------------------------
        # bt0_tc = tc_mean[-1] # Tuning curve at Btheta = 0
        # if not np.any(bt0_tc > prm.min_max_FR) :
        #     excluded[2] +=1 
        #     shutil.rmtree(clusters_path + cluster)
        #     continue
        
        
        # # TC modulation exclusion --------------------------------------
        # if not np.max(bt0_tc) >= (prm.tc_power/100) * np.min(bt0_tc) :
        #     excluded[3] +=1 
        #     shutil.rmtree(clusters_path + cluster)
        #     continue
        
        
        # Goodness of fit exclusion, based on the merged TC --------------------------------------
        if r_squared[0] < prm.min_R2 :
            excluded[2] +=1 
            shutil.rmtree(clusters_path + cluster)
            continue
        
        
        # # c50 exclusion --------------------------------------
        # if phi_c50 >= prm.max_c50 or cv_c50 >= prm.max_c50:
        #     excluded[5] +=1 
        #     shutil.rmtree(clusters_path + cluster)
        #     continue
        # # Rmax exclusion --------------------------------------
        # if cv_rmax >= 2 :
        #     excluded[6] +=1 
        #     shutil.rmtree(clusters_path + cluster)
        #    continue
    print('Out of %s total neurons, excluded %s via waveform, %s via firing rate, %s via goodness of fit.\nTotal excluded = %s/%s, remaining : %s' 
          % (total, int(excluded[0]), int(excluded[1]), int(excluded[2]), int(np.sum(excluded)), total, (total-int(np.sum(excluded)))) )
    #print('Exclusion done ! Excluded %s/%s clusters, %s neurons remaining for analysis.\n' % (excluded, total, total-excluded))