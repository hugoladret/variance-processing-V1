#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import utils_single_neuron as sn_utils 

import numpy as np 
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt 

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings 
warnings.catch_warnings()
warnings.simplefilter("ignore")

# --------------------------------------------------------------
# Tuning curves stats
# --------------------------------------------------------------
def make_stats() :
    print('Computing whether the peak-to-baseline ratio changes significantly from Bt0 to Bt36')
    changed = 0
    ratios = []
    for cluster_path in prm.cluster_list :
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        mean_FR = np.load(load_folder + '/TC.npy')

        w, p = wilcoxon(np.max(mean_FR[-1,:,:], axis = 0)/np.min(mean_FR[-1,:,:], axis = 0),
                        np.max(mean_FR[0,:,:], axis = 0)/np.max(mean_FR[0,:,:], axis = 0),
                        alternative = 'less', zero_method = 'pratt', correction = True)
        ratios.append(np.mean(np.max(mean_FR[0,:,:], axis = 0) / np.min(mean_FR[0,:,:], axis = 0)) / np.mean(np.max(mean_FR[-1,:,:], axis = 0) / np.min(mean_FR[-1,:,:], axis = 0)))
        if p <.05 : changed +=1
        if cluster_path in prm.main_neurons : print(cluster_path, w, p)
            
    print("Neurons that signif. changed amp from Bt0 to Bt36 :  %s / %s (%.2f percent)" % 
        (changed, len(prm.cluster_list), changed/len(prm.cluster_list)*100))


    print('\nComputing how many neurons are still tuned at Bt36')
    changed = 0
    list_changed = []
    for cluster_path in prm.cluster_list  :
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        mean_FR = np.load(load_folder + '/TC.npy')
        
        pref_ori = np.argmax(np.mean(mean_FR[-1,:,:], axis = -1))
        unpref_ori = pref_ori + 5
        if unpref_ori > 11 :
            unpref_ori = 0
            
        w, p = wilcoxon(mean_FR[0,pref_ori,:] , mean_FR[0,unpref_ori,:],
                    alternative = 'two-sided', zero_method = 'pratt', correction = True)
        if p <.05 : 
            changed +=1
            list_changed.append(cluster_path)
        if cluster_path in prm.main_neurons : print(cluster_path, w, p)
    print("Neurons that are still tuned at Bt36 :  %s / %s (%.2f percent)" % 
        (changed, len(prm.cluster_list), changed/len(prm.cluster_list)*100))

    print('\nComputing the last Btheta at which the neuron is tuned')
    changed = 0
    last_changed = []
    for cluster_path in prm.cluster_list :
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        mean_FR = np.load(load_folder + '/TC.npy')
        
        pref_ori = np.argmax(np.mean(mean_FR[-1,:,:], axis = -1))
        unpref_ori = pref_ori + 5
        if unpref_ori > 11 :
            unpref_ori = 0
            
        for ibt, bt in enumerate(prm.B_thetas) :
            w, p = wilcoxon(mean_FR[ibt,pref_ori,:] , mean_FR[ibt,unpref_ori,:],
                        alternative = 'two-sided', zero_method = 'pratt', correction = True)
            if p <.05 : 
                bt_changed = bt
                last_changed.append([cluster_path, bt_changed])
                break
            elif p>0.05 and ibt == len(prm.B_thetas)-1 : # tuned after btheta = 36
                bt_changed = bt
                last_changed.append([cluster_path, bt_changed])
                break
        if cluster_path in prm.main_neurons : print(cluster_path, w, p, bt*180/np.pi)

    # FLAG KMEANS 
    kmeans_last_bt_tuned = []
    for clust in last_changed : 
        kmeans_last_bt_tuned.append({'cluster' : clust[0],
                                    'last_bt' : clust[1]})  
    np.save('./data/%s/kmeans_last_bt_tuned.npy' % prm.postprocess_name, kmeans_last_bt_tuned)

    print('\nComputing the whether pref ori changed') 
    changed = 0
    for cluster_path in prm.cluster_list :
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        mean_FR = np.load(load_folder + '/TC.npy')
        
        pref_ori = np.argmax(np.mean(mean_FR[-1,:,:], axis = -1))
        unpref_ori = pref_ori + 6
        if unpref_ori > 11 :
            unpref_ori = 0
            
        # Get the pvals of the diff between pref and unpref ori at all bthetas
        ps = []
        for ibt, bt in enumerate(prm.B_thetas) :
            w, p = wilcoxon( mean_FR[ibt,pref_ori,:], 
                            mean_FR[ibt,unpref_ori,:])
            ps.append(p)
        # And the "last tuned" is the first <.05 = False in the list (reverse due to Bt array order) 
        try :
            last_tuned_idx = np.where(np.asarray(ps) < 0.05)[0][-1]
        except IndexError : #tuned all the way 
            last_tuned_idx = 0
            
        try :
            w, p = wilcoxon(np.argmax(mean_FR[-1,:,:], axis = 0),
                            np.argmax(mean_FR[last_tuned_idx,:,:], axis = 0),
                        alternative = 'two-sided', zero_method = 'pratt', correction = True)
            if p <.05 : changed +=1
        except ValueError :
            pass # wilcoxon returns a ValueError if x,y are exactly the same, in which case there's been no change
        if cluster_path in prm.main_neurons : print(cluster_path, w, p)
            
    print("Neurons that signif. changed pref ori from Bt0 to their last tuned Bt:  %s / %s (%.2f percent)" % 
        (changed, len(prm.cluster_list), changed/len(prm.cluster_list)*100))



# --------------------------------------------------------------
# Once post-processed, stats on last btheta used for Kmeans clustering
# --------------------------------------------------------------
def make_last_btheta_tuned() :
    last_changed = np.load('./data/%s/kmeans_last_bt_tuned.npy' % prm.postprocess_name, allow_pickle = True)
    tuned_bt_change, untuned_bt_change = [], []
    for clust in last_changed :
        if clust['cluster'] in prm.untuned_lst :
            untuned_bt_change.append(clust['last_bt'])
        elif clust['cluster'] in prm.tuned_lst :
            tuned_bt_change.append(clust['last_bt'])
    tuned_bt_change = np.asarray(tuned_bt_change)*180/np.pi
    untuned_bt_change = np.asarray(untuned_bt_change)*180/np.pi
    
    fig, ax = plt.subplots(figsize = (6,6))

    c = prm.col_untuned
    ax.boxplot(untuned_bt_change,
                        positions = [0],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white'))

    c = prm.col_tuned
    ax.boxplot(tuned_bt_change,
                        positions = [.3],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'Max $B_\theta$ ori. tuned (°)', fontsize = 18)

    ax.set_yticks([-.2, 18, 36.2]) 
    ax.set_yticklabels([0, 18, 36])
    ax.set_ylim(-.2, 36.2)
    ax.set_xticks([])
    fig.savefig('./figs/clustering_max_bt_tuned.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    print('Statistical difference between tuned and untuned neurons for last Bt tuned:')
    print(mannwhitneyu(tuned_bt_change, untuned_bt_change, alternative = 'greater'))
    print('\n')
    
    

# --------------------------------------------------------------
# plotting the tuning curves
# --------------------------------------------------------------
def make_tc():
    print('Plotting tuning curves for example neurons')
    for iclust, cluster_path in enumerate(prm.ex_neurons):
        folder_path = '_'.join(cluster_path.split('_')[:2])

        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        fitted_TC = np.load(load_folder + '/fitted_TC.npy')[::-1]
        mean_FR = np.load(load_folder + '/TC.npy').mean(axis = -1)[::-1]

        tc_duration = .3
        mult_TC = 1/tc_duration

        sn_utils.make_tc_multiaxes(figsize=(20,2.5),  filename=cluster_path,
                        fitted_TC=fitted_TC, mean_FR=mean_FR, B_thetas=prm.B_thetas,
                        mult_TC=mult_TC, 
                        do_legend=False, do_btlabs = False if iclust == 0 else False)
                        
                        
def make_baseline() :
    correlation_array = np.load('./data/%s/correlation_array.npy'%prm.postprocess_name, allow_pickle = True)
    tuned_bsl, untuned_bsl = [], []
    for i, clust in enumerate(correlation_array) :
        if clust['cluster'] in prm.tuned_lst :
            tuned_bsl.append(clust['baseline'])
        else :
            untuned_bsl.append(clust['baseline'])
            
    
    fig, ax = plt.subplots(figsize = (6,6))
    for i, ibt in enumerate([0,7]) :
    
        c = prm.col_untuned
        ax.boxplot(np.asarray([x[ibt] for x in untuned_bsl])*10,
                            positions = [0+i],
                            widths = .2, showmeans = False,
                    showfliers = False, patch_artist=True,
                        boxprops=dict(facecolor=c, color=c),
                        capprops=dict(color=c),
                        whiskerprops=dict(color=c),
                        flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color='white'))

        c = prm.col_tuned
        ax.boxplot(np.asarray([x[ibt] for x in tuned_bsl])*10,
                            positions = [.3+i],
                            widths = .2, showmeans = False,
                    showfliers = False, patch_artist=True,
                        boxprops=dict(facecolor=c, color=c),
                        capprops=dict(color=c),
                        whiskerprops=dict(color=c),
                        flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color='white')) 

        print('Statistical difference for baseline at btheta %s' % ibt )
        print(mannwhitneyu([x[ibt] for x in tuned_bsl], [x[ibt] for x in untuned_bsl], alternative = 'greater'))
        print('\n')   
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'Baseline', fontsize = 18)

    ax.set_yticks([-.2, 10, 20]) 
    ax.set_yticklabels([0, 10, 20])
    ax.set_ylim(-.2, 20)
    ax.set_xticks([])
    
    fig.savefig('./figs/clustering_baseline.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    