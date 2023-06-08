#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import utils_single_neuron as sn_utils 
import utils 

import os 
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --------------------------------------------------------------
# Early/late spike ratios of neurons
# --------------------------------------------------------------
def make_early_late_ratio() :
    # FLAG KMEANS
    print('Computing early/late spike ratios...')
    kmeans_spike_ratios = [] 
    for cluster_path in prm.cluster_list:
        folder_path = '_'.join(cluster_path.split('_')[:2])

        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        nm_PSTH_list = np.load(load_folder + '/PSTH.npy', allow_pickle=True)
        nm_PSTH_list = np.flip(nm_PSTH_list, axis=0)
        nmmeans = np.load(load_folder + '/TC.npy').mean(axis=-1)

        prefered_id = np.argmax(nmmeans[-1, :])
        orth_id = np.argmin(nmmeans[-1, :])

        ratios = np.zeros((len(prm.B_thetas), nm_PSTH_list.shape[-1]))
        for ibt, bt in enumerate(prm.B_thetas):
            for trial in range(nm_PSTH_list.shape[-1]):
                spiketrain = nm_PSTH_list[ibt, prefered_id, trial]
                early = len(np.where(spiketrain < 0.1)[0])
                late = len(np.where(spiketrain > 0.2)[0])
                try:
                    ratios[ibt, trial] = late/early
                except:
                    pass

        kmeans_spike_ratios.append({'cluster' : cluster_path,
                            'early_late_ratios': np.mean(ratios, axis=-1)})
    np.save('./data/%s/kmeans_early_late_ratios.npy' % prm.postprocess_name, kmeans_spike_ratios)
    
    
    all_ratios = np.load('./data/%s/kmeans_early_late_ratios.npy' % prm.postprocess_name, allow_pickle = True)
    all_ratios = [x['early_late_ratios'] for x in all_ratios]    
    
    for i, bt in enumerate([-1, 0]) :
        fig, ax = plt.subplots(figsize = (5,3))
        ax.hist(np.log([x[bt] for x in all_ratios]), bins = np.linspace(-4, 4, 20),
            facecolor=prm.colors[0] if i ==0 else prm.colors[-1], edgecolor='w')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xticks([-4,0,4])

        ax.set_yticks([0, 30, 60])
        ax.set_ylim(0, 60)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.set_ylabel('# neuron', fontsize=14)
        ax.set_xlabel(r'$log(\frac{early}{late})$ spike count ', fontsize=14)

        fig.tight_layout()
        fig.savefig('./figs/sn_dynamics_early_late_bt%s_histo.pdf' % bt, bbox_inches='tight', dpi=200, transparent=True)
        print('At Btheta = %s, Median value of early/late spike ratio: %.2f' % (bt,np.median(np.log([x[bt] for x in all_ratios]))))
        plt.show(block = prm.block_plot)
    print('Difference of delays (ratios is less for larger bthetas) :')
    print(mannwhitneyu(np.log([x[-1] for x in all_ratios]), np.log([x[0] for x in all_ratios]), alternative = 'less'))



# --------------------------------------------------------------
# Early/late spike for the clustering figure
# --------------------------------------------------------------
def make_early_late_ratio_plot() :
    fig, ax = plt.subplots(figsize=(6, 6))
    all_ratios = np.load('./data/%s/kmeans_early_late_ratios.npy' % prm.postprocess_name, allow_pickle = True)
    tuned_ratios = [x for x in all_ratios if x['cluster'] in prm.tuned_lst]
    untuned_ratios = [x for x in all_ratios if x['cluster'] in prm.untuned_lst]
    
    btplot = [7, 0]
    pos = [0, .9]
    for ibt, bt in enumerate(btplot):
        c = prm.col_tuned
        ratios_tuned = np.log([x['early_late_ratios'][bt] for x in tuned_ratios])
        vp = ax.boxplot(ratios_tuned,
                        positions=[pos[ibt]+.3],
                        widths=.2, showmeans=False,
                        showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor=c, color=c),
                        capprops=dict(color=c),
                        whiskerprops=dict(color=c),
                        flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color='white'))

    for ibt, bt in enumerate(btplot):
        c = prm.col_untuned
        ratios_untuned = np.log([x['early_late_ratios'][bt] for x in untuned_ratios])
        vp = ax.boxplot(ratios_untuned,
                        positions=[pos[ibt]],
                        widths=.2, showmeans=False,
                        showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor=c, color=c),
                        capprops=dict(color=c),
                        whiskerprops=dict(color=c),
                        flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color='white'))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'$B_\theta$ (°)', fontsize=18)
    ax.set_ylabel(r'$log(\frac{early}{late})$ spike count ', fontsize=18)

    ax.set_xticks([0.15, 1.05])
    ax.set_xticklabels(['0', '36'])
    ax.axhline(0, c='gray', linestyle='--', alpha=.5)
    ax.set_yticks([-4, 0, 4])
    ax.set_ylim([-4,  4])

    fig.savefig('./figs/clustering_early_late_ratios.pdf',
                bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    print('-- Stats on the early/late ratios')
    print('Tuned vs untuned at btheta 0°')
    print(mannwhitneyu([x['early_late_ratios'][-1] for x in tuned_ratios],
                    [x['early_late_ratios'][-1] for x in untuned_ratios],
                    alternative='greater'))

    print('Tuned vs untuned at btheta 36°')
    print(mannwhitneyu([x['early_late_ratios'][0] for x in tuned_ratios],
                    [x['early_late_ratios'][0] for x in untuned_ratios],
                    alternative='greater'))

    print('Tuned at btheta 0 vs tuned at btheta 36°')
    print(mannwhitneyu([x['early_late_ratios'][-1] for x in tuned_ratios],
                    [x['early_late_ratios'][0] for x in tuned_ratios],
                    alternative='less'))

    print('untuned at btheta 0 vs untuned at btheta 36°')
    print(mannwhitneyu([x['early_late_ratios'][-1] for x in untuned_ratios],
                    [x['early_late_ratios'][0] for x in untuned_ratios],
                    alternative='less'))
    
    

# --------------------------------------------------------------
# Plot the dynamical tuning curve of a neuron
# --------------------------------------------------------------
def make_dynamical_tc():
    print('Computing dynamical tuning curves')
    # For each cluster
    for iclust, cluster_path in enumerate(prm.ex_neurons):
        print(cluster_path)
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        spiketimes = np.load(load_folder + '/spiketimes.npy')
        spiketimes = spiketimes / prm.fs

        sequences_contents = np.load(load_folder + '/sequences_contents.npy', allow_pickle = True)
        # For each b_theta (in decreasing order)
        full_array = np.zeros((len(prm.B_thetas), len(prm.latencies), len(prm.thetas)))
        for i_bt, bt in enumerate(prm.B_thetas) :
            # For each latency
            for i_lat, lat in enumerate(prm.latencies) :
                # We recompute a tuning curve (sequences where bt and t match)
                tc_array = np.zeros(len(prm.thetas)) # array of size 12 (number of thetas)
                for seq in sequences_contents : 
                    if seq['sequence_btheta'] == bt :
                        seq_beg = (seq['sequence_beg'] / prm.fs) + lat
                        seq_end = seq_beg + prm.win_size
                        spikes_in = np.where((spiketimes >= seq_beg) & (spiketimes <= seq_end))[0]
                        seq_theta_idx = np.where(np.round(seq['sequence_theta'], 5)== np.round(prm.thetas, 5))[0][0]
                        tc_array[seq_theta_idx] += len(spikes_in)

                # And we renormalize by the number of time stimulations were shown
                tc_array = tc_array/30
                full_array[i_bt, i_lat, :] = tc_array
                
        tc, params = sn_utils.torch_fit_tc(full_array)

        
        plt_min, plt_max = np.min(full_array[-1,:,:]), np.max(full_array[-1,:,:])*1.1

        plot_idxs = np.linspace(3, len(prm.latencies)-1, len(prm.cm_timesteps), dtype = np.int16, endpoint = True)
        plots_idxs = prm.cm_timesteps
        fig = sn_utils.plot_temporal_array(tc, full_array, plot_idxs, plt_min, plt_max)
        fig.savefig('./figs/sn_dynamics_TC_%s.pdf' % cluster_path, bbox_inches='tight', dpi=200, transparent=True)
        plt.show(block = prm.block_plot)


# --------------------------------------------------------------
# Make the correlation array for clustering and the rest of the figures
# Althought not dynamical, the NKRs are also recomputed here
# --------------------------------------------------------------
def make_correlation_array():
    correlation_array = Parallel(n_jobs = -1)(delayed(sn_utils.return_cluster_delay)(cluster)
                                for cluster in tqdm(prm.cluster_list, desc = 'Computing dynamical array...'))
    correlation_array = [x for x in correlation_array if x is not None]
    np.save('./data/%s/correlation_array.npy'%prm.postprocess_name, correlation_array)
    
    # FLAG KMEANS  
    kmeans_optimal_delay = []
    kmeans_nkr_params = []
    kmeans_raw_cv = []
    for iclust, clust in enumerate(correlation_array) :
        kmeans_optimal_delay.append((clust['cluster'], clust['all_delays']))
        kmeans_nkr_params.append((clust['cluster'], clust['cv'][1]))
        kmeans_raw_cv.append((clust['cluster'], clust['cv'][0]))
    np.save('./data/%s/kmeans_optimal_delay.npy' % prm.postprocess_name,kmeans_optimal_delay)
    np.save('./data/%s/kmeans_nkr_params.npy' % prm.postprocess_name,kmeans_nkr_params)
    np.save('./data/%s/kmeans_raw_cv.npy' % prm.postprocess_name,kmeans_raw_cv)
        
        
        
# --------------------------------------------------------------
# Plot the histogram of optimal delays
# --------------------------------------------------------------  
def make_optimal_delays_plot() :
    correlation_array = np.load('./data/%s/correlation_array.npy'%prm.postprocess_name, allow_pickle = True)
    arr_delays = np.asarray([x['all_delays'] for x in correlation_array])
    
    for ibt, bt in enumerate([0,-1]) :
        fig, ax = plt.subplots(figsize = (5,3))
        
        ax.hist(arr_delays[:,bt], bins = np.linspace(0, 60, 12),
            facecolor=prm.colors[bt], edgecolor='w') #btheta 36

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xticks([0,30,60])
        ax.set_xticklabels([50, 225, 400])

        ax.set_yticks([0, 25, 50])
        ax.set_ylim(0, 50)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.set_xlabel('Optimal delay (ms)', fontsize=14)
        ax.set_ylabel('# neuron', fontsize=14)

        fig.tight_layout()
        fig.savefig('./figs/sn_dynamical_optimaldelay_bt%s.pdf' % bt, bbox_inches='tight', dpi=200, transparent=True)
        plt.show(block = prm.block_plot)
        
        print('Histogram of optimal delays for btheta %s - median : %.5f' % (bt, np.median(arr_delays[:,bt])))
    print('Histogram of optimal delays for btheta 36 vs btheta 0 is greater ?')
    print(mannwhitneyu(arr_delays[:,0], arr_delays[:,-1], alternative = 'greater'))
    
    
# --------------------------------------------------------------
# Plot the ratio of optimal delays
# --------------------------------------------------------------  
def make_optimal_delay_ratio_plot():
    correlation_array = np.load('./data/%s/correlation_array.npy'%prm.postprocess_name, allow_pickle = True)
    tuned_delays, untuned_delays = [], []
    for i, clust in enumerate(correlation_array) :
        if clust['cluster'] in prm.tuned_lst :
            tuned_delays.append(clust['all_delays'])
        else :
            untuned_delays.append(clust['all_delays'])
            
    delay_tuned_0 = [x[-1] for x in tuned_delays]
    delay_tuned_36 = [x[0] for x in tuned_delays]
    delay_untuned_0 = [x[-1] for x in untuned_delays]
    delay_untuned_36 = [x[0] for x in untuned_delays]
    
    latencies = np.linspace(0.05, .4, len(correlation_array[0]['cv_array'][0,:]))
    delay_tuned_0 = latencies[np.asarray(delay_tuned_0, int)]*1000
    delay_tuned_36 = latencies[np.asarray(delay_tuned_36, int)]*1000
    delay_untuned_0 = latencies[np.asarray(delay_untuned_0, int)]*1000
    delay_untuned_36 = latencies[np.asarray(delay_untuned_36, int)]*1000
    
    fig, ax = plt.subplots(figsize = (6,6))

    # Delays for bt_0
    c = prm.col_tuned
    ax.boxplot(delay_tuned_0,
                        positions = [0.3],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot(delay_untuned_0,
                        positions = [0],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 
        
        
    # Delay for bt_36
    c = prm.col_tuned
    ax.boxplot(delay_tuned_36,
                        positions = [1.2],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot(delay_untuned_36,
                        positions = [0.9],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.set_yticks(np.linspace(0,6,6))

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'$B_\theta$ (°)', fontsize = 18)
    ax.set_ylabel(r'Delay to max TC FR (ms)', fontsize = 18)

    ax.set_xticks([0.15, 1.05])
    ax.set_xticklabels(['0', '36'])
    ax.set_yticks([0, 250, 500])
    ax.set_ylim(0,500)
    fig.savefig('./figs/clustering_optimal_delay_ratios.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)