#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo

31/05/23 note : Clustering is bugging, for some reason I need to figure out
a) Sklearn update since the first iteration of the code ? 
b) Way of computing cirvar (this is now migrated directly into the clustering code) ? 
c) Double check the labelling of the neurons, maybe it's as trivial as the kmeans labels ? 
d) Zero-lagged dynamics units ? 
"""

import params as prm
import utils 
import utils_single_neuron as sn_utils

import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from scipy.stats import mannwhitneyu
import pandas as pd # simply for the pca components dump 

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --------------------------------------------------------------
# Make the clustering
# --------------------------------------------------------------  
def make_clustering():
    '''
    Meta function for the clustering calls
    Most commented out lines will be deleted in the short future
    '''
    print('Running K-means clustering...')

    kmeans_arr_early_late = np.load('./data/%s/kmeans_early_late_ratios.npy' % prm.postprocess_name, allow_pickle = True)
    kmeans_arr_last_bt = np.load('./data/%s/kmeans_last_bt_tuned.npy' % prm.postprocess_name, allow_pickle = True)
    kmeans_arr_optimal_delay = np.load('./data/%s/kmeans_optimal_delay.npy' % prm.postprocess_name, allow_pickle = True)
    kmeans_arr_clustering_array = np.load('./data/%s/correlation_array.npy' % prm.postprocess_name, allow_pickle = True)

    kmeans_data = np.zeros((len(prm.cluster_list)-4, 11))
    lst_clusters_kmeans = []
    lst_clusters_continuous = []
    for icluster, cluster in enumerate(prm.cluster_list) :
        if cluster == 'Steven_AA003_cl0' : # this is a MUA cluster TODO REMOVE
            pass
        else :
            
            try :
                load_folder = prm.grouping_path + '/' + cluster
                #fr_means = np.load(load_folder + '/TC.npy').mean(axis = -1)
                sequences_contents = np.load(load_folder + '/sequences_contents.npy', allow_pickle = True)
                bsl = np.load(load_folder + '/baseline.npy').mean()
                
                # Compute CV for the corrupted data files 
                try :
                    TC_list_btheta = []
                    for ibt, u_btheta in enumerate(prm.B_thetas):
                        TC_list_theta =[] 
                        for u_theta in prm.thetas :
                            
                            spikes_per_thetabtheta = []
                            for seq in sequences_contents :
                                if seq['sequence_theta'] == u_theta and seq['sequence_btheta'] == u_btheta and seq['stimtype'] == 'MC' :
                                    spikes_per_thetabtheta.append(seq['tot_spikes'] - bsl)
                                    
                            TC_list_theta.append(spikes_per_thetabtheta)
                        TC_list_btheta.append(TC_list_theta)
            
                    all_means, all_stds = [], []
                    for i in range(len(TC_list_btheta)):
                        means = np.mean(TC_list_btheta[i], axis = 1)
                        stds = np.std(TC_list_btheta[i], axis = 1)
                        
                        all_means.append(means)
                        all_stds.append(stds)
                        
                    fr_means = np.asarray(all_means)
                
                    CVs = []
                    for i, btheta in enumerate(prm.B_thetas) :
                        arr = fr_means[i, :]
                        R = np.sum( (arr * np.exp(2j*prm.thetas)) / np.sum(arr) )
                        cirvar = 1 - np.abs(np.real(R))
                        CVs.append(cirvar)
                    CVs = np.array(CVs)
                    #print(CVs)
            
                    fit_cv, _ = sn_utils.fit_nkr(CVs)
                except ValueError :
                    pass
                
                kmeans_data[icluster, 0] = icluster 

                kmeans_data[icluster, 1] = [x['last_bt'] for x in kmeans_arr_last_bt if x['cluster'] == cluster][0] 
                kmeans_data[icluster, 2] = fit_cv['c50']
                kmeans_data[icluster, 3] = fit_cv['b'] 
                kmeans_data[icluster, 4] = np.log(fit_cv['n'])
                
                kmeans_data[icluster, 5] = CVs[0]/CVs[-1]
                
                kmeans_data[icluster, 6] = [x[1][0] for x in kmeans_arr_optimal_delay if x[0] == cluster][0] 
                kmeans_data[icluster, 7] = [x[1][-1] for x in kmeans_arr_optimal_delay if x[0] == cluster][0] 
                
                kmeans_data[icluster, 8] = [x['early_late_ratios'][0] for x in kmeans_arr_early_late if x['cluster'] == cluster][0] 
                kmeans_data[icluster, 9] = [x['early_late_ratios'][-1] for x in kmeans_arr_early_late if x['cluster'] == cluster][0] 
                lst_clusters_continuous.append(icluster)
            except IndexError :
                pass
        lst_clusters_kmeans.append(cluster)
        
    np.save('./data/%s/continuous_clusters_list.npy' % prm.postprocess_name, lst_clusters_continuous)
    '''extra_arr = np.load('./data/%s/kmeans_arr_dontchange.npy' % prm.postprocess_name, allow_pickle = True)
    
    for icluster, cluster in enumerate(prm.cluster_list) :
        try :
            print(np.abs(kmeans_data[icluster, 1] - [x['last_bt'] for x in extra_arr if x['cluster'] == cluster]))
            #print(kmeans_data[icluster, 6] )
        except KeyError :
            pass'''
    
    # Compute a continuous resilience/vulnerable score --> the greater the more the resilience 
    # (1) for NKRs, we want min(f0), min(log(n)) and max(btheta50) for resilient neurons 
    # --> norm(btheta50) + (1-norm(log(n))) + (1-norm(f0)) where norm is the normalized (0,1) value
    # (2) for the cv and btheta max, we want min(cvbt0) and max(btheta max) for resilient neurons
    # --> norm(btheta max) + (1-norm(cvbt0)) where norm is the normalized (0,1) value
    # (3) for the temporal aspect, we want max(log(earlylateratio)) and max(delaymaxspike))
    # --> norm(log(earlylateratio)) + norm(delaymaxspike) where norm is the normalized (0,1) value
    norm_nkr_b50 = utils.norm_data(kmeans_data[:,2])
    norm_nkr_b = utils.norm_data(kmeans_data[:,3])
    norm_nkr_n = utils.norm_data(kmeans_data[:,4])
    norm_bthetamax = utils.norm_data(kmeans_data[:,1])
    norm_cv = utils.norm_data(kmeans_data[:,5]) # TODO maybe we must invert this 
    norm_early_ratio = utils.norm_data(kmeans_data[:,9])
    norm_delay = utils.norm_data(kmeans_data[:,7]) # TODO same maybe we must invert to get btheta = 0°
    # resilience score is actually computed after PCA is computed, a few lines later, to get the components
    
    
    # Scale for K-means
    scaler = StandardScaler()
    scaler.fit(kmeans_data)
    kmeans_data = scaler.transform(kmeans_data)

    # Make the plots for K-means parameters
    print('Validating K-means parameters...')
    n_clusters = 8
    fig, ax = plt.subplots(figsize = (6,6))

    # PCA variance
    pca = PCA(n_components = n_clusters)
    reduced_data = pca.fit_transform(kmeans_data[:,1:])
    
    var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
    lbls = [str(x) for x in range(1,len(var)+1)]
    ax.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)

    ax.set_ylim(0, 30)
    ax.set_yticks([0, 15, 30])
    ax.set_xticks([2, 4, 6, 8])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Variance explained (%)', fontsize = 18)
    ax.set_xlabel('# components', fontsize = 18)
    fig.savefig('./figs/clustering_pcavar.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

    # WCSS
    fig, ax = plt.subplots(figsize = (8,5))

    wcss = []
    for i in range(2,8):
        model = KMeans(n_clusters = i, init = "k-means++", n_init = 4, random_state = 42)
        model.fit(kmeans_data[:,1:])
        wcss.append(model.inertia_)
        
    ax.plot(range(2,8), wcss, lw = 2)

    ax.set_ylim(1000, 2000)
    ax.set_yticks([1000, 1500, 2000])
    ax.set_xticks([2, 3, 4, 5, 6, 7])
    ax.set_ylabel('WCSS', fontsize = 18)
    ax.set_xlabel('# clusters', fontsize = 18) 
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Within Cluster Sum of Squares
    fig.savefig('./figs/clustering_wcss.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

    # and running the kmeans 
    n_clusters = 2
    pca = PCA(n_components = n_clusters)
    reduced_data = pca.fit_transform(kmeans_data[:,1:])
    # all data points above 3,3 are considered outliers 
    reduced_data = np.array([x for x in reduced_data if x[0] < 3 and x[1] < 3])
    np.save('./data/%s/pca_components.npy' % prm.postprocess_name, pca.components_)
    mcp = np.abs(pca.components_.mean(axis = 0)) # absolute mean components    
    resilience_score = (mcp[1]*norm_nkr_b50 + mcp[3]*(1-norm_nkr_n) + mcp[2]*(1-norm_nkr_b)) + (mcp[0]*norm_bthetamax + mcp[4]*(1-norm_cv)) + (mcp[8]*norm_early_ratio + mcp[6]*norm_delay)
    np.save('./data/%s/resilience_score.npy' % prm.postprocess_name, utils.norm_data(resilience_score)) # this is automatically saved in the same order as prm.cluster_list

    kmeans = KMeans(init = 'k-means++', n_clusters = 2, n_init = 100, random_state = 42)
    kmeans.fit(reduced_data)

    # Making a kernel
    x_min, x_max = reduced_data[:, 0].min() - 0.2, reduced_data[:, 0].max() + .2
    y_min, y_max = reduced_data[:, 1].min() - .2, reduced_data[:, 1].max() + .2
    h = .01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    colors = []
    for lab in kmeans.labels_ :
        if lab == 0 :
            colors.append(prm.col_tuned)
        else :
            colors.append(prm.col_untuned)
            
    fig, ax = plt.subplots(figsize = (6,6))

    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 100, linewidths = 3,
            color = 'k', zorder = 10)

    #reduced_data[np.where(reduced_data > 3)[0],:][:,0] = 3 # fixing outside of axis bounds
    for i, el in enumerate(reduced_data) :
        if i in np.where(reduced_data > 3)[0] :
            reduced_data[i][0] = 3
    ax.scatter(reduced_data[:,0], reduced_data[:,1], 
            color = colors, s = 10)

    do_kernel = False
    if do_kernel :
        ax.imshow(Z, interpolation = 'nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                    cmap=plt.cm.Reds,
                    aspect="auto",
                    origin="lower")
        
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('PCA 1', fontsize=18)
    ax.set_ylabel('PCA 2', fontsize=18)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([-3,0,3])
    ax.set_yticks([-3,0,3])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #fig.savefig('./output/fig_3_clusters.pdf', bbox_inches='tight', dpi=200, transparent=True)
    fig.savefig('./figs/clustering_clusters.pdf', bbox_inches='tight', dpi=200, transparent=True)

    # And writing down some stats
    print('K-means clustering done !')
    tuned, untuned = [],[]
    for i, lab in enumerate(kmeans.labels_) :
        if lab == 0 :
            tuned.append(lst_clusters_kmeans[i])
        else :
            untuned.append(lst_clusters_kmeans[i])
    print('Tuned clusters : ', len(tuned))
    print('Untuned clusters : ', len(untuned))
    print('Are example neurons in tuned ?')
    for n in prm.ex_neurons :
        print(n, n in tuned)
    np.save('./data/%s/kmeans_tuned_lst.npy' % prm.postprocess_name, tuned) 
    print('Clustering done !')
    

# --------------------------------------------------------------
# Compute direction selectivity as an internal control
# --------------------------------------------------------------   
def make_direction_selectivity():
    '''
    Computes direction selectivity for each cluster
    as the difference between the mean firing rate of the preferred direction
    and the mean firing rate of the opposite direction
    '''
    di_lst = []
    for cluster_path in tqdm(prm.cluster_list, desc = 'Computing direction selectivity') :
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        
        spiketimes = np.load(load_folder + '/spiketimes.npy')
        seq_contents = np.load(load_folder + '/sequences_contents.npy', allow_pickle = True)
        opti_delay = np.load(load_folder + '/optimal_delay.npy')
        
        tc_array = np.zeros((len(prm.B_thetas), len(prm.thetas), len(prm.phases)))
        
        for i_seq, seq in enumerate(seq_contents) :
            beg_seq = seq['sequence_beg'] + (opti_delay * prm.fs)
            end_seq = beg_seq + (.3 * prm.fs)
            
            spikes_where = np.where((spiketimes > beg_seq) & (spiketimes < end_seq))[0]
            spike_count = len(spikes_where)
            
            idx_phase = np.where(prm.phases == seq['sequence_phase'])[0]
            idx_bt = np.where(prm.B_thetas == seq['sequence_btheta'])[0]
            idx_t = np.where(prm.thetas == seq['sequence_theta'])[0]
            
            tc_array[idx_bt, idx_t, idx_phase] += spike_count
            
        tc_array /= 30
        
        pref_tc = np.argmax((np.max(tc_array[-1,:,0]),
                            np.max(tc_array[-1,:,-1])))
        pref_ori = np.argmax(tc_array[-1,:,pref_tc]) #cleverish indexing
        
        r_pref = tc_array[-1,pref_ori,pref_tc]
        r_null = tc_array[-1,pref_ori,1 if pref_tc == 0 else 0]
        di = (r_pref - r_null)/r_pref
        if di < 0. :
            print(cluster_path)
            di = 0.
        di_lst.append({'cluster': cluster_path, 
                        'di': di})
        
    
    tuned_di = []
    for tuned_neuron in prm.tuned_lst :
        for di_el in di_lst :
            if di_el['cluster'] == tuned_neuron :
                tuned_di.append(di_el['di'])

    untuned_di = []
    for untuned_neuron in prm.untuned_lst :
        for di_el in di_lst :
            if di_el['cluster'] == untuned_neuron :
                untuned_di.append(di_el['di'])
            
    fig, ax = plt.subplots(figsize = (6,6))

    c = prm.col_tuned
    ax.boxplot(tuned_di,
                        positions = [0],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot(untuned_di,
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
    ax.set_ylabel('Direction index', fontsize = 18)
    ax.set_yticks(np.linspace(0., 1., 3))

    ax.set_xticks([])
    ax.set_ylim([-.01, 1])
    fig.savefig('./figs/clustering_direction.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    print('Direction selectivity stats :')
    print(mannwhitneyu(tuned_di, untuned_di, alternative = 'two-sided'))


# --------------------------------------------------------------
# Compute CSD depths for the neurons
# --------------------------------------------------------------      
def make_depth():
    '''
    Transforms the depth computed from Current Source Density analysis 
    into a depth relative to the 0 (pia)
    TODO : move the ipynb for CSD into this repo
    '''
    min_confidence = 4
    
    tuned_csds = []
    for cluster_path in prm.tuned_lst :
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path) 
        with open(load_folder + '/cluster_info.py', 'r') as file :
            data = file.read().splitlines()
            depth = int(data[0].split('= ')[1])
        tuned_csds.append(cluster_path+'_d_'+str(depth))
    tuned_masks = csd_masks(tuned_csds, min_confidence = min_confidence)

    untuned_csds = []
    for cluster_path in prm.untuned_lst :
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path) 
        with open(load_folder + '/cluster_info.py', 'r') as file :
            data = file.read().splitlines()
            depth = int(data[0].split('= ')[1])
        untuned_csds.append(cluster_path+'_d_'+str(depth))
    untuned_masks = csd_masks(untuned_csds, min_confidence = min_confidence)
    
    tuned_masks = np.array([len(np.where(x == True)[0]) for x in tuned_masks])
    tuned_masks = tuned_masks / np.sum(tuned_masks)
    untuned_masks = np.array([len(np.where(x == True)[0]) for x in untuned_masks])
    untuned_masks = untuned_masks/ np.sum(untuned_masks)
    
    fig, ax = plt.subplots(figsize = (8,6))
    ax.barh(np.arange(3), tuned_masks, height = .4,
        color = prm.col_tuned)
    ax.barh(np.arange(3)+.4, untuned_masks, height = .4,
        color = prm.col_untuned)

    for i0, mask in enumerate(tuned_masks) :
        ax.text(.015, np.arange(3)[i0]-.05, s = '%.1f %%' %(mask*100), c = 'w', fontsize = 14)
        
    for i0, mask in enumerate(untuned_masks) :
        ax.text(.015, np.arange(3)[i0]+.4-.05, s = '%.1f %%' %(mask*100), c = 'w', fontsize = 14)

    ax.set_yticks(np.arange(3) + .2)
    ax.set_yticklabels(['Infragranular', 'Granular', 'Supragranular'], fontsize = 14)
    ax.set_xticks([])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.savefig('./figs/clustering_depth.pdf', bbox_inches='tight', dpi=200, transparent=True)


# --------------------------------------------------------------
# Wrapper function for CSD arrays 
# TODO Find the CSD notebook, it must be somewhere in the old repo
# --------------------------------------------------------------   
def csd_masks(cluster_list, min_confidence) :
    '''
    Returns three masks of infra, gra and supra neurons based on hand-discriminated CSD
    '''
    inf_mask = np.zeros_like(cluster_list, dtype = bool)
    gra_mask = np.zeros_like(cluster_list, dtype = bool)
    sup_mask = np.zeros_like(cluster_list, dtype = bool)
    
    depth_dicts = np.load('./data/depth_dicts.npy', allow_pickle = True)
    
    for i0, cluster in enumerate(cluster_list) :
        cluster_depth = int(cluster.split('_d_')[1])
        insertion_name = '_'.join(cluster.split('_')[:2])
        
        for depth_dico in depth_dicts :
            if depth_dico['name'] == insertion_name and depth_dico['confidence'] >= min_confidence :
                if cluster_depth >= (32-depth_dico['SG']) :
                    sup_mask[i0] = True
                elif cluster_depth >=(32-depth_dico['G']) and cluster_depth < (32-depth_dico['SG']) :
                    gra_mask[i0] = True
                elif cluster_depth >=(32-depth_dico['IG']) and cluster_depth < (32-depth_dico['G']) :
                    inf_mask[i0] = True
                    
    return inf_mask, gra_mask, sup_mask