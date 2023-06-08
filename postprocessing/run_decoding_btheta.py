#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
File for decoding the orientation variance Btheta for all neurons
is usually run after decoding_theta, but before decoding_bthetatheta
"""

import params as prm
import utils 
import utils_decoding as dec_utils
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import os 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from scipy import stats
from mlxtend.evaluate import permutation_test

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --------------------------------------------------------------
# Decoding of orientation variance Btheta for all neurons
# --------------------------------------------------------------
def make_btheta_decoding_all() :    
    '''
    Meta function for decoding the orientation variance Btheta for all neurons
    '''
    print('Doing the decoding for Btheta (orientation variance)')
    
    # DECODING ------------------------------------
    print('Doing the decoding on all the neurons with K-fold = %s' % prm.n_splits)
    if not os.path.exists('./data/%s/decoding_btheta_all_kfold.npy' % prm.postprocess_name):
        kfold_scores = np.zeros((len(prm.timesteps), prm.n_splits))

        # Data
        if not os.path.exists('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name)):
            data, labels, le = dec_utils.par_load_data(timesteps = prm.timesteps, target_clusters = prm.cluster_list,
                                                    target_btheta = None, target_theta = None, data_type = 'bt_decoding',
                                                    disable_tqdm = False
                                                    )
            np.save('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name), [data, labels, le])
        else : 
            data, labels, le = np.load('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name), allow_pickle = True)

        # Classifying
        logreg = LogisticRegression(**prm.opts_LR)
        for ibin in tqdm(range(data.shape[0]), desc = 'Decoding') :
            scores = cross_val_score(logreg, data[ibin,:,:], labels, 
                                    cv = prm.n_splits, 
                                    scoring = 'balanced_accuracy')
            kfold_scores[ibin, :] = scores

        np.save('./data/%s/decoding_btheta_all_kfold.npy'% prm.postprocess_name, kfold_scores)
    else :
        kfold_scores = np.load('./data/%s/decoding_btheta_all_kfold.npy'% prm.postprocess_name)
    
    
    # PLOTTING ------------------------------------
    fig, ax = plt.subplots(figsize = (9,6))

    kfold_means = kfold_scores.mean(axis = -1)
    kfold_stderr = kfold_scores.std(axis = -1)
    
    ax.plot(prm.timesteps + prm.win_size, kfold_means, color = prm.col_bt)
    ax.fill_between(prm.timesteps + prm.win_size,
                kfold_means + kfold_stderr, kfold_means - kfold_stderr,
                facecolor = prm.col_bt, edgecolor = None, alpha = .7)
    mod_t = prm.timesteps + prm.win_size
    ax.hlines(.95, 0., .3, color = 'k', linewidth = 4)
    ax.axhline(1/8, c = 'gray', linestyle = '--')
    print('Max btheta decoding accuracy at time %.2fs = %.2f' % (mod_t[np.argmax(kfold_means)],
                                            np.max(kfold_means)))
        
    ax.set_xlabel('PST (ms)', fontsize = 18)
    ax.set_ylabel('Classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ax.set_xticklabels((labs*1000).astype(np.int16))
    yticks = np.linspace(0, 1., 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,1))
    ax.set_xlim(prm.timesteps[0]+prm.win_size, prm.timesteps[-1]+prm.win_size)
    ax.set_ylim(0, 1.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig('./figs/decoding_btheta_all_timesteps.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    


# --------------------------------------------------------------
# Making population TC for all neurons
# this is a separate instance because we report Bthetas x Bthetas 
# --------------------------------------------------------------
def make_btheta_population_tc_all():
    print('Doing the population TC for Btheta (orientation variance)')
    
    # DECODING --------------------------------------------------
    if not os.path.exists('./data/%s/decoding_btheta_all_population_tc.npy' % prm.postprocess_name):
        population_likelihoods = np.zeros((len(prm.timesteps), prm.N_B_thetas, prm.N_B_thetas)) 
        
        # Data
        if not os.path.exists('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name)):
            data, labels, le = dec_utils.par_load_data(timesteps = prm.timesteps, target_clusters = prm.cluster_list,
                                                    target_btheta = None, target_theta = None, data_type = 'bt_decoding',
                                                    disable_tqdm = False
                                                    )
            np.save('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name), [data, labels, le])
        else : 
            data, labels, le = np.load('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name), allow_pickle = True)

        # Classifying
        logreg = LogisticRegression(**prm.opts_LR)

        for ibin in tqdm(range(data.shape[0]), desc = 'Decoding') :
            xtrain, xtest, ytrain, ytest = train_test_split(data[ibin,:,:], labels, test_size=prm.test_size, random_state=42)
            logreg.fit(xtrain, ytrain)
        
            proba = logreg.predict_proba(xtest)

            for i_test in np.unique(ytest):
                population_likelihoods[ibin, i_test, :] = proba[ytest==i_test, :].mean(axis=0)

        np.save('./data/%s/decoding_btheta_all_population_tc.npy'% prm.postprocess_name, population_likelihoods)
    else :
        population_likelihoods = np.load('./data/%s/decoding_btheta_all_population_tc.npy'% prm.postprocess_name, allow_pickle = True)
        
    # PLOTTING --------------------------------------------------
    fig, axs = plt.subplots(figsize = (15, 4), nrows = 1, ncols = 4)

    for it, t in enumerate([20, 30, 40, 50]) :
        ax = axs[it]

        ll = population_likelihoods[t, :, :]
        local_ll = []
        for ibtheta in range(prm.N_B_thetas):
            local_ll.append(np.roll(ll[ibtheta,:], shift = -ibtheta+4, axis = -1))
                
        raw_ll = np.mean(local_ll, axis = 0)
        fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
        fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    **fitted_ll_params)
        ax.plot(np.linspace(-np.pi, np.pi, 100), fitted_ll,
                    color = prm.col_bt, lw = 2)

        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_yticks([0, .5, 1.])
        if it == 0:
            ax.set_xticklabels([-17.5, 0,  17.5])
            ax.set_xlabel(r'Btheta true (°)', fontsize = 18)
            ax.set_ylabel('likelihood correct pred.', fontsize = 18)
            ax.tick_params(axis='both', which='major', labelsize=14)
        else :
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
                
    fig.tight_layout()
    fig.savefig('./figs/decoding_btheta_all_population_tc.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)



# --------------------------------------------------------------
# Decoding of orientation variance for the two groups
# --------------------------------------------------------------
def make_btheta_decoding_groups():
    
    # DECODING --------------------------------------------------
    print('Doing the decoding on the two groups with boostraps')
    if not os.path.exists('./data/%s/decoding_btheta_groups_bootstrap.npy'% prm.postprocess_name):
        bootstrapped_results = np.zeros((2, prm.n_bootstrap, len(prm.timesteps))) # we only report the mean per kfold
        for ibootstrap in tqdm(range(prm.n_bootstrap), desc = 'Bootstrapping') :
            # Randomly pick subgroups in each population
            np.random.seed(prm.seed+ibootstrap)
            picked_res = np.random.choice(prm.tuned_lst, size = prm.n_subgroups)
            picked_vul = np.random.choice(prm.untuned_lst, size = prm.n_subgroups)
            idxs_res = [list(prm.cluster_list).index(x) for x in picked_res]
            idxs_vul = [list(prm.cluster_list).index(x) for x in picked_vul]
            
            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name), allow_pickle = True)
            data_res = data[:, :, idxs_res]
            data_vul = data[:, :, idxs_vul]

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)
            
            for ibin in range(data.shape[0]) :
                scores_res = cross_val_score(logreg, data_res[ibin,:,:], labels, 
                                            cv = prm.n_splits, 
                                            scoring = 'balanced_accuracy')
                scores_vul = cross_val_score(logreg, data_vul[ibin,:,:], labels,
                                            cv = prm.n_splits,
                                            scoring='balanced_accuracy')
                bootstrapped_results[0, ibootstrap, ibin] = np.mean(scores_res)
                bootstrapped_results[1, ibootstrap, ibin] = np.mean(scores_vul)

        np.save('./data/%s/decoding_btheta_groups_bootstrap.npy'% prm.postprocess_name, bootstrapped_results)
    else :
        bootstrapped_results = np.load('./data/%s/decoding_btheta_groups_bootstrap.npy' % prm.postprocess_name)
    
    # PLOTTING --------------------------------------------------------
    fig, ax = plt.subplots(figsize = (8,5))
    
    # Untuned bootstrapping
    bootstrap_mean = bootstrapped_results[1,:,:].mean(axis = 0)
    bootstrap_std = bootstrapped_results[1,:,:].std(axis = 0)
    
    ax.plot(prm.timesteps + prm.win_size, bootstrap_mean, color = prm.col_untuned)
    ax.fill_between(prm.timesteps + prm.win_size,
                bootstrap_mean + bootstrap_std, bootstrap_mean - bootstrap_std,
                facecolor = prm.col_untuned, edgecolor = None, alpha = .7,
                label = prm.name_untuned)
    
    # Tuned bootstrapping
    bootstrap_mean = bootstrapped_results[0,:,:].mean(axis = 0)
    bootstrap_std = bootstrapped_results[0,:,:].std(axis = 0)
    
    ax.plot(prm.timesteps + prm.win_size, bootstrap_mean, color = prm.col_tuned)
    ax.fill_between(prm.timesteps + prm.win_size,
                bootstrap_mean + bootstrap_std, bootstrap_mean - bootstrap_std,
                facecolor = prm.col_tuned, edgecolor = None, alpha = .7,
                label = prm.name_tuned)
    
    # Statistical tests
    pvals_array = np.zeros_like(prm.timesteps)
    for i, t in tqdm(enumerate(prm.timesteps), total = len(prm.timesteps), desc = 'Computing pvals'):
        p, val = stats.wilcoxon(bootstrapped_results[0,:, i], bootstrapped_results[1,:, i],
                                    alternative = 'two-sided')
        '''p = permutation_test(bootstrapped_results[0,:, i],
                            bootstrapped_results[1,:, i],
                            func = 'x_mean > y_mean',
                            num_rounds=1000)'''
        pvals_array[i] = p
        
    show_pvals = True
    if show_pvals :
        for i, pval in enumerate(pvals_array[:-1]) :
            if pval <0.01: # don't care about  pre-stim
                ax.axvspan(prm.timesteps[i]+prm.win_size, prm.timesteps[i+1]+prm.win_size, alpha = .3,
                        facecolor = 'gray', edgecolor = 'None',
                        zorder = -20)
    
    ax.hlines(.95, 0., .3, color = 'k', linewidth = 4)
    ax.axhline(1/8, c = 'gray', linestyle = '--')
    
    ax.set_xlabel('PST (ms)', fontsize = 18)
    ax.set_ylabel('Classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ax.set_xticklabels((labs*1000).astype(np.int16))
    yticks = np.linspace(0, 1., 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,1))
    ax.set_xlim(prm.timesteps[0]+prm.win_size, prm.timesteps[-1]+prm.win_size)
    ax.set_ylim(0, 1.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = (.025, .75), frameon = False, fontsize = 14, markerscale = .2)
    
    fig.tight_layout()
    fig.savefig('./figs/decoding_btheta_groups_timesteps.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    

# --------------------------------------------------------------
# Making population TC for the two groups 
# --------------------------------------------------------------
def make_btheta_population_tc_groups():

    # DECODING ------------------------------------------------------
    print('Doing population tuning curve on the two groups with boostraps')
    if not os.path.exists('./data/%s/decoding_btheta_groups_population_tc.npy'% prm.postprocess_name):
        bootstrapped_likelihoods= np.zeros((2, prm.n_bootstrap,  len(prm.timesteps), prm.N_B_thetas, prm.N_B_thetas)) # we only report the mean per kfold
        for ibootstrap in tqdm(range(prm.n_bootstrap), desc = 'Bootstrapping') :
            # Randomly pick subgroups in each population
            np.random.seed(prm.seed+ibootstrap)
            picked_res = np.random.choice(prm.tuned_lst, size = prm.n_subgroups)
            picked_vul = np.random.choice(prm.untuned_lst, size = prm.n_subgroups)
            idxs_res = [list(prm.cluster_list).index(x) for x in picked_res]
            idxs_vul = [list(prm.cluster_list).index(x) for x in picked_vul]
            
            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name), allow_pickle = True)
            data_res = data[:, :, idxs_res]
            data_vul = data[:, :, idxs_vul]

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)
            
            for ibin in range(data.shape[0]) :
                xtrain, xtest, ytrain, ytest = train_test_split(data_res[ibin,:,:], labels, test_size=prm.test_size, random_state=42)
                logreg.fit(xtrain, ytrain)
                proba = logreg.predict_proba(xtest)
                for i_test in np.unique(ytest):
                    bootstrapped_likelihoods[0, ibootstrap, ibin, i_test, :] = proba[ytest==i_test, :].mean(axis=0)
                    
                xtrain, xtest, ytrain, ytest = train_test_split(data_vul[ibin,:,:], labels, test_size=prm.test_size, random_state=42)
                logreg.fit(xtrain, ytrain)
                proba = logreg.predict_proba(xtest)
                for i_test in np.unique(ytest):
                    bootstrapped_likelihoods[1, ibootstrap, ibin, i_test, :] = proba[ytest==i_test, :].mean(axis=0)

        np.save('./data/%s/decoding_btheta_groups_population_tc.npy'% prm.postprocess_name, bootstrapped_likelihoods)
    else :
        bootstrapped_likelihoods = np.load('./data/%s/decoding_btheta_groups_population_tc.npy' % prm.postprocess_name)
        
    # PLOTTING --------------------------------------------------
    population_likelihood_res = bootstrapped_likelihoods[0, :, :, :, :].mean(axis = 0)
    population_likelihood_vul = bootstrapped_likelihoods[1, :, :, :, :].mean(axis = 0)
    
    fig, axs = plt.subplots(figsize = (15,4), nrows = 1, ncols = 4)
    for it, t in enumerate([20, 30, 40, 50]) :

        ax = axs[it]
        ll = population_likelihood_res[t, :, :]
        local_ll = []
        for ibtheta in range(prm.N_B_thetas):
            local_ll.append(np.roll(ll[ibtheta,:], shift = -ibtheta+4, axis = -1))

        raw_ll = np.mean(local_ll, axis = 0)
        fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
        fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    **fitted_ll_params)
        ax.plot(np.linspace(-np.pi, np.pi, 100), fitted_ll,
                    color = prm.col_tuned, lw = 2)

        ll = population_likelihood_vul[t, :, :]
        local_ll = []
        for ibtheta in range(prm.N_B_thetas):
            local_ll.append(np.roll(ll[ibtheta,:], shift = -ibtheta+4, axis = -1))

        raw_ll = np.mean(local_ll, axis = 0)
        fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
        fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    **fitted_ll_params)
        ax.plot(np.linspace(-np.pi, np.pi, 100), fitted_ll,
                    color = prm.col_untuned, lw = 2)
        
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_yticks([0, .5, 1.])
        if it == 0:
            ax.set_xticklabels([-17.5, 0,  17.5])
            ax.set_xlabel(r'Btheta true (°)', fontsize = 18)
            ax.set_ylabel('likelihood correct pred.', fontsize = 18)
            ax.tick_params(axis='both', which='major', labelsize=14)
        else :
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig('./figs/decoding_btheta_groups_population_tc.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)