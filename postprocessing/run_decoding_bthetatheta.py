#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
Runs the decoding of both orientation (theta) and variance (btheta) for all neurons
i.e. a 96 class problem (12 orientations x 8 variances)
"""

import params as prm
import utils 
import utils_decoding as dec_utils
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn import metrics
import os 
import matplotlib.pyplot as plt 
from matplotlib import colors as mcols
from tqdm import tqdm 
from scipy import stats
#from mlxtend.evaluate import permutation_test
from scipy.stats import permutation_test

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --------------------------------------------------------------
# Decoding of orientation variance Btheta x orientation theta for all neurons
# --------------------------------------------------------------
def make_bthetatheta_decoding_all() : 
    '''
    Meta function that calls the decoding of orientation variance Btheta x orientation theta for all neurons
    '''   
    print('Doing the decoding for Btheta (orientation variance) x Theta (orientation)')
    
    # DECODING ------------------------------------
    print('Doing the decoding on all the neurons with K-fold = %s' % prm.n_splits)
    if not os.path.exists('./data/%s/decoding_bthetatheta_all_kfold.npy' % prm.postprocess_name):
        kfold_scores = np.zeros((len(prm.timesteps), prm.n_splits))

        # Data
        if not os.path.exists('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name)):
            data, labels, le = dec_utils.par_load_data(timesteps = prm.timesteps, target_clusters = prm.cluster_list,
                                                    target_btheta = None, target_theta = None, data_type = 'all_t_bt',
                                                    disable_tqdm = False
                                                    )
            np.save('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), [data, labels, le])
        else : 
            data, labels, le = np.load('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), allow_pickle = True)

        # Classifying
        logreg = LogisticRegression(**prm.opts_LR)

        for ibin in tqdm(range(data.shape[0]), desc = 'Decoding') :
            scores = cross_val_score(logreg, data[ibin,:,:], labels, 
                                    cv = prm.n_splits, 
                                    scoring = 'balanced_accuracy')
            kfold_scores[ibin, :] = scores

        np.save('./data/%s/decoding_bthetatheta_all_kfold.npy'% prm.postprocess_name, kfold_scores)
    else :
        kfold_scores = np.load('./data/%s/decoding_bthetatheta_all_kfold.npy'% prm.postprocess_name)
    
    
    # PLOTTING ------------------------------------
    fig, ax = plt.subplots(figsize = (9,6))

    kfold_means = kfold_scores.mean(axis = -1)
    kfold_stderr = kfold_scores.std(axis = -1)
    
    ax.plot(prm.timesteps + prm.win_size, kfold_means, color = 'k')
    ax.fill_between(prm.timesteps + prm.win_size,
                kfold_means + kfold_stderr, kfold_means - kfold_stderr,
                facecolor = 'k', edgecolor = None, alpha = .7)
    mod_t = prm.timesteps + prm.win_size
    ax.hlines(.95, 0., .18, color = 'k', linewidth = 4)
    ax.axhline(1/96, c = 'gray', linestyle = '--')
    print('Max bthetaxtheta decoding accuracy max at time %.2fs = %.2f' % (mod_t[np.argmax(kfold_means)],
                                            np.max(kfold_means)))
        
    ax.set_xlabel('PST (ms)', fontsize = 18)
    ax.set_ylabel('Classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ax.set_xticklabels((labs*1000).astype(np.int16))
    yticks = np.linspace(0, .2, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,2))
    ax.set_xlim(prm.timesteps[0]+prm.win_size, prm.timesteps[-1]+prm.win_size)
    ax.set_ylim(0, .2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = (.025, .65), frameon = False, fontsize = 14, markerscale = 2, ncol = 1)   

    fig.tight_layout()
    fig.savefig('./figs/decoding_bthetatheta_all_timesteps.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    


# --------------------------------------------------------------
# Making population TC for all neurons
# this is a separate instance because we report (Bthetasxthetas) x (Bthetasxthetas) 
# --------------------------------------------------------------
def make_bthetatheta_population_tc_all():
    print('Doing the population TC for Btheta (orientation variance) x Theta (orientation)')
    
    # DECODING --------------------------------------------------
    if not os.path.exists('./data/%s/decoding_bthetatheta_all_population_tc.npy' % prm.postprocess_name):
        population_likelihoods = np.zeros((len(prm.timesteps),
                                        prm.N_B_thetas*prm.N_thetas,
                                        prm.N_B_thetas*prm.N_thetas)) 
        
        # Data
        if not os.path.exists('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name)):
            data, labels, le = dec_utils.par_load_data(timesteps = prm.timesteps, target_clusters = prm.cluster_list,
                                                    target_btheta = None, target_theta = None, data_type = 'all_t_bt',
                                                    disable_tqdm = False
                                                    )
            np.save('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), [data, labels, le])
        else : 
            data, labels, le = np.load('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), allow_pickle = True)

        # Classifying
        logreg = LogisticRegression(**prm.opts_LR)

        for ibin in tqdm(range(data.shape[0]), desc = 'Decoding') :
            xtrain, xtest, ytrain, ytest = train_test_split(data[ibin,:,:], labels, test_size=prm.test_size, random_state=42)
            logreg.fit(xtrain, ytrain)
        
            proba = logreg.predict_proba(xtest)

            for i_test in np.unique(ytest):
                population_likelihoods[ibin, i_test, :] = proba[ytest==i_test, :].mean(axis=0)

        np.save('./data/%s/decoding_bthetatheta_all_population_tc.npy'% prm.postprocess_name, population_likelihoods)
    else :
        population_likelihoods = np.load('./data/%s/decoding_bthetatheta_all_population_tc.npy'% prm.postprocess_name, allow_pickle = True)
        
    # PLOTTING --------------------------------------------------
    fig, axs = plt.subplots(figsize = (15, 4), nrows = 1, ncols = 4)
    for it, t in enumerate([20, 30, 40, 50]) :
        ax = axs[it]

        ll = population_likelihoods[t, :, :]
        local_ll = []
        for iboth in range(prm.N_B_thetas*prm.N_thetas):
            local_ll.append(np.roll(ll[iboth,:], shift = -iboth+48, axis = -1))
        raw_ll = np.mean(local_ll, axis = 0)
        local_params = []
        local_fits = []
        for ibt in range(prm.N_B_thetas) :
            fitted_ll_params = utils.fit_tc(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas], init_kappa = 3)
            fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    bsl = fitted_ll_params['bsl'], mu = fitted_ll_params['mu'],
                                    fmax = fitted_ll_params['fmax'],
                                    kappa = fitted_ll_params['kappa'])        
            local_params.append(fitted_ll_params)
            local_fits.append(fitted_ll)
            
        global_fit = np.concatenate(local_fits)
        global_fit = np.roll(global_fit, shift = -52, axis = 0)
        ax.plot(np.linspace(0, 96, 800), global_fit,
                    color = 'k', lw = 2)
        
        ax.set_xticks([0, 48, 96])
        ax.set_yticks([0, .15/2, 0.15])
        if it == 0:
            ax.set_xticklabels([-48, 0,  48])
            ax.set_xlabel(r'both true (°)', fontsize = 18)
            ax.set_ylabel('likelihood correct pred.', fontsize = 18)
            ax.tick_params(axis='both', which='major', labelsize=14)
        else :
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        ax.set_ylim(0, .15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
                
    fig.tight_layout()
    fig.savefig('./figs/decoding_bthetatheta_all_population_tc.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

# --------------------------------------------------------------
# Decoding of bthetaxtheta for the two groups
# --------------------------------------------------------------
def make_bthetatheta_decoding_groups():
    
    # DECODING --------------------------------------------------
    print('Doing the decoding on the two groups with boostraps')
    if not os.path.exists('./data/%s/decoding_bthetatheta_groups_bootstrap.npy'% prm.postprocess_name):
        bootstrapped_results = np.zeros((2, prm.n_bootstrap, len(prm.timesteps))) # we only report the mean per kfold
        for ibootstrap in tqdm(range(prm.n_bootstrap), desc = 'Bootstrapping') :
            # Randomly pick subgroups in each population
            np.random.seed(ibootstrap)
            picked_res = np.random.choice(prm.tuned_lst, size = prm.n_subgroups)
            picked_vul = np.random.choice(prm.untuned_lst, size = prm.n_subgroups)
            idxs_res = [list(prm.cluster_list).index(x) for x in picked_res]
            idxs_vul = [list(prm.cluster_list).index(x) for x in picked_vul]
            
            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), allow_pickle = True)
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

        np.save('./data/%s/decoding_bthetatheta_groups_bootstrap.npy'% prm.postprocess_name, bootstrapped_results)
    else :
        bootstrapped_results = np.load('./data/%s/decoding_bthetatheta_groups_bootstrap.npy' % prm.postprocess_name)
    
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
    
    ax.hlines(.95, 0., .18, color = 'k', linewidth = 4)
    ax.axhline(1/96, c = 'gray', linestyle = '--')
    
    ax.set_xlabel('PST (ms)', fontsize = 18)
    ax.set_ylabel('Classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ax.set_xticklabels((labs*1000).astype(np.int16))
    yticks = np.linspace(0, .2, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,1))
    ax.set_xlim(prm.timesteps[0]+prm.win_size, prm.timesteps[-1]+prm.win_size)
    ax.set_ylim(0, .2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = (.025, .75), frameon = False, fontsize = 14, markerscale = .2)
    
    fig.tight_layout()
    fig.savefig('./figs/decoding_bthetatheta_groups_timesteps.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    

# --------------------------------------------------------------
# Making population TC for the two groups 
# --------------------------------------------------------------
def make_bthetatheta_population_tc_groups():

    # DECODING ------------------------------------------------------
    print('Doing population tuning curve on the two groups with boostraps')
    if not os.path.exists('./data/%s/decoding_bthetatheta_groups_population_tc.npy'% prm.postprocess_name):
        bootstrapped_likelihoods= np.zeros((2, prm.n_bootstrap,  len(prm.timesteps),
                                            prm.N_B_thetas*prm.N_thetas,
                                            prm.N_B_thetas*prm.N_thetas)) # we only report the mean per kfold
        for ibootstrap in tqdm(range(prm.n_bootstrap), desc = 'Bootstrapping'):
            # Randomly pick subgroups in each population
            np.random.seed(ibootstrap)
            picked_res = np.random.choice(prm.tuned_lst, size = prm.n_subgroups)
            picked_vul = np.random.choice(prm.untuned_lst, size = prm.n_subgroups)
            idxs_res = [list(prm.cluster_list).index(x) for x in picked_res]
            idxs_vul = [list(prm.cluster_list).index(x) for x in picked_vul]
            
            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), allow_pickle = True)
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

        np.save('./data/%s/decoding_bthetatheta_groups_population_tc.npy'% prm.postprocess_name, bootstrapped_likelihoods)
    else :
        bootstrapped_likelihoods = np.load('./data/%s/decoding_bthetatheta_groups_population_tc.npy' % prm.postprocess_name)
        
    # PLOTTING --------------------------------------------------
    population_likelihood_res = bootstrapped_likelihoods[0, :, :, :, :].mean(axis = 0)
    population_likelihood_vul = bootstrapped_likelihoods[1, :, :, :, :].mean(axis = 0)
    
    fig, axs = plt.subplots(figsize = (15,4), nrows = 1, ncols = 4)
    for it, t in enumerate([20, 30, 40, 50]) :
        ax = axs[it]

        ll = population_likelihood_res[t, :, :]
        local_ll = []
        for iboth in range(prm.N_B_thetas*prm.N_thetas):
            local_ll.append(np.roll(ll[iboth,:], shift = -iboth+48, axis = -1))

        
        raw_ll = np.mean(local_ll, axis = 0)
        local_params = []
        local_fits = []
        for ibt in range(prm.N_B_thetas) :
            fitted_ll_params = utils.fit_tc(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas], init_kappa = 3)
            fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    bsl = fitted_ll_params['bsl'], mu = fitted_ll_params['mu'],
                                    fmax = fitted_ll_params['fmax'],
                                    kappa = fitted_ll_params['kappa'])        
            local_params.append(fitted_ll_params)
            local_fits.append(fitted_ll)
            
        global_fit = np.concatenate(local_fits)
        global_fit = np.roll(global_fit, shift = -52, axis = 0)
        ax.plot(np.linspace(0, 96, 800), global_fit,
                    color = prm.col_tuned, lw = 2)

        ax.set_xticks([0, 48, 96])
        ax.set_yticks([0, .15/2, 0.15])
        if it == 0:
            ax.set_xticklabels([-48, 0,  48])
            ax.set_xlabel(r'both true (°)', fontsize = 18)
            ax.set_ylabel('likelihood correct pred.', fontsize = 18)
            ax.tick_params(axis='both', which='major', labelsize=14)
        else :
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        ax.set_ylim(0, .15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig('./figs/decoding_bthetatheta_groups_res_population_tc.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    
    fig, axs = plt.subplots(figsize = (15,4), nrows = 1, ncols = 4)
    for it, t in enumerate([20, 30, 40, 50]) :
        ax = axs[it]

        ll = population_likelihood_vul[t, :, :]
        local_ll = []
        for iboth in range(prm.N_B_thetas*prm.N_thetas):
            local_ll.append(np.roll(ll[iboth,:], shift = -iboth+48, axis = -1))

        
        raw_ll = np.mean(local_ll, axis = 0)
        local_params = []
        local_fits = []
        for ibt in range(prm.N_B_thetas) :
            fitted_ll_params = utils.fit_tc(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas], init_kappa = 3)
            fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    bsl = fitted_ll_params['bsl'], mu = fitted_ll_params['mu'],
                                    fmax = fitted_ll_params['fmax'],
                                    kappa = fitted_ll_params['kappa'])        
            local_params.append(fitted_ll_params)
            local_fits.append(fitted_ll)
            
        global_fit = np.concatenate(local_fits)
        global_fit = np.roll(global_fit, shift = -52, axis = 0)
        ax.plot(np.linspace(0, 96, 800), global_fit,
                    color = prm.col_untuned, lw = 2)

        ax.set_xticks([0, 48, 96])
        ax.set_yticks([0, .15/2, 0.15])
        if it == 0:
            ax.set_xticklabels([-48, 0,  48])
            ax.set_xlabel(r'both true (°)', fontsize = 18)
            ax.set_ylabel('likelihood correct pred.', fontsize = 18)
            ax.tick_params(axis='both', which='major', labelsize=14)
        else :
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        ax.set_ylim(0, .15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig('./figs/decoding_bthetatheta_groups_vul_population_tc.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    tempo_likelihoods = np.load('./data/%s/decoding_bthetatheta_all_population_tc.npy'% prm.postprocess_name, allow_pickle = True)
    all_params = []
    for it, t in enumerate(prm.timesteps) :
        ll = tempo_likelihoods[it, :, :]
        local_ll = []
        for iboth in range(prm.N_B_thetas*prm.N_thetas):
            local_ll.append(np.roll(ll[iboth,:], shift = -iboth+48, axis = -1))

        raw_ll = np.mean(local_ll, axis = 0)
        local_params = []
        for ibt in range(prm.N_B_thetas) :
            fitted_ll_params = utils.fit_tc(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas], init_kappa = 3)      
            local_params.append((utils.cirvar(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas]),fitted_ll_params))
            
        all_params.append(local_params)
    
    untuned_all_params = []
    for it, t in enumerate(prm.timesteps) :
        ll = population_likelihood_vul[it, :, :]
        local_ll = []
        for iboth in range(prm.N_B_thetas*prm.N_thetas):
            local_ll.append(np.roll(ll[iboth,:], shift = -iboth+48, axis = -1))

        raw_ll = np.mean(local_ll, axis = 0)
        local_params = []
        for ibt in range(prm.N_B_thetas) :
            fitted_ll_params = utils.fit_tc(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas], init_kappa = 3)      
            local_params.append((utils.cirvar(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas]),fitted_ll_params))
            
        untuned_all_params.append(local_params)
        
    untuned_all_median_fmax = np.asarray([x[4][1]['fmax'] for x in all_params])
    untuned_all_median_cv = 1-np.asarray([x[4][0] for x in all_params])
    bootstrapped_results = np.load('./data/%s/decoding_bthetatheta_groups_bootstrap.npy' % prm.postprocess_name)
    untuned_accuracies =  bootstrapped_results[1,:,:].mean(axis = 0)
    _ = []
    for i in range(prm.N_B_thetas) :
        if i == 4 : 
            pass
        else :
            _.append(np.asarray([x[i][1]['fmax'] for x in untuned_all_params]))
    untuned_mean_offmedian_fmax = np.mean(_, axis = 0)
    _ = []
    for i in range(prm.N_B_thetas) :
        if i == 4 : 
            pass
        else :
            _.append(np.asarray([x[i][0] for x in untuned_all_params]))
    untuned_mean_offmedian_cv = 1-np.mean(_, axis = 0) 
    
    
    tuned_all_params = []
    for it, t in enumerate(prm.timesteps) :
        ll = population_likelihood_res[it, :, :]
        local_ll = []
        for iboth in range(prm.N_B_thetas*prm.N_thetas):
            local_ll.append(np.roll(ll[iboth,:], shift = -iboth+48, axis = -1))

        raw_ll = np.mean(local_ll, axis = 0)
        local_params = []
        for ibt in range(prm.N_B_thetas) :
            fitted_ll_params = utils.fit_tc(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas], init_kappa = 3)      
            local_params.append((utils.cirvar(raw_ll[(ibt*prm.N_thetas) : (ibt+1)*prm.N_thetas]),fitted_ll_params))
            
        tuned_all_params.append(local_params)
        
    tuned_all_median_fmax = np.asarray([x[4][1]['fmax'] for x in all_params])
    tuned_all_median_cv = 1-np.asarray([x[4][0] for x in all_params])
    tuned_accuracies = bootstrapped_results[0,:,:].mean(axis = 0)
    _ = []
    for i in range(prm.N_B_thetas) :
        if i == 4 : 
            pass
        else :
            _.append(np.asarray([x[i][1]['fmax'] for x in tuned_all_params]))
    tuned_mean_offmedian_fmax = np.mean(_, axis = 0)
    _ = []
    for i in range(prm.N_B_thetas) :
        if i == 4 : 
            pass
        else :
            _.append(np.asarray([x[i][0] for x in tuned_all_params]))
    tuned_mean_offmedian_cv = 1-np.mean(_, axis = 0)

    fig, ax = plt.subplots(figsize = (5, 5))

    ax.scatter(untuned_all_median_cv, untuned_accuracies, color = prm.col_untuned, s = 20)

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(untuned_all_median_cv, untuned_accuracies)
    print('Correlation btheta x theta decoding for ON MEDIAN')
    print('median')
    print('Slope = %.3f ; Intercept = %.3f ; pval = %s' % (slope, intercept, pvalue))
    ax.plot(np.linspace(0, 1, 100),
            intercept + slope * np.linspace(0,1,100), c = prm.col_untuned)
    r, pval = stats.spearmanr(untuned_all_median_cv, untuned_accuracies)
    print('Spearman R = %.3f ; pval = %s'  % (r, pval))

    ax.scatter(tuned_all_median_cv, tuned_accuracies, color = prm.col_tuned, s = 20)

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(tuned_all_median_cv, tuned_accuracies)
    print('median')
    print('Slope = %.3f ; Intercept = %.3f ; pval = %s' % (slope, intercept, pvalue))
    ax.plot(np.linspace(0, 1, 100),
            intercept + slope * np.linspace(0,1,100), c = prm.col_tuned)
    r, pval = stats.spearmanr(tuned_all_median_cv, tuned_accuracies)
    print('Spearman R = %.3f ; pval = %s'  % (r, pval))

    ax.set_xlim(0, 1)
    ax.set_ylim(0,.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks([0, .5, 1.])
    ax.set_yticks([0, .1, .2])

    ax.set_xlabel('1 - Population circular variance', fontsize = 18)
    ax.set_ylabel('Classification accuracy', fontsize = 18)

    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.plot((0,1), (0,1), c = 'gray', linestyle = '--', alpha = .75, 
            zorder = -1)

    fig.tight_layout()
    fig.savefig('./figs/decoding_bthetatheta_linregress_median.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    
    fig, ax = plt.subplots(figsize = (5, 5))


    ax.scatter(untuned_mean_offmedian_cv, untuned_accuracies, color = prm.col_untuned, s = 20)

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(untuned_mean_offmedian_cv, untuned_accuracies)

    print('offmedian')
    print('Slope = %.3f ; Intercept = %.3f ; pval = %s' % (slope, intercept, pvalue))
    ax.plot(np.linspace(0, 1, 100),
            intercept + slope * np.linspace(0,1,100), c = prm.col_untuned)
    r, pval = stats.spearmanr(untuned_mean_offmedian_cv, untuned_accuracies)
    print('Spearman R = %.3f ; pval = %s'  % (r, pval))

    ax.scatter(tuned_mean_offmedian_cv, tuned_accuracies, color = prm.col_tuned, s = 20)

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(tuned_mean_offmedian_cv, tuned_accuracies)

    print('offmedian')
    print('Slope = %.3f ; Intercept = %.3f ; pval = %s' % (slope, intercept, pvalue))
    ax.plot(np.linspace(0, 1, 100),
            intercept + slope * np.linspace(0,1,100), c = prm.col_tuned)
    r, pval = stats.spearmanr(tuned_mean_offmedian_cv, tuned_accuracies)
    print('Spearman R = %.3f ; pval = %s'  % (r, pval))

    ax.set_xlim(0, 1)
    ax.set_ylim(0,.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks([0, .5, 1.])
    ax.set_yticks([0, .1, .2])


    ax.set_xlabel('1 - Population circular variance', fontsize = 18)
    ax.set_ylabel('Classification accuracy', fontsize = 18)

    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.plot((0,1), (0,1), c = 'gray', linestyle = '--', alpha = .75, 
            zorder = -1)

    fig.tight_layout()
    fig.savefig('./figs/decoding_bthetatheta_linregress_offmedian.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

# --------------------------------------------------------------
# Running a marginalized decoding over btheta
# --------------------------------------------------------------
def make_bthetatheta_population_marginalization_groups():
    
    # DECODING --------------------------------------------------
    print('Doing the decoding on the two groups with boostraps AND marginalization')
    if not os.path.exists('./data/%s/decoding_bthetatheta_groups_bootstrap_marginalization.npy'% prm.postprocess_name):
        bootstrapped_cm = np.zeros((2, prm.n_bootstrap, len(prm.timesteps), prm.n_splits), dtype = object) 
        for ibootstrap in tqdm(range(prm.n_bootstrap), desc = 'Bootstrapping') :
            # Randomly pick subgroups in each population
            np.random.seed(ibootstrap)
            picked_res = np.random.choice(prm.tuned_lst, size = prm.n_subgroups)
            picked_vul = np.random.choice(prm.untuned_lst, size = prm.n_subgroups)
            idxs_res = [list(prm.cluster_list).index(x) for x in picked_res]
            idxs_vul = [list(prm.cluster_list).index(x) for x in picked_vul]
            
            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), allow_pickle = True)
            data_res = data[:, :, idxs_res]
            data_vul = data[:, :, idxs_vul]
            
            kf = KFold(n_splits = prm.n_splits)

            for ibin in range(data.shape[0]) :
                for isplit, (train_index, test_index) in enumerate(kf.split(data_res[ibin,:,:], labels)):
                    xtrain, xtest = data_res[ibin,train_index,:], data_res[ibin,test_index,:]
                    ytrain, ytest = labels[train_index], labels[test_index]
                    logreg = LogisticRegression(**prm.opts_LR)
                    logreg.fit(xtrain, ytrain)
                    cm = metrics.confusion_matrix(ytest, logreg.predict(xtest), normalize = 'all')
                    cm *= len(le.classes_)
                    bootstrapped_cm[0, ibootstrap, ibin, isplit] = cm
                    
                for isplit, (train_index, test_index) in enumerate(kf.split(data_vul[ibin,:,:], labels)):
                    xtrain, xtest = data_vul[ibin,train_index,:], data_vul[ibin,test_index,:]
                    ytrain, ytest = labels[train_index], labels[test_index]
                    logreg = LogisticRegression(**prm.opts_LR)
                    logreg.fit(xtrain, ytrain)
                    cm = metrics.confusion_matrix(ytest, logreg.predict(xtest), normalize = 'all')
                    cm *= len(le.classes_)
                    bootstrapped_cm[1, ibootstrap, ibin, isplit] = cm

        np.save('./data/%s/decoding_bthetatheta_groups_bootstrap_marginalization.npy'% prm.postprocess_name, bootstrapped_cm)
    else :
        bootstrapped_cm = np.load('./data/%s/decoding_bthetatheta_groups_bootstrap_marginalization.npy' % prm.postprocess_name, allow_pickle = True)
        
    # Recollapsing the data  - tuned (TODO : do it for untuned and tuned in a single loop)
    all_btt_cm_tuned, btt_cm_means_tuned, btt_cm_accs_tuned, btt_cm_stds_tuned =[],[],[],[]
    data = bootstrapped_cm[0,:,:,:].mean(axis = 0) # average across bootstraps
    for i, t in enumerate(prm.timesteps) :
        cm_mean_ = np.mean(data[i], axis = 0)
        margin_cm = []
        for x in np.arange(prm.N_B_thetas):
            for y in np.arange(prm.N_B_thetas):
                a = cm_mean_[x*prm.N_thetas : (x+1)*prm.N_thetas,
                            y*prm.N_thetas : (y+1)*prm.N_thetas]
                margin_cm.append(a)
        margin_cm = np.asarray(margin_cm)
        cm_mean = np.mean(margin_cm, axis = 0)
        
        btt_cm_means_tuned.append(cm_mean)
        btt_cm_accs_tuned.append(np.mean(np.diag(cm_mean)))
        all_btt_cm_tuned.append(np.diag(cm_mean))
        
        # The standard deviation is the std of all possible diagonals 
        diags = []
        for mat in range(margin_cm.shape[0]) :
            diag = np.diag(margin_cm[mat])
            diags.append(diag)
        btt_cm_stds_tuned.append(np.std(diags))

    btt_cm_accs_tuned = np.asarray(btt_cm_accs_tuned) * 8
    btt_cm_stds_tuned = np.asarray(btt_cm_stds_tuned)
    btt_cm_means_tuned = np.asarray(btt_cm_means_tuned) * 8
    all_btt_cm_tuned = np.asarray(all_btt_cm_tuned) * 8
    
    all_btt_cm_untuned, btt_cm_means_untuned, btt_cm_accs_untuned, btt_cm_stds_untuned =[],[],[],[]
    data = bootstrapped_cm[1,:,:,:].mean(axis = 0) # average across bootstraps
    for i, t in enumerate(prm.timesteps) :
        cm_mean_ = np.mean(data[i], axis = 0)
        margin_cm = []
        for x in np.arange(prm.N_B_thetas):
            for y in np.arange(prm.N_B_thetas):
                a = cm_mean_[x*prm.N_thetas : (x+1)*prm.N_thetas,
                            y*prm.N_thetas : (y+1)*prm.N_thetas]
                margin_cm.append(a)
        margin_cm = np.asarray(margin_cm)
        cm_mean = np.mean(margin_cm, axis = 0)
        
        btt_cm_means_untuned.append(cm_mean)
        btt_cm_accs_untuned.append(np.mean(np.diag(cm_mean)))
        all_btt_cm_untuned.append(np.diag(cm_mean))
        
        # The standard deviation is the std of all possible diagonals 
        diags = []
        for mat in range(margin_cm.shape[0]) :
            diag = np.diag(margin_cm[mat])
            diags.append(diag)
        btt_cm_stds_untuned.append(np.std(diags))

    btt_cm_accs_untuned = np.asarray(btt_cm_accs_untuned) * 8
    btt_cm_stds_untuned = np.asarray(btt_cm_stds_untuned)
    btt_cm_means_untuned = np.asarray(btt_cm_means_untuned) * 8
    all_btt_cm_untuned = np.asarray(all_btt_cm_untuned) * 8
    
    fig, ax = plt.subplots(figsize = (8, 5))

    ax.plot(prm.timesteps + prm.win_size, btt_cm_accs_untuned, color = prm.col_untuned, linewidth = 3)
    ax.plot(prm.timesteps + prm.win_size, btt_cm_accs_tuned, color = prm.col_tuned, linewidth = 3)
    print(np.max(btt_cm_accs_untuned)/(1/12))
    print(np.max(btt_cm_accs_tuned)/(1/12))
    ax.fill_between(prm.timesteps + prm.win_size,
                    btt_cm_accs_untuned + btt_cm_stds_untuned,
                    btt_cm_accs_untuned - btt_cm_stds_untuned,
                    facecolor = prm.col_untuned, edgecolor = None, alpha = .5)

    ax.fill_between(prm.timesteps + prm.win_size,
                    btt_cm_accs_tuned + btt_cm_stds_untuned,
                    btt_cm_accs_tuned - btt_cm_stds_tuned,
                    facecolor =  prm.col_tuned, edgecolor = None, alpha = .5)
    
    # Statistical tests
    def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        
    pvals_array = np.zeros_like(prm.timesteps)
    for i, t in tqdm(enumerate(prm.timesteps), total = len(prm.timesteps), desc = 'Computing pvals'):
        p, val = stats.wilcoxon(all_btt_cm_untuned[i], all_btt_cm_tuned[i],
                                    alternative = 'two-sided')
        '''p = permutation_test(all_btt_cm_untuned[ i],
                            all_btt_cm_tuned[i],
                            func = 'x_mean < y_mean',
                            num_rounds=100)
        pvals_array[i] = p'''
        
        res = permutation_test((all_btt_cm_untuned[i], all_btt_cm_tuned[i]), statistic, vectorized=True,
                                n_resamples=1000, alternative='less')
        
        pvals_array[i] = res.pvalue
        
    show_pvals = True
    if show_pvals :
        for i, pval in enumerate(pvals_array[:-1]) :
            if pval <0.01: # don't care about  pre-stim
                ax.axvspan(prm.timesteps[i]+prm.win_size, prm.timesteps[i+1]+prm.win_size, alpha = .3,
                        facecolor = 'gray', edgecolor = 'None',
                        zorder = -20)

    ax.hlines(.95, 0., .18, color = 'k', linewidth = 4)
    ax.axhline(1/12, c = 'gray', linestyle = '--')

    ax.legend(loc = (.025, .75), frameon = False, fontsize = 14, markerscale = .3)
    ax.set_xlim(prm.timesteps[0]+prm.win_size, prm.timesteps[-1]+prm.win_size)
    ax.set_ylim(0, .9)
    ax.set_xlabel('PST (s)', fontsize = 18)
    ax.set_ylabel('Classification accuracy', fontsize = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ax.set_xticklabels((labs*1000).astype(np.int16))
    
    yticks = np.linspace(0, 1., 6)
    ax.set_yticks(yticks)
    labs = np.round(ax.get_yticks().tolist(),1)
    ax.set_yticklabels((labs*100).astype(np.int16))
    
    ax.set_yticklabels(np.round(yticks,2))
    ax.set_ylim(0, 1.)

    fig.savefig('./figs/decoding_bthetatheta_groups_marginalization.pdf', bbox_inches='tight', dpi=200, transparent=True)

    plt.show(block = prm.block_plot)


# ----------------------------------------------------------------------------- 
# Making the coefficients maps 
# -----------------------------------------------------------------------------
def make_coeffmaps():
    # DECODING --------------------------------------------------
    print('Doing the coeffmaps')
    if not os.path.exists('./data/%s/decoding_coeffmaps.npy'% prm.postprocess_name):
        all_coeffs = np.zeros((len(prm.timesteps)), dtype = object) 
        # Fetch the main data, do logreg, and so on
        data, labels, le = np.load('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), allow_pickle = True)

        for ibin in tqdm(range(data.shape[0])) :
            logreg = LogisticRegression(**prm.opts_LR)
            cv_results = cross_validate(logreg, data[ibin,:,:],labels, scoring = 'balanced_accuracy',
                                cv=prm.n_splits, return_estimator=True, n_jobs = -1)
            '''coeffs = []
            for model in cv_results['estimator'] :
                coeffs.append(model.coef_)'''
            coeffs = [model.coef_ for model in cv_results['estimator']]
            all_coeffs[ibin] = np.mean(coeffs, axis = 0)

        np.save('./data/%s/decoding_coeffmaps.npy'% prm.postprocess_name, all_coeffs)
    else :
        all_coeffs = np.load('./data/%s/decoding_coeffmaps.npy' % prm.postprocess_name, allow_pickle = True)
       
    new_coeffs = np.zeros((len(prm.timesteps), prm.N_B_thetas* prm.N_thetas, len(prm.cluster_list))) 
    for i in range(len(prm.timesteps)) :
        new_coeffs[i,:,:] = all_coeffs[i]
    all_coeffs = new_coeffs.swapaxes(0,1)
    
    # PLOTTING --------------------------------------------------
    plt_times = [5, 20, 30, 40, 50]
    avg_mat = np.zeros((len(plt_times), prm.N_thetas, prm.N_B_thetas, len(prm.untuned_lst)))
    
    for i_col in range(len(plt_times)) :
        for idx, clust in enumerate(prm.cluster_list) :
            for i_clust, ut_clust in enumerate(prm.untuned_lst) :
                if clust == ut_clust :
                    
                    mat = utils.norm_minus(all_coeffs[:,:,idx])
                    
                    start = plt_times[i_col]
                    stop = start + 10
                    coeffs = mat[:, start:stop].mean(axis=1).reshape((prm.N_B_thetas, prm.N_thetas))
                    
                    roll_n = -int(np.mean(np.argmax(coeffs, axis = 1)))
                    rolled_coeffs = np.roll(coeffs.T, roll_n, axis = 0)
                    avg_mat[i_col, :, :, i_clust] += rolled_coeffs
                    
                    
    fig, axs = plt.subplots(ncols = len(plt_times), nrows = 1,
                        figsize = (12,3), subplot_kw = dict(projection = 'polar', aspect = 'equal'))

    for i_col in range(len(plt_times)) :
        ax = axs[i_col]
        coeffs_to_plot = avg_mat[i_col,:,:,:].mean(axis = -1)
        print(coeffs_to_plot.min(), coeffs_to_plot.max())
        pc = ax.pcolormesh(prm.AZ*2+np.pi/2-np.pi/prm.N_thetas, prm.EL, coeffs_to_plot,
                    cmap="coolwarm", edgecolors = 'none', linewidth = 0.,
                    antialiased = True, norm = mcols.TwoSlopeNorm(
                    vmin = -.1 ,
                    vmax = .1, vcenter = 0))
        
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticklabels([])
        
    fig.tight_layout()
    fig.savefig('./figs/decoding_coeffs_vul.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    avg_mat = np.zeros((len(plt_times), prm.N_thetas, prm.N_B_thetas, len(prm.tuned_lst)))

    for i_col in range(len(plt_times)) :
        for idx, clust in enumerate(prm.cluster_list) :
            for i_clust, t_clust in enumerate(prm.tuned_lst) :
                if clust == t_clust :
                    
                    mat = utils.norm_minus(all_coeffs[:,:,idx])
                    
                    start = plt_times[i_col]
                    stop = start + 10
                    coeffs = mat[:, start:stop].mean(axis=1).reshape((prm.N_B_thetas, prm.N_thetas))
                    
                    roll_n = -int(np.mean(np.argmax(coeffs, axis = 1)))
                    rolled_coeffs = np.roll(coeffs.T, roll_n, axis = 0)
                    avg_mat[i_col, :, :, i_clust] += rolled_coeffs
                    
    fig, axs = plt.subplots(ncols = len(plt_times), nrows = 1,
                        figsize = (12,3), subplot_kw = dict(projection = 'polar', aspect = 'equal'))

    for i_col in range(len(plt_times)) :
        ax = axs[i_col]
        coeffs_to_plot = avg_mat[i_col,:,:,:].mean(axis = -1)
        print(coeffs_to_plot.min(), coeffs_to_plot.max())
        pc = ax.pcolormesh(prm.AZ*2+np.pi/2-np.pi/prm.N_thetas, prm.EL, coeffs_to_plot,
                    cmap="coolwarm", edgecolors = 'none', linewidth = 0.,
                    antialiased = True, norm = mcols.TwoSlopeNorm(
                    vmin = -.1 ,
                    vmax = .1, vcenter = 0))
        
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticklabels([])
        
    fig.tight_layout()
    fig.savefig('./figs/decoding_coeffs_res.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)