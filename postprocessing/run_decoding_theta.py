#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
The first file to call, which should run the decoding of orientation from all the clusters
"""

import params as prm
import utils 
import utils_single_neuron as sn_utils 
import utils_decoding as dec_utils
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
import os 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from scipy import stats
from mlxtend.evaluate import permutation_test

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --------------------------------------------------------------
# Decoding of orientation for all neurons
# --------------------------------------------------------------
def make_theta_decoding_all() :    
    print('Doing the decoding for theta (orientation)')
    
    # DECODING  ------------------------------------
    print('Doing the decoding on all the neurons with K-fold = %s' % prm.n_splits)
    if not os.path.exists('./data/%s/decoding_theta_all_kfold.npy' % prm.postprocess_name):
        kfold_scores = np.zeros((len(prm.B_thetas), len(prm.timesteps), prm.n_splits)) 
        for ibt, bt in enumerate(prm.B_thetas) :
            print('Running for btheta = %s' % (bt*180/np.pi))
            # Data
            if not os.path.exists('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt)):
                data, labels, le = dec_utils.par_load_data(timesteps = prm.timesteps, target_clusters = prm.cluster_list,
                                                        target_btheta = bt, target_theta = None, data_type = 'one_bt',
                                                        disable_tqdm = False
                                                        )
                np.save('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt), [data, labels, le])
            else : 
                data, labels, le = np.load('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt), allow_pickle = True)

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)

            for ibin in tqdm(range(data.shape[0]), desc = 'Decoding') :
                scores = cross_val_score(logreg, data[ibin,:,:], labels, 
                                        cv = prm.n_splits, 
                                        scoring = 'balanced_accuracy')
                kfold_scores[ibt, ibin, :] = scores

        np.save('./data/%s/decoding_theta_all_kfold.npy'% prm.postprocess_name, kfold_scores)
    else :
        kfold_scores = np.load('./data/%s/decoding_theta_all_kfold.npy'% prm.postprocess_name)
    
    
    # PLOTTING  ------------------------------------
    fig, ax = plt.subplots(figsize = (9,6))
    plot_bthetas = [7,  0]
    for ibt in plot_bthetas :
        kfold_means = np.asarray([x.mean() for x in kfold_scores[ibt]])
        kfold_stderr = np.asarray([x.std() for x in kfold_scores[ibt]])
        
        ax.plot(prm.timesteps + prm.win_size, kfold_means, color = prm.colors[ibt])
        ax.fill_between(prm.timesteps + prm.win_size,
                    kfold_means + kfold_stderr, kfold_means - kfold_stderr,
                    facecolor = prm.colors[ibt], edgecolor = None, alpha = .7,
                    label = r'B$_\theta$ = %.1f°' % (prm.B_thetas[ibt] * 180/np.pi))
        mod_t = prm.timesteps + prm.win_size
        print('Btheta %.2f - max at time %.2fs = %.2f' % (prm.B_thetas[ibt], mod_t[np.argmax(kfold_means)],
                                                np.max(kfold_means)))
        
    ax.hlines(.95, 0., .3, color = 'k', linewidth = 4)
    ax.axhline(1/12, c = 'gray', linestyle = '--')

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
    ax.legend(loc = (.025, .65), frameon = False, fontsize = 14, markerscale = 2, ncol = 1)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(prm.colors[7])
    leg.legendHandles[0].set_alpha(1)
    leg.legendHandles[1].set_color(prm.colors[0])
    leg.legendHandles[1].set_alpha(1)
    
    fig.tight_layout()
    fig.savefig('./figs/decoding_theta_all_timesteps.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    
    # FITTING ------------------------------------
    all_params, all_means = [], []
    for ibt,_ in enumerate(prm.B_thetas):
        kfold_means = np.asarray([x.mean() for x in kfold_scores[ibt]])
        all_means.append(kfold_means)
        fit_params, r2 = dec_utils.fit_accum(kfold_means)
        all_params.append(fit_params)
    
    params, titles = dec_utils.fit_all(all_params, all_means)
    fig, axs = plt.subplots(figsize = (9,4), ncols = 2)
    for i, key in enumerate(['L', 't0']):
        ax = axs[i]
        param_list = params[key]

        ax.plot(prm.B_thetas*180/np.pi, param_list, c = 'k', zorder = 5, lw = 2)
        ax.scatter(prm.B_thetas*180/np.pi, param_list, c = prm.colors[np.arange(0,8,1)], zorder = 10, s = 75)

        if key == 'L' :
            ax.set_xlabel(r'$B_{\theta}$ (°)', fontsize = 16)
            ax.set_ylim(0., 1.)
            ax.set_yticks([0,.5,1.])
            ax.axhline(1/12, c = 'gray', linestyle = '--')

        if key == 't0' or key == 't1' :
            curr_ticks = [17, 27, 37]
            ax.set_yticks(curr_ticks)
            time_values = np.round((prm.timesteps+prm.win_size)[curr_ticks],2) 
            ax.set_yticklabels(time_values*1000)
            ax.set_ylim(17, 37)

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticks([0, 18, 36])
        ax.set_xticklabels(['0', '18', '36'])
        ax.set_ylabel(titles[key], fontsize = 16)        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig('./figs/decoding_theta_all_params.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)



# --------------------------------------------------------------
# Making population TC for all neurons
# this is a separate instance because we report thetas x thetas 
# --------------------------------------------------------------
def make_theta_population_tc_all():
    print('Doing the population TC for theta (orientation)')
    
    # DECODING --------------------------------------------------
    if not os.path.exists('./data/%s/decoding_theta_all_population_tc.npy' % prm.postprocess_name):
        population_likelihoods = np.zeros((len(prm.B_thetas), len(prm.timesteps), prm.N_thetas, prm.N_thetas)) 
        for ibt, bt in enumerate(prm.B_thetas) :
            print('Running for btheta = %s' % (bt*180/np.pi))
            # Data
            if not os.path.exists('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt)):
                data, labels, le = dec_utils.par_load_data(timesteps = prm.timesteps, target_clusters = prm.cluster_list,
                                                        target_btheta = bt, target_theta = None, data_type = 'one_bt',
                                                        disable_tqdm = False
                                                        )
                np.save('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt), [data, labels, le])
            else : 
                data, labels, le = np.load('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt), allow_pickle = True)

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)

            for ibin in tqdm(range(data.shape[0]), desc = 'Decoding') :
                xtrain, xtest, ytrain, ytest = train_test_split(data[ibin,:,:], labels, test_size=prm.test_size, random_state=42)
                logreg.fit(xtrain, ytrain)
            
                proba = logreg.predict_proba(xtest)

                for i_test in np.unique(ytest):
                    population_likelihoods[ibt,ibin, i_test, :] = proba[ytest==i_test, :].mean(axis=0)

        np.save('./data/%s/decoding_theta_all_population_tc.npy'% prm.postprocess_name, population_likelihoods)
    else :
        population_likelihoods = np.load('./data/%s/decoding_theta_all_population_tc.npy'% prm.postprocess_name, allow_pickle = True)
      
        
    # PLOTTING --------------------------------------------------
    fig, axs = plt.subplots(figsize = (15, 4), nrows = 1, ncols = 4)
    new_cols = prm.colors[[7, 0]]

    for ibt, bt in enumerate([7, 0]) :
        for it, t in enumerate([20, 30, 40, 50]) :
            ax = axs[it]
            
            ll = population_likelihoods[bt, t, :, :]
            local_ll = []
            for itheta in range(prm.N_thetas):
                local_ll.append(np.roll(ll[itheta,:], shift = -itheta+6, axis = -1))
                
            raw_ll = np.mean(local_ll, axis = 0)
            fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
            fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    **fitted_ll_params)
            ax.plot(np.linspace(-np.pi, np.pi, 100), fitted_ll,
                    color = new_cols[ibt], lw = 2)
            
            ax.set_xticks([-3.14, 0, 3.14])
            ax.set_yticks([0, .5, 1.])
            if ibt == 2 and it == 0:
                ax.set_xlabel(r'$\theta$ true (°)', fontsize = 18)
                ax.set_ylabel('likelihood correct', fontsize = 18)
                ax.set_xticklabels([-90, 0, 90])
                ax.tick_params(axis='both', which='major', labelsize=14)
            elif it > 0  :
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                
    fig.tight_layout()
    fig.savefig('./figs/decoding_theta_all_population_tc.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    
    # FITTING --------------------------------------------------
    print('\nDoing the population TC linear regression')
    all_kfold_scores = np.load('./data/%s/decoding_theta_all_kfold.npy' % prm.postprocess_name)

    all_params = []
    for ibt, bt in enumerate([7, 4, 0]) :
        one_bt_params = []
        for it, t in enumerate(prm.timesteps) :
            
            ll = population_likelihoods[bt,it, :, :]
            local_ll = []
            for itheta in range(prm.N_thetas):
                local_ll.append(np.roll(ll[itheta,:], shift = -itheta+6, axis = -1))
                
            raw_ll = np.mean(local_ll, axis = 0)
            fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
            one_bt_params.append((utils.cirvar(raw_ll),fitted_ll_params))
            
        all_params.append(one_bt_params)
    
    fig, ax = plt.subplots(figsize = (5, 5))
    for ibt, bt in enumerate([7,0]) :
        cvs = 1-np.asarray([x[0] for x in all_params[ibt]])[:50]
        accs = np.asarray([x.mean() for x in all_kfold_scores[bt]])[:50]
        ax.scatter(cvs, accs, color = prm.colors[bt], s = 20)
        
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(cvs, accs)
        ax.plot(np.linspace(0, 1, 100),
                intercept + slope * np.linspace(0,1,100), c = prm.colors[bt])
        r, pval = stats.spearmanr(cvs, accs)
        print(prm.B_thetas[bt])
        print('Slope = %.3f ; Intercept = %.3f ; pval = %s' % (slope, intercept, pvalue))
        print('Spearman R = %.3f ; pval = %s'  % (r, pval))
        
        ax.plot((0,1), (0,1), c = 'gray', linestyle = '--', alpha = .75, 
                zorder = -1)
            
        ax.set_xlabel('1 - Population circular variance', fontsize = 18)
        ax.set_ylabel('Classification accuracy', fontsize = 18)
        ax.set_xticks([0, .5, 1.])
        ax.set_yticks([0, .5, 1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    fig.tight_layout()
    fig.savefig('./figs/decoding_theta_all_population_linregress.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

# --------------------------------------------------------------
# Decoding of orientation for the two groups
# --------------------------------------------------------------
def make_theta_decoding_groups():
    
    # DECODING --------------------------------------------------
    print('Doing the decoding on the two groups with boostraps')
    if not os.path.exists('./data/%s/decoding_theta_groups_bootstrap.npy'% prm.postprocess_name):
        bootstrapped_results = np.zeros((2, prm.n_bootstrap, len(prm.B_thetas), len(prm.timesteps))) # we only report the mean per kfold
        for ibootstrap in tqdm(range(prm.n_bootstrap), desc = 'Bootstrapping') :
            # Randomly pick subgroups in each population
            np.random.seed(prm.seed+ibootstrap)
            picked_res = np.random.choice(prm.tuned_lst, size = prm.n_subgroups)
            picked_vul = np.random.choice(prm.untuned_lst, size = prm.n_subgroups)
            idxs_res = [list(prm.cluster_list).index(x) for x in picked_res]
            idxs_vul = [list(prm.cluster_list).index(x) for x in picked_vul]
            
            # Fetch the main data, do logreg, and so on
            for ibt, bt in enumerate(prm.B_thetas) :
                data, labels, le = np.load('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt), allow_pickle = True)
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
                    bootstrapped_results[0, ibootstrap, ibt, ibin] = np.mean(scores_res)
                    bootstrapped_results[1, ibootstrap, ibt, ibin] = np.mean(scores_vul)

        np.save('./data/%s/decoding_theta_groups_bootstrap.npy'% prm.postprocess_name, bootstrapped_results)
    else :
        bootstrapped_results = np.load('./data/%s/decoding_theta_groups_bootstrap.npy' % prm.postprocess_name)
    
    # PLOTTING --------------------------------------------------------
    plot_bthetas = [7,  0]
    for ibt in plot_bthetas :
        fig, ax = plt.subplots(figsize = (8,5))
        
        # Untuned bootstrapping
        bootstrap_mean = bootstrapped_results[1,:,ibt,:].mean(axis = 0)
        bootstrap_std = bootstrapped_results[1,:,ibt,:].std(axis = 0)
        
        ax.plot(prm.timesteps + prm.win_size, bootstrap_mean, color = prm.col_untuned)
        ax.fill_between(prm.timesteps + prm.win_size,
                    bootstrap_mean + bootstrap_std, bootstrap_mean - bootstrap_std,
                    facecolor = prm.col_untuned, edgecolor = None, alpha = .7,
                    label = prm.name_untuned)
        
        # Tuned bootstrapping
        bootstrap_mean = bootstrapped_results[0,:,ibt,:].mean(axis = 0)
        bootstrap_std = bootstrapped_results[0,:,ibt,:].std(axis = 0)
        
        ax.plot(prm.timesteps + prm.win_size, bootstrap_mean, color = prm.col_tuned)
        ax.fill_between(prm.timesteps + prm.win_size,
                    bootstrap_mean + bootstrap_std, bootstrap_mean - bootstrap_std,
                    facecolor = prm.col_tuned, edgecolor = None, alpha = .7,
                    label = prm.name_tuned)
        
        # Statistical tests
        pvals_array = np.zeros_like(prm.timesteps)
        for i, t in tqdm(enumerate(prm.timesteps), total = len(prm.timesteps), desc = 'Computing pvals'):
            p, val = stats.wilcoxon(bootstrapped_results[0,:,ibt, i], bootstrapped_results[1,:,ibt, i],
                                    alternative = 'two-sided')
            '''p = permutation_test(bootstrapped_results[0,:,ibt, i],
                                bootstrapped_results[1,:,ibt, i],
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
        ax.axhline(1/12, c = 'gray', linestyle = '--')
        
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
        fig.savefig('./figs/decoding_theta_groups_timesteps_bt%s.pdf' % ibt, bbox_inches='tight', dpi=200, transparent=True)
        plt.show(block = prm.block_plot)
        

    # FITTING ------------------------------------
    fig, axs = plt.subplots(figsize = (9,4), ncols = 2)
    for ntype in [0,1] :
        all_params, all_means = [], []
        for ibt,_ in enumerate(prm.B_thetas):
            bootstrap_mean = bootstrapped_results[ntype,:,ibt,:].mean(axis = 0)
            all_means.append(bootstrap_mean)
            fit_params, r2 = dec_utils.fit_accum(bootstrap_mean)
            all_params.append(fit_params)
        
        params, titles = dec_utils.fit_all(all_params, all_means)
        for i, key in enumerate(['L', 't0']):
            ax = axs[i]
            param_list = params[key]
            
            ax.plot(prm.B_thetas*180/np.pi, param_list, c = prm.col_tuned if ntype == 0 else prm.col_untuned, zorder = 5, lw = 2)
            ax.scatter(prm.B_thetas*180/np.pi, param_list, c = prm.colors[np.arange(0,8,1)], zorder = 10, s = 75)
            
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xticks([0, 18, 36])
            ax.set_xticklabels(['0', '18', '36'])
            yticks = np.linspace(np.min(param_list), np.max(param_list), 3)
            ax.set_yticks(np.round(yticks, 2))
            
            if i == 0 :
                ax.set_xlabel(r'$B_{\theta}$ (°)', fontsize = 16)
                ax.set_ylim(0., 1.)
                yticks = np.linspace(0., 1., 4)
                ax.set_yticks(np.round(yticks, 1))
                ax.axhline(1/12, c = 'gray', linestyle = '--')
            ax.set_ylabel(titles[key], fontsize = 16)
            
            if key == 't0' or key == 't1' :
                curr_ticks = [17, 27, 37]
                ax.set_yticks(curr_ticks)
                time_values = np.round((prm.timesteps+prm.win_size)[curr_ticks],2) 
                ax.set_yticklabels(time_values*1000)
                        
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
    axs[0].set_ylim(0, 1.)
    axs[0].set_yticks([0,.5,1.])
    axs[1].set_ylim(17, 37)

    fig.tight_layout()
    fig.savefig('./figs/decoding_theta_group_params.pdf', bbox_inches='tight', dpi=200, transparent=True)
 
 
    
# --------------------------------------------------------------
# Making population TC for the two groups 
# --------------------------------------------------------------
def make_theta_population_tc_groups():

    # DECODING ------------------------------------
    print('Doing population tuning curve on the two groups with boostraps')
    if not os.path.exists('./data/%s/decoding_theta_groups_population_tc.npy'% prm.postprocess_name):
        bootstrapped_likelihoods= np.zeros((2, prm.n_bootstrap, len(prm.B_thetas), len(prm.timesteps),
                                        prm.N_thetas, prm.N_thetas)) # we only report the mean per kfold
        for ibootstrap in tqdm(range(prm.n_bootstrap), desc = 'Bootstrapping'):
            # Randomly pick subgroups in each population
            np.random.seed(prm.seed+ibootstrap)
            picked_res = np.random.choice(prm.tuned_lst, size = prm.n_subgroups)
            picked_vul = np.random.choice(prm.untuned_lst, size = prm.n_subgroups)
            idxs_res = [list(prm.cluster_list).index(x) for x in picked_res]
            idxs_vul = [list(prm.cluster_list).index(x) for x in picked_vul]
            
            # Fetch the main data, do logreg, and so on
            for ibt, bt in enumerate(prm.B_thetas) :
                data, labels, le = np.load('./data/%s/decoding_theta_bt%s_data.npy' % (prm.postprocess_name, ibt), allow_pickle = True)
                data_res = data[:, :, idxs_res]
                data_vul = data[:, :, idxs_vul]

                # Classifying
                logreg = LogisticRegression(**prm.opts_LR)
                
                for ibin in range(data.shape[0]) :
                    xtrain, xtest, ytrain, ytest = train_test_split(data_res[ibin,:,:], labels, test_size=prm.test_size, random_state=42)
                    logreg.fit(xtrain, ytrain)
                    proba = logreg.predict_proba(xtest)
                    for i_test in np.unique(ytest):
                        bootstrapped_likelihoods[0, ibootstrap, ibt, ibin, i_test, :] = proba[ytest==i_test, :].mean(axis=0)
                        
                    xtrain, xtest, ytrain, ytest = train_test_split(data_vul[ibin,:,:], labels, test_size=prm.test_size, random_state=42)
                    logreg.fit(xtrain, ytrain)
                    proba = logreg.predict_proba(xtest)
                    for i_test in np.unique(ytest):
                        bootstrapped_likelihoods[1, ibootstrap, ibt, ibin, i_test, :] = proba[ytest==i_test, :].mean(axis=0)

        np.save('./data/%s/decoding_theta_groups_population_tc.npy'% prm.postprocess_name, bootstrapped_likelihoods)
    else :
        bootstrapped_likelihoods = np.load('./data/%s/decoding_theta_groups_population_tc.npy' % prm.postprocess_name)
        
    # PLOTTING --------------------------------------------------
    population_likelihood_res = bootstrapped_likelihoods[0, :, :, :, :, :].mean(axis = 0)
    population_likelihood_vul = bootstrapped_likelihoods[1, :, :, :, :, :].mean(axis = 0)
    
    fig, axs = plt.subplots(figsize = (4.5*3, 2*3), nrows = 2, ncols = 4)
    for ibt, bt in enumerate([7, 0]) :
        for it, t in enumerate([20, 30, 40, 50]) :
            
            ax = axs[ibt,it]
            
            ll = population_likelihood_res[bt, t, :, :]
            local_ll = []
            for itheta in range(prm.N_thetas):
                local_ll.append(np.roll(ll[itheta,:], shift = -itheta+6, axis = -1))
            raw_ll = np.mean(local_ll, axis = 0)
            fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
            fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    **fitted_ll_params)
            ax.plot(np.linspace(-np.pi, np.pi, 100), fitted_ll,
                    color = prm.col_tuned, lw = 2)
            
            ll = population_likelihood_vul[bt, t, :, :]
            local_ll = []
            for itheta in range(prm.N_thetas):
                local_ll.append(np.roll(ll[itheta,:], shift = -itheta+6, axis = -1))
            raw_ll = np.mean(local_ll, axis = 0)
            fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
            fitted_ll = utils.tuning_function(x = np.linspace(-np.pi, np.pi, 100),
                                    **fitted_ll_params)
            ax.plot(np.linspace(-np.pi, np.pi, 100), fitted_ll,
                    color = prm.col_untuned, lw = 2)
            
            ax.set_xticks([-3.14, 0, 3.14])
            ax.set_yticks([0, .5, 1.])
            if ibt == 1 and it == 0:
                ax.set_xticklabels([-90, 0, 90])
                ax.set_xlabel(r'$\theta$ true (°)', fontsize = 18)
                ax.set_ylabel('likelihood correct', fontsize = 18)
                ax.tick_params(axis='both', which='major', labelsize=14)
            else  :
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                
    fig.tight_layout()
    fig.savefig('./figs/decoding_theta_groups_population_tc.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    # FITTING --------------------------------------------------
    print('\nDoing the population TC linear regression')
    tuned_params, untuned_params = [], []
    for ibt, bt in enumerate([7, 0]) :
        one_bt_params_untuned, one_bt_params_tuned = [], []
        for it, t in enumerate(prm.timesteps) :
            
            ll = population_likelihood_vul[bt,it, :, :]
            local_ll = []
            for itheta in range(prm.N_thetas):
                local_ll.append(np.roll(ll[itheta,:], shift = -itheta+6, axis = -1))
                
            raw_ll = np.mean(local_ll, axis = 0)
            fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
            one_bt_params_untuned.append((utils.cirvar(raw_ll),fitted_ll_params))
            
            ll = population_likelihood_res[bt,it, :, :]
            local_ll = []
            for itheta in range(prm.N_thetas):
                local_ll.append(np.roll(ll[itheta,:], shift = -itheta+6, axis = -1))
                
            raw_ll = np.mean(local_ll, axis = 0)
            fitted_ll_params = utils.fit_tc(raw_ll, init_kappa = 1)
            one_bt_params_tuned.append((utils.cirvar(raw_ll),fitted_ll_params))
            
        tuned_params.append(one_bt_params_tuned)
        untuned_params.append(one_bt_params_untuned)
    
    bootstrapped_results = np.load('./data/%s/decoding_theta_groups_bootstrap.npy' % prm.postprocess_name)
    tuned_kfold_scores = bootstrapped_results[0, :, :, :].mean(axis = 0)
    untuned_kfold_scores = bootstrapped_results[1, :, :, :].mean(axis = 0)
    for ibt, bt in enumerate([7,0]) :
        print('bt %s' % ibt)
        
        fig, ax = plt.subplots(figsize = (5, 5))
        cvs = 1-np.asarray([x[0] for x in tuned_params[ibt]])[:50]
        accs = np.asarray([x.mean() for x in tuned_kfold_scores[bt]])[:50]
        ax.scatter(cvs, accs, color = prm.col_tuned, s = 20)    
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(cvs, accs)
        ax.plot(np.linspace(0, 1, 100),
                intercept + slope * np.linspace(0,1,100), c = prm.col_tuned)
        r, pval = stats.spearmanr(cvs, accs)
        print('Resilient')
        print('Slope = %.3f ; Intercept = %.3f ; pval = %s' % (slope, intercept, pvalue))
        print('Spearman R = %.3f ; pval = %s'  % (r, pval))
            
        cvs = 1-np.asarray([x[0] for x in untuned_params[ibt]])[:50]
        accs = np.asarray([x.mean() for x in untuned_kfold_scores[bt]])[:50]
        ax.scatter(cvs, accs, color = prm.col_untuned, s = 20)
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(cvs, accs)
        ax.plot(np.linspace(0, 1, 100),
                intercept + slope * np.linspace(0,1,100), c = prm.col_untuned)
        r, pval = stats.spearmanr(cvs, accs)
        print('-\nVulnerable')
        print('Slope = %.3f ; Intercept = %.3f ; pval = %s' % (slope, intercept, pvalue))
        print('Spearman R = %.3f ; pval = %s'  % (r, pval))       
        print('------')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([0, .5, 1.])
        ax.set_yticks([0, .5, 1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.plot((0,1), (0,1), c = 'gray', linestyle = '--', alpha = .75, 
                zorder = -1)
        
        fig.tight_layout()
        fig.savefig('./figs/decoding_theta_groups_population_linregress_bt%s.pdf'%bt, bbox_inches='tight', dpi=200, transparent=True)
        plt.show(block = prm.block_plot)