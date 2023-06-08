#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
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
from mlxtend.evaluate import permutation_test

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --------------------------------------------------------------
# Decoding theta with continuous paradigm
# --------------------------------------------------------------
def make_continuous_theta_decoding():
    # DECODING --------------------------------------------------
    print('Doing the continuous decoding on theta, for BT = 0°')
    continuous_list = np.load('./data/%s/continuous_clusters_list.npy' % prm.postprocess_name, allow_pickle = True)
    neurons_score = np.load('./data/%s/resilience_score.npy' % prm.postprocess_name, allow_pickle = True)
    paired_score = sorted(list(zip(continuous_list, neurons_score)), key = lambda x: x[1])
    
    if not os.path.exists('./data/%s/continuous_decoding_theta_bt0.npy'% prm.postprocess_name):
        continuous_decoding = np.zeros((prm.total_loops, 2)) # bootstrap, (max acc, res/vul score)
        for ibootstrap in tqdm(range(prm.total_loops), desc = 'Bootstrapping continuous theta') :
            group_score = [x[1] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]
            idxs_neurons = [x[0] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]

            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_theta_bt0_data.npy' % (prm.postprocess_name), allow_pickle = True)
            data_selected = data[:, :, idxs_neurons]

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)
            
            tmp_scores = []
            for ibin in range(15,data.shape[0]) :
                scores = cross_val_score(logreg, data_selected[ibin,:,:], labels, 
                                            cv = prm.n_splits, 
                                            scoring = 'balanced_accuracy')
                tmp_scores.append(np.mean(scores))
            continuous_decoding[ibootstrap, 0] = np.max(tmp_scores)
            continuous_decoding[ibootstrap, 1] = np.mean(group_score)

        np.save('./data/%s/continuous_decoding_theta_bt0.npy'% prm.postprocess_name, continuous_decoding)
    else :
        continuous_decoding = np.load('./data/%s/continuous_decoding_theta_bt0.npy' % prm.postprocess_name)
    
    # PLOTTING --------------------------------------------------------
    fig, ax = plt.subplots(figsize = (8,5))
    
    ax.scatter(continuous_decoding[:,1], continuous_decoding[:,0], color = prm.colors[0], s = 100)
    
    cor, pval = stats.spearmanr(continuous_decoding[:,1], continuous_decoding[:,0])
    print('Spearman R = %.3f' % cor)
    print('Spearman p-value = %.3f' % pval)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(continuous_decoding[:,1], continuous_decoding[:,0])
    print('LinReg p-value = %.3f' % pvalue)
    print('LinReg r^2 = %.3f'% rvalue)
    print('Linreg slope, intercept ', slope, intercept)
    xplots = np.linspace(np.min(continuous_decoding[:,1]),
                        np.max(continuous_decoding[:,1]),
                        100)
    ax.plot(xplots, intercept + slope * xplots, c = prm.colors[0], lw = 2, ls = '--')

    ax.set_xlabel('Res/vul score', fontsize = 18)
    ax.set_ylabel('Max classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ylim_min, ylim_max = 0.305*.8, 0.305*1.2
    ax.set_ylim(ylim_min, ylim_max)
    yticks = np.linspace(ylim_min, ylim_max, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,2))
    ax.set_xlim(0.15, 0.5)
    ax.set_xticks(np.linspace(0.15, 0.5, 4, endpoint = True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = (.025, .75), frameon = False, fontsize = 14, markerscale = .2)
    
    fig.tight_layout()
    fig.savefig('./figs/continuous_decoding_theta_bt0.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

    # DECODING --------------------------------------------------
    print('Doing the continuous decoding on theta, for BT = 36°')
    continuous_list = np.load('./data/%s/continuous_clusters_list.npy' % prm.postprocess_name, allow_pickle = True)
    neurons_score = np.load('./data/%s/resilience_score.npy' % prm.postprocess_name, allow_pickle = True)
    paired_score = sorted(list(zip(continuous_list, neurons_score)), key = lambda x: x[1])
    if not os.path.exists('./data/%s/continuous_decoding_theta_bt7.npy'% prm.postprocess_name):
        continuous_decoding = np.zeros((prm.total_loops, 2)) # bootstrap, (max acc, res/vul score)
        for ibootstrap in tqdm(range(prm.total_loops), desc = 'Bootstrapping continuous theta') :
            group_score = [x[1] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]
            idxs_neurons = [x[0] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]

            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_theta_bt7_data.npy' % (prm.postprocess_name), allow_pickle = True)
            data_selected = data[:, :, idxs_neurons]

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)
            
            tmp_scores = []
            for ibin in range(15,data.shape[0]) :
                scores = cross_val_score(logreg, data_selected[ibin,:,:], labels, 
                                            cv = prm.n_splits, 
                                            scoring = 'balanced_accuracy')
                tmp_scores.append(np.mean(scores))
            continuous_decoding[ibootstrap, 0] = np.max(tmp_scores)
            continuous_decoding[ibootstrap, 1] = np.mean(group_score)

        np.save('./data/%s/continuous_decoding_theta_bt7.npy'% prm.postprocess_name, continuous_decoding)
    else :
        continuous_decoding = np.load('./data/%s/continuous_decoding_theta_bt7.npy' % prm.postprocess_name)
    
    # PLOTTING --------------------------------------------------------
    fig, ax = plt.subplots(figsize = (8,5))
    
    ax.scatter(continuous_decoding[:,1], continuous_decoding[:,0], color = prm.colors[-1], s = 100)
    
    cor, pval = stats.spearmanr(continuous_decoding[:,1], continuous_decoding[:,0])
    print('Spearman R = %.3f' % cor)
    print('Spearman p-value = %.3f' % pval)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(continuous_decoding[:,1], continuous_decoding[:,0])
    print('LinReg p-value = %.3f' % pvalue)
    print('LinReg r^2 = %.3f'% rvalue)
    print('Linreg slope, intercept ', slope, intercept)
    
    xplots = np.linspace(np.min(continuous_decoding[:,1]),
                        np.max(continuous_decoding[:,1]),
                        100)
    ax.plot(xplots, intercept + slope * xplots, c = prm.colors[-1], lw = 2, ls = '--')

    ax.set_xlabel('Res/vul score', fontsize = 18)
    ax.set_ylabel('Max classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ylim_min, ylim_max = 0.7*.8 , .7*1.2
    ax.set_ylim(ylim_min, ylim_max)
    yticks = np.linspace(ylim_min, ylim_max, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,2))
    
    ax.set_xlim(0.15, 0.5)
    ax.set_xticks(np.linspace(0.15, 0.5, 4, endpoint = True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = (.025, .75), frameon = False, fontsize = 14, markerscale = .2)
    
    fig.tight_layout()
    fig.savefig('./figs/continuous_decoding_theta_bt7.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    
# --------------------------------------------------------------
# Decoding btheta with continuous paradigm
# --------------------------------------------------------------
def make_continuous_btheta_decoding():
    # DECODING --------------------------------------------------
    print('Doing the continuous decoding on btheta')
    continuous_list = np.load('./data/%s/continuous_clusters_list.npy' % prm.postprocess_name, allow_pickle = True)
    neurons_score = np.load('./data/%s/resilience_score.npy' % prm.postprocess_name, allow_pickle = True)
    paired_score = sorted(list(zip(continuous_list, neurons_score)), key = lambda x: x[1])
    if not os.path.exists('./data/%s/continuous_decoding_btheta.npy'% prm.postprocess_name):
        continuous_decoding = np.zeros((prm.total_loops, 2)) # bootstrap, (max acc, res/vul score)
        for ibootstrap in tqdm(range(prm.total_loops), desc = 'Bootstrapping continuous btheta') : #technically it's not a bootstrap
            group_score = [x[1] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]
            idxs_neurons = [x[0] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]

            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_btheta_data.npy' % (prm.postprocess_name), allow_pickle = True)
            data_selected = data[:, :, idxs_neurons]

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)
            
            tmp_scores = []
            for ibin in range(15,data.shape[0]) :
                scores = cross_val_score(logreg, data_selected[ibin,:,:], labels, 
                                            cv = prm.n_splits, 
                                            scoring = 'balanced_accuracy')
                tmp_scores.append(np.mean(scores))
            continuous_decoding[ibootstrap, 0] = np.max(tmp_scores)
            continuous_decoding[ibootstrap, 1] = np.mean(group_score)

        np.save('./data/%s/continuous_decoding_btheta.npy'% prm.postprocess_name, continuous_decoding)
    else :
        continuous_decoding = np.load('./data/%s/continuous_decoding_btheta.npy' % prm.postprocess_name)
    
    # PLOTTING --------------------------------------------------------
    fig, ax = plt.subplots(figsize = (8,5))
    
    ax.scatter(continuous_decoding[:,1], continuous_decoding[:,0], color = prm.col_bt, s = 100)
    
    cor, pval = stats.spearmanr(continuous_decoding[:,1], continuous_decoding[:,0])
    print('Spearman R = %.3f' % cor)
    print('Spearman p-value = %.3f' % pval)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(continuous_decoding[:,1], continuous_decoding[:,0])
    print('LinReg p-value = %.3f' % pvalue)
    print('LinReg r^2 = %.3f'% rvalue)
    print('Linreg slope, intercept ', slope, intercept)
    xplots = np.linspace(np.min(continuous_decoding[:,1]),
                        np.max(continuous_decoding[:,1]),
                        100)
    ax.plot(xplots, intercept + slope * xplots, c = prm.col_bt, lw = 2, ls = '--')

    ax.set_xlabel('Res/vul score', fontsize = 18)
    ax.set_ylabel('Max classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ylim_min, ylim_max = 0.22*.8 , 0.22 * 1.2
    yticks = np.linspace(ylim_min, ylim_max, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,2))
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlim(0.15, 0.5)
    ax.set_xticks(np.linspace(0.15, 0.5, 4, endpoint = True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.legend(loc = (.025, .75), frameon = False, fontsize = 14, markerscale = .2)
    
    fig.tight_layout()
    fig.savefig('./figs/continuous_decoding_btheta.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    
# --------------------------------------------------------------
# Decoding btheta with continuous paradigm
# --------------------------------------------------------------
def make_continuous_btheta_theta_decoding():
    # DECODING --------------------------------------------------
    print('Doing the continuous decoding on btheta x theta')
    continuous_list = np.load('./data/%s/continuous_clusters_list.npy' % prm.postprocess_name, allow_pickle = True)
    neurons_score = np.load('./data/%s/resilience_score.npy' % prm.postprocess_name, allow_pickle = True)
    paired_score = sorted(list(zip(continuous_list, neurons_score)), key = lambda x: x[1])
    if not os.path.exists('./data/%s/continuous_decoding_btheta_theta.npy'% prm.postprocess_name):
        continuous_decoding = np.zeros((prm.total_loops, 2)) # bootstrap, (max acc, res/vul score)
        for ibootstrap in tqdm(range(prm.total_loops), desc = 'Bootstrapping continuous btheta') :
            group_score = [x[1] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]
            idxs_neurons = [x[0] for x in paired_score][ibootstrap * prm.n_continuous : prm.n_subgroups + (ibootstrap*prm.n_continuous)]

            # Fetch the main data, do logreg, and so on
            data, labels, le = np.load('./data/%s/decoding_bthetatheta_data.npy' % (prm.postprocess_name), allow_pickle = True)
            data_selected = data[:, :, idxs_neurons]

            # Classifying
            logreg = LogisticRegression(**prm.opts_LR)
            
            tmp_scores = []
            for ibin in range(15,data.shape[0]) :
                scores = cross_val_score(logreg, data_selected[ibin,:,:], labels, 
                                            cv = prm.n_splits, 
                                            scoring = 'balanced_accuracy')
                tmp_scores.append(np.mean(scores))
            continuous_decoding[ibootstrap, 0] = np.max(tmp_scores)
            continuous_decoding[ibootstrap, 1] = np.mean(group_score)

        np.save('./data/%s/continuous_decoding_btheta_theta.npy'% prm.postprocess_name, continuous_decoding)
    else :
        continuous_decoding = np.load('./data/%s/continuous_decoding_btheta_theta.npy' % prm.postprocess_name)
    
    # PLOTTING --------------------------------------------------------
    fig, ax = plt.subplots(figsize = (8,5))
    
    ax.scatter(continuous_decoding[:,1], continuous_decoding[:,0], color = 'k', s = 100)
    
    cor, pval = stats.spearmanr(continuous_decoding[:,1], continuous_decoding[:,0])
    print('Spearman R = %.3f' % cor)
    print('Spearman p-value = %.3f' % pval)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(continuous_decoding[:,1], continuous_decoding[:,0])
    print('LinReg p-value = %.3f' % pvalue)
    print('LinReg r^2 = %.3f'% rvalue)
    print('Linreg slope, intercept ', slope, intercept)
    xplots = np.linspace(np.min(continuous_decoding[:,1]),
                        np.max(continuous_decoding[:,1]),
                        100)
    ax.plot(xplots, intercept + slope * xplots, c = 'k', lw = 2, ls = '--')

    ax.set_xlabel('Res/vul score', fontsize = 18)
    ax.set_ylabel('Max classification accuracy', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    labs = np.round(ax.get_xticks().tolist(),1)
    ylim_min, ylim_max = 0.12*.8, 0.12*1.2
    ax.set_ylim(ylim_min, ylim_max)
    yticks = np.linspace(ylim_min, ylim_max, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks,2))
    ax.set_xlim(0.15, 0.5)
    ax.set_xticks(np.linspace(0.15, 0.5, 4, endpoint = True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = (.025, .75), frameon = False, fontsize = 14, markerscale = .2)
    
    fig.tight_layout()
    fig.savefig('./figs/continuous_decoding_btheta_theta.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)