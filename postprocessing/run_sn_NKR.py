#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import utils_single_neuron as sn_utils

from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt 
from lmfit import Model, Parameters 
from tqdm import tqdm 
from scipy import stats
import scikit_posthocs as sp
import utils 

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings 
warnings.catch_warnings()
warnings.simplefilter("ignore")

# --------------------------------------------------------------
# Plotting NKR of neurons
# --------------------------------------------------------------
def make_NKR(dtype = 'CV') :
    print('Making NKR for example neurons')
    for cluster_path in prm.main_neurons:
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        
        if dtype == 'CV' :
            CVs = np.load(load_folder + '/cirvar.npy')
            fit_cv = np.load(load_folder + '/cirvar_fit.npy', allow_pickle = True)
            fit_cv = fit_cv.item()
            print(cluster_path)
            print('CV n = %.2f ; B50 = %.2f°, r^2 = %.2f' %
                (fit_cv['n'], fit_cv['c50']*np.max(prm.B_thetas), fit_cv['r2']))

            sn_utils.make_NKR_monoaxis(figsize=(5,2.5), filename=cluster_path,
                            fit = fit_cv, raw = CVs,
                            B_thetas=prm.B_thetas, sharey = True, dtype = dtype)

        elif dtype == 'Rmax':
            CVs = np.load(load_folder + '/rmax.npy') # variable name is kept but it is rmax indeed
            fit_cv = np.load(load_folder + '/rmax_fit.npy', allow_pickle = True)
            fit_cv = fit_cv.item()
            print(cluster_path)
            print('Rmax n = %.2f ; B50 = %.2f°, r^2 = %.2f' %
                (fit_cv['n'], fit_cv['c50']*np.max(prm.B_thetas), fit_cv['r2']))

            sn_utils.make_NKR_monoaxis(figsize=(5,2.5), filename=cluster_path,
                            fit = fit_cv, raw = CVs,
                            B_thetas=prm.B_thetas, sharey = True, dtype = dtype)        
        
        elif dtype == 'HWHH':
            CVs = np.load(load_folder + '/hwhh.npy')
            fit_cv = np.load(load_folder + '/hwhh_fit.npy', allow_pickle = True)
            fit_cv = fit_cv.item()
            print(cluster_path)
            print('hwhh n = %.2f ; B50 = %.2f°, r^2 = %.2f' %
                (fit_cv['n'], fit_cv['c50']*np.max(prm.B_thetas), fit_cv['r2']))

            sn_utils.make_NKR_monoaxis(figsize=(5,2.5), filename=cluster_path,
                            fit = fit_cv, raw = CVs,
                            B_thetas=prm.B_thetas, sharey = True, dtype = dtype)
                            
                            
# --------------------------------------------------------------
# Plotting population parameters
# --------------------------------------------------------------
def make_NKR_histogram(dtype = 'CV') :
    nkrcvs = []
    data_cvs = []
    for cluster_path in prm.cluster_list:
        load_folder = prm.grouping_path + '/' + cluster_path
        try :
            if dtype == 'CV' :
                data = np.load(load_folder + '/cirvar.npy')
            elif dtype == 'Rmax':
                data = utils.norm_data(np.load(load_folder + '/rmax.npy'))
            elif dtype == 'HWHH' :
                data = utils.norm_data(np.load(load_folder + '/hwhh.npy'))
            

            arr_fit, r2 = sn_utils.fit_nkr(data)
            if arr_fit['b']< -0. or arr_fit['b'] > 1. : continue
            nkrcvs.append(arr_fit)
            data_cvs.append(data)
            
        except ValueError :
            pass
        
    n_cv = np.asarray([x['n'] for x in nkrcvs])
    b50_cv = np.asarray([x['c50'] for x in nkrcvs])
    max_cv = np.asarray([x['rmax'] for x in nkrcvs])
    b_cv = np.asarray([x['b'] for x in nkrcvs])
    
    fig, axs = plt.subplots(figsize=(12, 4), ncols=3)

    n_bins = 8
    pct = 5
    # Exponant n
    bins = np.linspace(np.percentile(np.log(n_cv), pct),
                                    np.percentile(np.log(n_cv), 100-pct),
                                    n_bins)
    ax_bins = np.linspace(np.percentile(np.log(n_cv), pct),
                                        np.percentile(np.log(n_cv), 100-pct),
                                        3)
    axs[0].hist(np.log(n_cv), bins = bins,
                facecolor=r'#3C5488', edgecolor='w')
    axs[0].set_xticks(np.round(ax_bins,1))
    axs[0].set_xlabel(r'$\log (n)$', fontsize = 14)
    axs[0].set_ylabel('# neurons', fontsize=14)
    axs[0].set_yticks(np.linspace(0, 60, 3, dtype = int))
    axs[0].set_ylim(0, 60)

    # B50
    axs[1].hist(b50_cv*180/np.pi, bins=prm.B_thetas[::-1]*180/np.pi,
                facecolor=r'#3C5488', edgecolor='w')
    axs[1].set_xticks([0.0,18.0,36.0])
    axs[1].set_xticklabels(['0.0', '18.0', '36.0'])
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].set_xlabel(r'$B_{\theta50} (°)$', fontsize=14)
    axs[1].set_yticks(np.linspace(0, 30, 3, dtype = int))
    axs[1].set_ylim(0, 30)

    # F0
    bins=np.linspace(np.percentile(b_cv, pct),
                np.percentile(b_cv, 100-pct),
                n_bins)
    ax_bins=np.linspace(np.percentile(b_cv, pct),
                np.percentile(b_cv, 100-pct),
                3)
    axs[2].hist(b_cv, bins = bins,
                facecolor=r'#3C5488', edgecolor='w')
    axs[2].set_xticks(np.round(ax_bins,2))
    axs[2].set_xticklabels(np.round(ax_bins,1))
    axs[2].tick_params(axis='both', which='major', labelsize=12)
    axs[2].set_xlabel(r'$f_{0} (°)$', fontsize=14)
    axs[2].set_yticks(np.linspace(0, 50, 3, dtype = int))
    axs[2].set_ylim(0, 50)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig.tight_layout()
    fig.savefig('./figs/sn_NKR_%s_histo.pdf' % dtype, bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

    print('\nMedian for %s values (baseline, n, btheta50' % dtype)
    print(np.median(b_cv))
    print(np.median(np.log(n_cv)))
    histo = b50_cv[np.where((b50_cv*180/np.pi) < 36.0)[0]]*180/np.pi #rescale
    print(np.median(histo))

# --------------------------------------------------------------
# Plotting circular variance for the two groups
# --------------------------------------------------------------
def make_CV_plot(dtype = 'CV') :
    data_cvs_tuned, data_cvs_untuned = [], []
    for cluster_path in prm.cluster_list:
        load_folder = prm.grouping_path + '/' + cluster_path
        
        try :
            if dtype == 'CV' :
                data = np.load(load_folder + '/cirvar.npy')
            elif dtype == 'Rmax':
                data = utils.norm_data(np.load(load_folder + '/rmax.npy'))
            elif dtype == 'HWHH' :
                data = utils.norm_data(np.load(load_folder + '/hwhh.npy'))

            arr_fit, r2 = sn_utils.fit_nkr(data)
            if arr_fit['b']> 0.95 : continue
            
            if dtype == 'CV' :
                data = np.load(load_folder + '/cirvar.npy')
            elif dtype == 'Rmax':
                data = np.load(load_folder + '/rmax.npy')
            elif dtype == 'HWHH' :
                data = np.load(load_folder + '/hwhh.npy')
                
            if cluster_path in prm.tuned_lst :
                data_cvs_tuned.append(data)
            else :
                data_cvs_untuned.append(data)
        except ValueError :
            pass
        
    fig, ax = plt.subplots(figsize = (6,6))

    # Circular variance at B_theta = 0°
    c = prm.col_tuned
    ax.boxplot([x[-1] for x in data_cvs_tuned],
                        positions = [0],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot([x[-1] for x in data_cvs_untuned],
                        positions = [.3],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    # Circular variance at B_theta = 36°
    c = prm.col_tuned
    ax.boxplot([x[0] for x in data_cvs_tuned],
                        positions = [.8],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot([x[0] for x in data_cvs_untuned],
                        positions = [.8+.3],
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
    ax.set_ylabel('%s' % dtype, fontsize = 18)
    ax.set_yticks(np.linspace(0., 1., 3))

    if dtype == 'CV' :
        ax.set_ylim(0, 1)
    elif dtype == 'Rmax':
        ymax = np.max(np.concatenate((data_cvs_tuned, data_cvs_untuned)))
        ax.set_ylim(0, 1.1*ymax)
        ax.set_yticks(np.linspace(0., 1.1*ymax,3))
    elif dtype == 'HWHH' :
        ymax = np.max(np.concatenate((data_cvs_tuned, data_cvs_untuned)))
        ax.set_ylim(0, 1.1*ymax)
        ax.set_yticks(np.linspace(0., 1.1*ymax,3))
    

    ax.set_xticks([.15, .95])
    ax.set_xticklabels([r'$B_\theta = 0°$', r'$B_\theta = 35°$'])
    fig.savefig('./figs/clustering_%s.pdf' % dtype, bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

# --------------------------------------------------------------
# Plotting the parameters for the two groups
# --------------------------------------------------------------
def make_params_plot(dtype = 'CV') :
    correlation_array = np.load('./data/%s/correlation_array.npy' % prm.postprocess_name, allow_pickle = True)
    cv_n_tuned, cv_b_tuned, cv_b50_tuned = [], [], []
    cv_n_untuned, cv_b_untuned, cv_b50_untuned = [], [], []
    for iclust, clust in enumerate(correlation_array) :
        if clust['cluster'] in prm.tuned_lst :
            if dtype == 'CV' :
                cv_n_tuned.append(clust['cv'][1]['n'])
                cv_b_tuned.append(clust['cv'][1]['b'])
                cv_b50_tuned.append(clust['cv'][1]['c50'])
            elif dtype == 'HWHH' :
                cv_n_tuned.append(clust['hwhh'][1]['n'])
                cv_b_tuned.append(clust['hwhh'][1]['b'])
                cv_b50_tuned.append(clust['hwhh'][1]['c50'])
            elif dtype == 'Rmax' :
                cv_n_tuned.append(clust['rmax'][1]['n'])
                cv_b_tuned.append(clust['rmax'][1]['b'])
                cv_b50_tuned.append(clust['rmax'][1]['c50'])
                
        else :
            if dtype == 'CV' :
                cv_n_untuned.append(clust['cv'][1]['n'])
                cv_b_untuned.append(clust['cv'][1]['b'])
                cv_b50_untuned.append(clust['cv'][1]['c50'])
            elif dtype == 'HWHH' :
                cv_n_untuned.append(clust['hwhh'][1]['n'])
                cv_b_untuned.append(clust['hwhh'][1]['b'])
                cv_b50_untuned.append(clust['hwhh'][1]['c50'])
            elif dtype == 'Rmax' :
                cv_n_untuned.append(clust['rmax'][1]['n'])
                cv_b_untuned.append(clust['rmax'][1]['b'])
                cv_b50_untuned.append(clust['rmax'][1]['c50'])
                
    cv_n_tuned, cv_n_untuned = np.log(cv_n_tuned), np.log(cv_n_untuned)
    
    # log(n) --------------------------------------------------------------
    fig, ax = plt.subplots(figsize = (6,6))

    c = prm.col_tuned
    ax.boxplot(cv_n_tuned,
                        positions = [0.3],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot(cv_n_untuned,
                        positions = [0],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks(np.linspace(0,6,4))

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'log(n)', fontsize = 18)
    ax.set_ylim(-.1,6.)
    ax.set_yticks([0,3,6.])
    ax.set_xticks([])
    fig.savefig('./figs/clustering_nkr_n_%s.pdf' % dtype, bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    
    print('%s n tuned vs untuned' % dtype)
    print(mannwhitneyu(cv_n_tuned, cv_n_untuned, alternative = 'less'))
    
    
    # baseline --------------------------------------------------------------
    fig, ax = plt.subplots(figsize = (6,6))

    c = prm.col_tuned
    ax.boxplot(cv_b_tuned,
                        positions = [0.3],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot(cv_b_untuned,
                        positions = [0],
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
    #ax.set_xlabel(r'Orientation descriptor', fontsize = 18)
    ax.set_ylabel(r'$f_0$', fontsize = 18)
    ax.set_yticks([0,.5,1.])
    ax.set_ylim(0,1)
    
    if dtype == 'HWHH' or dtype == 'Rmax':
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_ylim(0, 1)

    fig.savefig('./figs/clustering_nkr_b_%s.pdf' % dtype, bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    print('%s b tuned vs untuned' % dtype)
    print(mannwhitneyu(cv_b_tuned, cv_b_untuned, alternative = 'less'))
        
        
    # btheta50 --------------------------------------------------------------
    fig, ax = plt.subplots(figsize = (6,6))

    c = prm.col_tuned
    ax.boxplot(cv_b50_tuned,
                        positions = [0.3],
                        widths = .2, showmeans = False,
                showfliers = False, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white')) 

    c = prm.col_untuned
    ax.boxplot(cv_b50_untuned,
                        positions = [0],
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
    #ax.set_xlabel(r'Orientation descriptor', fontsize = 18)
    ax.set_ylabel(r'$B_{\theta50}$', fontsize = 18)

    ax.set_yticks([0, .5, 1.])
    ax.set_yticklabels([0, 18, 36])
    ax.set_ylim(-.02, 1.)

    ax.set_xticks([])
    fig.savefig('./figs/clustering_nkr_b50_%s.pdf' % dtype, bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)
    print('%s b50 tuned vs untuned' % dtype)
    print(mannwhitneyu(cv_b50_tuned, cv_b50_untuned, alternative = 'greater'))    
    
    
def make_NKR_BIC(dtype = 'CV'):
    print('Computing Bayesian Information Criterion for NKR')
    method_list = ['nkr', 'ReLU', 'sigmoid', 'pol2', 'pol3']
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,8))


    stat_list = []

    dico = make_new_dico()
    data = get_data(method_list, varname = dtype, cluster_list = prm.cluster_list, grouping_path = prm.grouping_path, dico = dico)
    plot_diffs(dico, method_list, dtype, ax, do_label = True)
    
    ks, kpval, dunn = compute_stats(dico, method_list, criterion = 'bic')
    stat_list.append([dtype, ks, kpval, dunn])
        
    fig.savefig('./figs/sn_NKR_BIC_%s.pdf' % dtype, bbox_inches='tight', dpi=200, transparent=True)

    plt.show(block = prm.block_plot)
    
    print('Variable %s' % stat_list[0])
    print('Kruskal value %.3f, pval %.12f' % (stat_list[0][1], stat_list[0][2]))
    print('Dunn Posthoc :')
    print(stat_list[0][-1])
    print('-------------------\n\n')
    
def nakarushton(x, rmax, c50, b, n):
    nkr = b + (rmax-b) * ((x**n) / (x**n + c50**n))
    return nkr

def ReLU(x, gamma) :
     return x * gamma * (x > 0)
    
def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))
    
def pol2(x, a, b, c) :
    return a*x**2 + b * x + c

def pol3(x, a, b, c, d) :
    return a * x ** 3 + b * x ** 2 + c * x + d

def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def fit_data(data, method) :
    y = norm_data(data)
    
    if method == 'nkr' :
        x = np.linspace(0, 1, len(y))
        mod = Model(nakarushton)
        pars = Parameters()
        pars.add_many(('rmax',  np.max(y), True,  0.0,  100.),
                      ('c50', .5, True,  0.001, 10.),
                      ('b', y[0], True, y[0] * .5 + .001, y[0] * 3 + .002 ),
                      ('n', 4, True,  1., 100.))
        out = mod.fit(y, pars, x=x, nan_policy='omit')
        
    elif method == 'ReLU' :
        x = np.linspace(-1, 1, len(y))
        model = Model(ReLU)
        params = Parameters()
        params.add('gamma', value=.5, min=.1, max=80., vary = True)
        out = model.fit(y, params, x=x, nan_policy='omit')
        
    elif method == 'sigmoid' :
        x = np.linspace(0, 1, len(y))
        mod = Model(sigmoid)
        pars = Parameters()
        pars.add_many(('a',  np.max(y), True,  0.0,  100.),
                      ('b', .5, True,  0.0, 10.))
        out = mod.fit(y, pars, x=x, nan_policy='omit')
        
        
    elif method == 'pol2' :
        x = np.linspace(0, 1, len(y))
        mod = Model(pol2)
        pars = Parameters()
        pars.add_many(('a',  .5, True,  0.0,  100.),
                      ('b', .5, True,  0.0, 10.),
                     ('c', .5, True,  0.0, 10.))
        out = mod.fit(y, pars, x=x, nan_policy='omit')
        
    elif method == 'pol3' :
        x = np.linspace(0, 1, len(y))
        mod = Model(pol2)
        pars = Parameters()
        pars.add_many(('a',  .5, True,  0.0,  100.),
                      ('b', .5, True,  0.0, 10.),
                     ('c', .5, True,  0.0, 10.),
                     ('d', .5, True,  0.0, 10.))
        out = mod.fit(y, pars, x=x, nan_policy='omit')
        
    return out.chisqr, out.aic, out.bic, out.values

def make_new_dico() :
    dico = {'nkr' : {'bic' : []},
            'ReLU' : {'bic' : []},
            'sigmoid' : {'bic' : []},
            'pol2' : {'bic' : []},
            'pol3' : {'bic' : []}
            }
    return dico

def plot_diffs(dico, method_list, varname, ax, do_label) :
    
    for i, method in enumerate(method_list) :
        violin_parts = ax.violinplot(dico[method]['bic'], positions = [i], showmeans = True)
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = violin_parts[partname]
            vp.set_edgecolor('k')
            vp.set_linewidth(1.5)
        for vp in violin_parts['bodies']:
            vp.set_facecolor(r'#00a087' if varname == 'HWHH' else r'#3C5488FF')
            vp.set_edgecolor(None)
            vp.set_linewidth(1)
            vp.set_alpha(0.5)
           
    if do_label :
        ax.set_xticks(np.arange(len(dico)))
        ax.set_xticklabels(method_list, fontsize = 18)
        
    else :
        ax.set_xticklabels([])
        
    ax.set_ylabel('BIC (%s)' % varname, fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    ax.set_xlim(-1, len(dico) - .5)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return ax
    
def compute_stats(dico, method_list, criterion) :
    list_stats = [dico[x][criterion] for x in method_list]
    kruskal_stat, kruskal_pval = stats.kruskal(*list_stats)

    dunn = sp.posthoc_dunn(a = list_stats, sort = True)
    dunn = dunn.rename(columns = {1:method_list[0], 2:method_list[1], 3: method_list[2], 4 : method_list[3]},
               index = {1:method_list[0], 2:method_list[1], 3: method_list[2], 4 : method_list[3]})

    return kruskal_stat, kruskal_pval, dunn

def get_data(method_list, varname, cluster_list, grouping_path, dico):
    for method in tqdm(method_list) : 
        chisqr, aic, bic = [], [], []
        for cluster_path in cluster_list :
            
            folder_path = '_'.join(cluster_path.split('_')[:2])

            if varname == 'CV' :
                #raw_data = np.load(grouping_path + '/%s/0.000_plot_neurometric_fitted_TC.npy' % (cluster_path))
                #raw_data = np.max(raw_data, axis = 1) < ----- this guy could be useful 
                raw_data = np.load(grouping_path + '/%s/cirvar.npy' % (cluster_path))
            elif varname == 'HWHH' : 
                raw_data = np.load(grouping_path + '/%s/hwhh.npy' % (cluster_path))
            elif varname == 'Rmax' :
                raw_data = np.load(grouping_path + '/%s/rmax.npy' % (cluster_path))

            results = fit_data(raw_data, method = method)

            chisqr.append(results[0])
            aic.append(results[1])
            bic.append(results[2])

        dico[method]['chisqr'] = chisqr
        dico[method]['aic'] = aic
        dico[method]['bic'] = bic
    return dico