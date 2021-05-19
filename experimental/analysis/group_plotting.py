#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:10:30 2020

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # analysis:ignore
from matplotlib.ticker import MaxNLocator

import group_params as prm

import fileinput
import os
import sys
import imp


# --------------------------------------------------------------
# Recapitulative plots
# --------------------------------------------------------------
def recap() :
    print('# Plotting summary graphs #\n')
    
    clusters_path = './results/%s/' % prm.group_name
    
    min_cvs = cv_histogram()
    min_phis = phi_histogram()
    
    
    # Cv against phi
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.scatter(min_phis, min_cvs,
               c = np.array(min_cvs) + np.array(min_phis), cmap = 'plasma',
               edgecolor = 'k')
    
    ax.set_xlabel(r'Minimum $\varphi$', fontsize = 14)
    ax.set_ylabel('Minimum CirVar', fontsize = 14)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(min_phis)-2, np.max(min_phis)+2)
    
    fig.savefig(clusters_path + 'plot_recap_scatter_cvphi.pdf', bbox_inches = 'tight', format ='pdf')
    plt.close(fig) 
    
    
    
    # NKR distributions, Circular Variance
    nkr_bs, nkr_rmax, nkr_c50 = nkr_histogram(var = 'CirVar')

    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (14, 4))
    hist0 = axs[0].hist(nkr_bs, bins = np.linspace(np.min(nkr_bs), np.max(nkr_bs), prm.nkr_bins),
                rwidth = .9, facecolor = r'#2ea7c2', edgecolor = 'k')
    hist1 = axs[1].hist(nkr_c50, bins = np.linspace(np.min(nkr_c50), np.max(nkr_c50), prm.nkr_bins),
                rwidth = .9, facecolor = r'#2ea7c2', edgecolor = 'k')
    hist2 = axs[2].hist(nkr_rmax, bins = np.linspace(np.min(nkr_rmax), np.max(nkr_rmax), prm.nkr_bins),
            rwidth = .9, facecolor = r'#2ea7c2', edgecolor = 'k')
    
    ymax = np.max([h[0] for h in [hist0, hist1, hist2]]) #and the award to the ugliest one liner goes to...
    
    
    axs[0].set_ylabel('# neurons', fontsize = 14)
    axs[0].set_xlabel('Baseline (A.U.)', fontsize = 14)
    axs[1].set_xlabel(r'$B_{50}$ (°)', fontsize = 14)
    axs[2].set_xlabel(r'$R_{max}$ (A.U.)', fontsize = 14)
    
    for ax in axs :
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(0, ymax + .2) #bit more otherwise its cut
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.suptitle('Naka Rushton parameters, Circular Variance', fontsize = 16)
    fig.savefig(clusters_path + 'plot_recap_nkr_cirvar.pdf', bbox_inches = 'tight', format ='pdf')
    plt.close(fig) 
    
    
    # NKR distributions, Phi
    nkr_bs, nkr_rmax, nkr_c50 = nkr_histogram(var = 'Phi')

    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (14, 4))
    hist0 = axs[0].hist(nkr_bs, bins = np.linspace(np.min(nkr_bs), np.max(nkr_bs), prm.nkr_bins),
                rwidth = .9, facecolor = r'#00a087', edgecolor = 'k')
    hist1 = axs[1].hist(nkr_c50, bins = np.linspace(np.min(nkr_c50), np.max(nkr_c50), prm.nkr_bins),
                rwidth = .9, facecolor = r'#00a087', edgecolor = 'k')
    hist2 = axs[2].hist(nkr_rmax, bins = np.linspace(np.min(nkr_rmax), np.max(nkr_rmax), prm.nkr_bins),
                rwidth = .9, facecolor = r'#00a087', edgecolor = 'k')
    
    ymax = np.max([h[0] for h in [hist0, hist1, hist2]]) #and the award to the ugliest one liner goes to...
    
    
    axs[0].set_ylabel('# neurons', fontsize = 14)
    axs[0].set_xlabel('Baseline (°)', fontsize = 14)
    axs[1].set_xlabel(r'$B_{50}$ (°)', fontsize = 14)
    axs[2].set_xlabel(r'$R_{max}$ (°)', fontsize = 14)
    
    
    for ax in axs :
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(0, ymax + .2) #bit more otherwise its cut
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.suptitle(r'Naka Rushton parameters, $\varphi$', fontsize = 16)
    fig.savefig(clusters_path + 'plot_recap_nkr_phi.pdf', bbox_inches = 'tight', format ='pdf')
    plt.close(fig) 
    
 
# --------------------------------------------------------------
# Waveform plots
# --------------------------------------------------------------
def plot_waveform_KMeans() :
    '''
    Plots Kmeans classification of all the clusters
    '''
    print('# Plotting waveform kmeans#\n')
    
    xs1, ys1, xs2, ys2 = np.load('./results/%s/waveforms_clusters.npy' % prm.group_name, allow_pickle = True)
    
    fig, ax = plt.subplots(figsize = (8,8))

    ax.scatter(xs1, ys1,
               facecolor = 'firebrick', edgecolor = 'k', label = 'Putative regular spiking (exc)')
    ax.scatter(xs2, ys2,
               facecolor = 'royalblue', edgecolor = 'k', label = 'Putative fast spiking (inh)')
    
    ax.set_xlabel('Through to peak (ms)')
    ax.set_ylabel('Half width (ms)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.grid()
    fig.savefig('./results/%s/plot_kmeans_waveforms.pdf' % prm.group_name, format = 'pdf', bbox_inches = 'tight')
    plt.close(fig)

  
# --------------------------------------------------------------
# NKR correlation plots
# --------------------------------------------------------------    
def plot_nkr_correlation(var = 'Phi') :   
    print('# Plotting NKR correlation graphs #\n')
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    nkr_list = []
    for clust in clusters_list :
        if var == 'CirVar' :
            nkr_params = np.load(clusters_path + clust + '/0.000_cirvar_fit.npy', allow_pickle = True)
            nkr_params = nkr_params.item()
            nkr_list.append([nkr_params['b'],  nkr_params['c50']])
        if var == 'Phi' :
            nkr_params = np.load(clusters_path + clust + '/0.000_phi_fit.npy', allow_pickle = True)
            nkr_params = nkr_params.item()
            nkr_list.append([nkr_params['b'], nkr_params['c50']])
    
    baselines = [x[0] for x in nkr_list]
    b50s = np.asarray([x[1] for x in nkr_list]) * (prm.B_thetas.max() * 180 / np.pi)
    r, pvals = np.load('./results/%s/spearmanr_%s.npy' %  (prm.group_name, var))
    
    
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(b50s, baselines, s = 20,
               facecolor = r'#00a087' if var == 'Phi' else r'#2ea7c2', edgecolor = 'k')
    
    ax.set_xlabel(r'$B_{50}$ (°)', fontsize = 14)
    if var == 'Phi' :
        ax.set_ylabel('Baseline (°)', fontsize = 14)
    else :
        ax.set_ylabel('Baseline (U.A.)', fontsize = 14)
        
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.text(.85, .85, 'r = %.2f\np = %.2f' % (r, pvals),
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        fontsize = 12)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig('./results/%s/plot_nkr_correlations_%s.pdf' % (prm.group_name, var), format = 'pdf', bbox_inches = 'tight')
            
# --------------------------------------------------------------
# Fisher information
# --------------------------------------------------------------        
def plot_psth_fits() :
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    pref_list, orth_list = [], []
    for clust in clusters_list :
        pref_psth_params = np.load(clusters_path + clust + '/0.000_psth_fit_params_pref.npy', allow_pickle = True)
        orth_psth_params = np.load(clusters_path + clust + '/0.000_psth_fit_params_orth.npy', allow_pickle = True)
        
        pref_list.append(pref_psth_params) 
        orth_list.append(orth_psth_params) 
        
    keys = [] 
    for key, val in pref_list[0][0].items () : 
        keys.append(key)
    
    fig, ax  = plt.subplots(figsize = (12,12), ncols = 2, nrows = len(keys))
    
    # Pref
    for i0, key in enumerate(keys) :
        param_lst = []
        for i1, btheta in enumerate(prm.B_thetas) :
            neuron_lst = []
            for neuron in pref_list : 
                neuron_lst.append(neuron[i1][key])
            param_lst.append(np.mean(neuron_lst))
        
        ax[i0, 0].plot(prm.B_thetas * 180 / np.pi,param_lst)
        ax[i0, 0].set_title(key)
        ax[i0, 0].spines['right'].set_visible(False)
        ax[i0, 0].spines['top'].set_visible(False)
        
        
    # Orth
    for i0, key in enumerate(keys) :
        param_lst = []
        for i1, btheta in enumerate(prm.B_thetas) :
            neuron_lst = []
            for neuron in orth_list : 
                neuron_lst.append(neuron[i1][key])
            param_lst.append(np.mean(neuron_lst))
        
        ax[i0, 1].plot(prm.B_thetas * 180 / np.pi,param_lst)
        ax[i0, 1].set_title(key)
        ax[i0, 1].spines['right'].set_visible(False)
        ax[i0, 1].spines['top'].set_visible(False)
            
    plt.suptitle(y = 1.01, t = 'Pref                                        Orth')
    plt.tight_layout()
    fig.savefig('./results/%s/plot_psth_params.pdf' % (prm.group_name), format = 'pdf', bbox_inches = 'tight')
            
            
            
# --------------------------------------------------------------
# Fisher information
# -------------------------------------------------------------- 
def plot_fisher_info() :
    print('# Plotting Fisher information #\n')
    
    fisher_info = np.load('./results/%s/pop_fisher_info.npy' % prm.group_name, allow_pickle = True)
    
    fig, ax = plt.subplots(figsize = (12,8))
    # parts = ax.violinplot([x for x in fisher_info], positions = np.round(prm.B_thetas * 180 / np.pi,1),
    #               widths = 1.5, showmeans = True, showextrema = True )
    # for pc, z in zip(parts['bodies'], prm.colors):
    #     pc.set_facecolor(z)
    #     pc.set_edgecolor('k')
    #     pc.set_alpha(.8)
    # for partname in ('cbars','cmins','cmaxes','cmeans'):
    #     pc = parts[partname]
    #     pc.set_edgecolor('k')
    #     pc.set_linewidth(1) 
        
    # pc.set_edgecolor('black')
    # pc.set_alpha(1)
    #fisher_info = np.load('./results/%s/pop_fisher_info.npy' % prm.group_name)
    ax.plot(np.round(prm.B_thetas * 180 / np.pi,1), np.mean(fisher_info, axis = 1) )
    
    ax.set_xticks(np.round(prm.B_thetas * 180 / np.pi,1))
    ax.set_xlabel(r'$B_{\theta}$ (°)', fontsize = 14)
    ax.set_ylabel(r'Mean Fisher information ($\theta^2$)', fontsize = 14)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig('./results/%s/plot_fisher_information.pdf' % (prm.group_name), format = 'pdf', bbox_inches = 'tight')
    
    
# --------------------------------------------------------------
# Decoding
# -------------------------------------------------------------- 
def plot_ori_decoding() :
    print('# Plotting orientation decoder #\n')

    for exp in prm.exp_list :
        log_likelihoods = np.load('./results/%s/%s_decoder_ori_likelihoods.npy' % (prm.group_name,exp))
        log_likelihoods = np.flip(log_likelihoods, axis = 0 ) #easier to specify the indices in parameters this way
        colors = prm.colors[::-1] 
        #fitted_logs = np.load('./results/%s/%s_decoder_ori_tcs.npy' % (prm.group_name,exp), allow_pickle = True)
        #fit_xs = np.linspace(prm.thetas[0], prm.thetas[-1], len(fitted_logs[0,0]))
        

        fig, axs = plt.subplots(nrows = len(prm.dec_ori_idxs),
                                ncols = len(prm.dec_ori_thetas),
                                figsize = (16,6))
        
        mins, maxs = [], []
        for i0, btheta in enumerate(prm.dec_ori_idxs) : 
            mins.append(np.min(log_likelihoods[btheta,:,:], axis = -1).min())
            maxs.append(np.max(log_likelihoods[btheta,:,:], axis = -1).max())
            for i1, theta in enumerate(prm.dec_ori_thetas) :
                llk = log_likelihoods[btheta, theta, :]

                axs[i0, i1].plot(prm.thetas, llk,
                                 color = colors[btheta])
                axs[i0, i1].vlines(prm.thetas[theta], ymin = 0., ymax = llk[theta],
                                  linestyle = '--', color = 'gray')
                #axs[i0, i1].set_title(np.max(norm_llk) / np.mean(norm_llk))
                #axs[i0, i1].plot(fit_xs, fitted_logs[btheta, theta])
                #axs[i0, i1].set_ylim(mins[i0] - (mins[i0] / 8), maxs[i0] + (maxs[i0] / 8))
                
                
    
        for x in range(len(prm.dec_ori_idxs)) :
            for y in range(len(prm.dec_ori_thetas)) :
                #axs[x,y].set_yticks(np.linspace(mins[x], maxs[x], 3, endpoint = True))
                if not (x == len(prm.dec_ori_idxs) -1 and y == 0) :
                    axs[x,y].set_yticklabels([])
                axs[x,y].set_xticks(prm.thetas[::2])
                axs[x,y].set_xticklabels([])
                axs[x,y].spines['right'].set_visible(False)
                axs[x,y].spines['top'].set_visible(False)
                
            # abs is for the -0 = 0 
            #axs[x, 0].set_yticklabels(np.abs(np.round(np.linspace(mins[x], maxs[x], 3, endpoint = True), -1)))
    
        # integer conversion nightmare
        axs[-1, 0].set_xticklabels(np.round(prm.thetas[::2] * 180 / np.pi , -1).astype(np.int16), fontsize = 9)
        axs[-1, 0].set_xlabel(r'$\theta$ (°)', fontsize = 12)
        axs[-1, 0].set_ylabel('norm. log-likelihood', fontsize = 12)
      
        plt.tight_layout(w_pad = .75, h_pad = 1.4)
        fig.savefig('./results/%s/plot_decoder_ori_%s.pdf' % (prm.group_name, exp), format = 'pdf', bbox_inches = 'tight')  
    
    
def plot_ori_decoding_errors() :
    print('# Plotting orientation decoder precision #\n')

    for exp in prm.exp_list :
        # Decoder error (angular)
        log_errors = np.load('./results/%s/%s_decoder_ori_errors.npy' % (prm.group_name,exp))

        fig, ax = plt.subplots(figsize = (12,6))
        
        ax.plot(prm.B_thetas * 180 / np.pi, np.mean(log_errors, axis = 1))
        ax.set_xlabel(r'$B_{\theta}$ (°)', fontsize = 14)
        ax.set_ylabel(r'Mean decoder error (°)', fontsize = 14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        fig.savefig('./results/%s/plot_decoder_ori_errors_%s.pdf' % (prm.group_name, exp), format = 'pdf', bbox_inches = 'tight')  
        
        
        # Decoder SNR (peak to mean)
        log_snrs = np.load('./results/%s/%s_decoder_ori_snrs.npy' % (prm.group_name,exp))

        fig, ax = plt.subplots(figsize = (12,6))
        
        ax.plot(prm.B_thetas * 180 / np.pi, np.mean(log_snrs, axis = 1))
        ax.set_xlabel(r'$B_{\theta}$ (°)', fontsize = 14)
        ax.set_ylabel(r'Max/mean decoder (u.a.)', fontsize = 14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        fig.savefig('./results/%s/plot_decoder_ori_snrs_%s.pdf' % (prm.group_name, exp), format = 'pdf', bbox_inches = 'tight')  
     
# --------------------------------------------------------------
# Associated functions to bin data into histograms
# --------------------------------------------------------------
def cv_histogram() :
    '''
    Circular variance histogram, plotted later
    '''
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    min_cvs = []
    for clust in clusters_list :
        cv = np.load(clusters_path + clust + '/0.000_cirvar.npy')
        min_cvs.append(np.min(cv))
    
    cv_hist = np.histogram(min_cvs, bins = prm.min_cv_nbins)
    np.save('./results/%s/%s' % (prm.group_name, 'cv_hist.npy'),
            cv_hist)
    
    return min_cvs
    
def phi_histogram() :
    '''
    Phi histogram, plotted later
    '''
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    min_phis = []
    for clust in clusters_list :
        phi = np.load(clusters_path + clust + '/0.000_plot_neurometric_Btheta_fits.npy')
        min_phis.append(np.min(phi))
    
    phi_hist = np.histogram(min_phis, bins = prm.min_phi_nbins)
    np.save('./results/%s/%s' % (prm.group_name, 'phi_hist.npy'),
            phi_hist)
    
    return min_phis

def nkr_histogram(var) :
    '''
    Naka Rushton parameters histograms
    '''
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    nkr_list = []
    for clust in clusters_list :
        if var == 'CirVar' :
            nkr_params = np.load(clusters_path + clust + '/0.000_cirvar_fit.npy', allow_pickle = True)
            nkr_params = nkr_params.item()
            nkr_list.append([nkr_params['b'], nkr_params['rmax'], nkr_params['c50']])
        if var == 'Phi' :
            nkr_params = np.load(clusters_path + clust + '/0.000_phi_fit.npy', allow_pickle = True)
            nkr_params = nkr_params.item()
            nkr_list.append([nkr_params['b'], nkr_params['rmax'], nkr_params['c50']])
    
    b_hist = [x[0] for x in nkr_list]
    rmax_hist = [x[1] for x in nkr_list]
    c50_hist = np.asarray([x[2] for x in nkr_list]) * (prm.B_thetas.max() * 180 / np.pi)
    
    return b_hist, rmax_hist, c50_hist


# --------------------------------------------------------------
# Misc utils
# --------------------------------------------------------------
    
def replace_if_exist(file, searchExp, replaceExp):
    '''
    Changes the value of a variable in a .py file if it exists, otherwise writes it
    replaceExp must contain \n for formatting
    '''
    
    infile = False
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = replaceExp
            infile = True
        sys.stdout.write(line)
     
    if infile == False :
        with open(file, 'a') as file :
            file.write(replaceExp)
            
def get_var_from_file(filename):
    f = open(filename)
    cluster_info = imp.load_source('cluster_info', filename)
    f.close()
    
    return cluster_info


