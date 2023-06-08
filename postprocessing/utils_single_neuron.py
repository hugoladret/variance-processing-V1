#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import numpy as np 
import utils 
from lmfit import Model, Parameters
import matplotlib.pyplot as plt 
from tqdm import tqdm
from matplotlib import gridspec
import sys
sys.path.append('../preprocessing/cat/') # used to re-import the fitting Torch Functions
import pipeline_fit as pfit

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings 
warnings.catch_warnings()
warnings.simplefilter("ignore")

# --------------------------------------------------------------
# Plotting Tuning curves
# --------------------------------------------------------------
def make_tc_multiaxes(figsize, filename, fitted_TC, mean_FR,
                    B_thetas, mult_TC, do_legend, do_btlabs):
    
    fig, axs = plt.subplots(nrows = 1, ncols = 8, figsize=figsize)

    fitted_TC *= mult_TC
    mean_FR *= mult_TC

    for i, _ in enumerate(B_thetas):
        ax = axs[i]
        xs = np.linspace(0, 180, len(fitted_TC[i]))

        ax.plot(xs, fitted_TC[i], c=prm.colors[i], zorder=i,
                label=r'$B_\theta$ = %.2f°' % (B_thetas[i] * 180/np.pi))
        ax.fill_between(xs, np.min(mean_FR[i]), fitted_TC[i], color = prm.colors[i],
                    zorder = 0, alpha = .6)
        ax.scatter(np.linspace(0, 180, 12), mean_FR[i], facecolor = prm.colors[i],
                zorder=i-.5, edgecolor=None, s=10)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_ylim(.98*np.min(fitted_TC[0]), 1.1 * np.max(fitted_TC[0]))

        if i == 0 :
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_xlabel(r'$\theta$(°)', fontsize=16)
            ax.set_ylabel(r'Firing rate (sp.s$^{-1})$', fontsize = 16)
            yticks = np.linspace(.98*np.min(fitted_TC[0]), 1.1 * np.max(fitted_TC[0]), 4, dtype = np.int16)
            ax.set_yticks(yticks)
        else :
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_xticklabels([])
            
        if do_btlabs :
            ax.text(x = -110, y = .5 * np.max(fitted_TC[0]), s = r'$B_\theta$=%.1f°'% (B_thetas[::-1][i] * 180/np.pi),
                color = prm.colors[i], fontsize = 14, va = 'center', ha = 'center') 

    fig.subplots_adjust(wspace = 0.1, hspace = 0)
    if do_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], frameon = False, fontsize = 14)
    fig.savefig('./figs/sn_TC_%s.pdf' %
                filename, bbox_inches='tight', dpi=200, transparent=True)
    
    plt.show(block = prm.block_plot)
    
    
# --------------------------------------------------------------
# Plotting PSTH of neurons
# --------------------------------------------------------------
def make_reduced_raster(figsize,  filename,
                        nm_PSTH_list, nmmeans,
                        bt_idxs):

    n_bin = (((prm.end_psth) - (prm.beg_psth)) * 1000) / prm.binsize
    
    # Figure init
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(9, 4)

    # The same for btheta by btheta
    mins, maxs = [], []
    for i in range(0, 8):
        hists = [np.histogram(np.concatenate(theta), int(n_bin))[0]
                for theta in nm_PSTH_list[i-1, :, :]]
        mins.append(np.min(hists))
        maxs.append(np.max(hists))

    # Pref/Orth orientation based on btheta = 0 FR
    prefered_id = np.argmax(nmmeans[-1, :])

    # Preferred orientation
    # Rasters
    for i, y in enumerate([2, 7]):
        bt = bt_idxs[i]
        ax = plt.subplot(gs[y:y+2, :4])
        theta = nm_PSTH_list[bt, prefered_id, :]
        for it_1, trial in enumerate(theta):
            ax.scatter(trial, np.full_like(
                trial, it_1), s=1, color=prm.colors[::-1][bt])

        xticks = np.linspace(prm.beg_psth, prm.end_psth, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.asarray(xticks*1000, dtype=int))
        ax.set_xlim(prm.beg_psth, prm.end_psth)
        # ax.set_yticks([0,30])
        ax.set_yticks([])
        if i >= 1:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Trial', fontsize=13)

        if i == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('PST (ms)', fontsize=14)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

    # PSTH
    for i, y in enumerate([0, 5]):
        bt = bt_idxs[i]
        ax = plt.subplot(gs[y:y+2, :4])

        ax.hist(np.concatenate(nm_PSTH_list[bt, prefered_id, :]),
                np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)),
                color=prm.colors[::-1][bt], edgecolor=prm.colors[::-1][bt])

        ax.set_ylim(np.min(mins), np.max(maxs))
        ax.set_xlim(prm.beg_psth, prm.end_psth)
        ax.set_xticks([])
        ax.set_yticks([0, np.max(maxs)/2, np.max(maxs)])
        ax.set_yticklabels(np.asarray(
            [0, np.max(maxs)/2, np.max(maxs)], dtype=int))
        if i >= 1:
            ax.set_yticklabels([])
            # ax.set_yticks([])
        else:
            ax.set_ylabel('sp/bin', fontsize=13)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

    # fig.tight_layout()
    fig.align_ylabels()
    fig.savefig('./figs/sn_PSTH_%s.pdf' %
                filename, bbox_inches='tight', dpi=200, transparent=True)

    plt.show(block = prm.block_plot)
    
    
# --------------------------------------------------------------
# Performing refit (for temporal TC)
# --------------------------------------------------------------
def torch_fit_tc(full_array):
    fit_array = np.zeros((full_array.shape[0],
                        full_array.shape[1],
                        256))
    params_array = np.zeros((full_array.shape[1], 5), dtype = object)
    
    for delay in tqdm(np.arange(full_array.shape[1]), desc = 'Fitting for each delay') :
        TC_data = full_array[:,delay,:]
        theta0, R0, Rmax, kappas, fitted_fr, ploss = pfit.torch_fit(TC_data, disable_cuda = False)
        hwhh = .5*np.arccos(1+ np.log((1+np.exp(-2*kappas))/2)/kappas)
        hwhh = hwhh * 180 / np.pi
        fit_array[:,delay,:] = fitted_fr
        params_array[delay, :] = [theta0, R0, Rmax, kappas, hwhh]
        
    return fit_array, params_array

# --------------------------------------------------------------
# Plotting those temporal TC
# --------------------------------------------------------------
def plot_temporal_array(tc, full_array, plot_idxs, plt_min, plt_max) :
    fig, axs = plt.subplots(ncols = len(plot_idxs), nrows = 2,
                          figsize = (3*len(plot_idxs), 6))
    
    x_fits = np.linspace(0, np.pi, 256)
    x_raw = np.linspace(0, np.pi, 12)
    
    yticks = np.linspace(plt_min, plt_max, 3)
    
    for i0, ax in enumerate(axs[0,:]) :
        ax.scatter(x_raw, full_array[-1,plot_idxs[i0],:],
                   facecolor=prm.colors[-1], edgecolor=None,
                   zorder=-.5, s=30)
        ax.plot(x_fits, tc[-1,plot_idxs[i0],:],
                color = prm.colors[-1], linewidth = 1)
        ax.fill_between(x_fits, 0, tc[-1,plot_idxs[i0],:],
                        color = prm.colors[-1],
                        zorder = 0, alpha = .6)
        ax.set_title(r't = %.2fs' % prm.latencies[plot_idxs[i0]], fontsize = 14)

    for i0, ax in enumerate(axs[1,:]) :
        ax.scatter(x_raw, full_array[0,plot_idxs[i0],:],
                   facecolor=prm.colors[0], edgecolor=None,
                   zorder=-.5, s=30)
        ax.plot(x_fits, tc[0,plot_idxs[i0],:],
                color = prm.colors[0], linewidth = 1)
        ax.fill_between(x_fits, 0, tc[0,plot_idxs[i0],:],
                        color = prm.colors[0],
                        zorder = 0, alpha = .6)

    for ax in axs.flatten() :
        ax.set_xticks(np.linspace(0, 3.14, 5))
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
            
        ax.set_ylim(plt_min, plt_max)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    for ax in [axs[0,0], axs[1,0]] :
        ax.set_xticks(np.linspace(0, 3.14, 5))
        ax.set_xticklabels([0, 45, 90, 135, 180])
        ax.set_xlabel(r'$\theta(°)$', fontsize = 16)
        ax.set_ylabel('Spikes', fontsize = 16)
        ax.spines['left'].set_visible(True)
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.round(yticks,1))
        ax.tick_params(axis='both', which='major', labelsize=13)
    fig.subplots_adjust(wspace=0.05, hspace=0.3)
    
    return fig

# --------------------------------------------------------------
# Getting the info for the clustering in dynamical terms
# --------------------------------------------------------------
def return_cluster_delay(cluster) :
    load_folder = prm.grouping_path + '/' + cluster
    sequences_contents = np.load(load_folder + '/sequences_contents.npy', allow_pickle = True)
    spiketimes = np.load(load_folder + '/spiketimes.npy')
    nmmeans = np.load(load_folder + '/TC.npy').mean(axis = -1)
    pref_ori = np.argmax(nmmeans[-1,:])
    
    # NKR ---------------------------------
        
    try :
        CVs = np.load(load_folder + '/cirvar.npy')
        fit_cv, _ = fit_nkr(CVs)
        
        HWHHs = utils.norm_data(np.load(load_folder + '/hwhh.npy'))
        fit_hwhh, _ = fit_nkr(HWHHs)
        HWHHs = np.load(load_folder + '/hwhh.npy')
        
        Rmaxs = utils.norm_data(np.load(load_folder + '/rmax.npy'))
        fit_rmaxs, _ = fit_nkr(Rmaxs)
        Rmaxs = np.load(load_folder + '/rmax.npy')
    except ValueError :
        return None

    
    # Max values of TC temporal evolution ---------------------------------
    latencies = np.linspace(0.05, .4, 61)
    window_size = .1
    spiketimes = np.load(load_folder + '/spiketimes.npy')
    spiketimes = spiketimes / prm.fs

    # For each b_theta (in decreasing order)
    full_array = np.zeros((len(prm.B_thetas), len(latencies), len(prm.thetas)))
    for i_bt, bt in enumerate(prm.B_thetas) :
        # For each latency
        for i_lat, lat in enumerate(latencies) :
            # We recompute a tuning curve (sequences where bt and t match)
            tc_array = np.zeros(len(prm.thetas)) # array of size 12 (number of thetas)
            for seq in sequences_contents : 
                if seq['sequence_btheta'] == bt :
                    seq_beg = (seq['sequence_beg'] / prm.fs) + lat
                    seq_end = seq_beg + window_size
                    spikes_in = np.where((spiketimes >= seq_beg) & (spiketimes <= seq_end))
                    spikes_in = spikes_in[0]
                    seq_theta_idx = np.where(np.round(seq['sequence_theta'], 5)== np.round(prm.thetas, 5))[0][0]
                    tc_array[seq_theta_idx] += len(spikes_in)

            # And we renormalize by the number of time stimulations were shown
            tc_array = tc_array/30
            full_array[i_bt, i_lat, :] = tc_array
    
    cv_array = np.zeros((len(prm.B_thetas), len(latencies)))
    for i_bt, bt in enumerate(prm.B_thetas) :
        for i_lat, lat in enumerate(latencies) :
            
            arr = full_array[i_bt, i_lat, :]
            R = np.sum( (arr * np.exp(2j*prm.thetas)) / np.sum(arr) )
            cirvar = 1 - np.abs(np.real(R))
            cv_array[i_bt, i_lat] = cirvar
            
    full_array_norm = utils.norm_data(full_array) #normalizing is convenient for plotting
    
            
    delays = np.zeros(len(prm.B_thetas))
    for ibt, bt in enumerate(prm.B_thetas) :
        delays[ibt] = np.argmax(full_array_norm[ibt,:,pref_ori])
    
    baselines = np.load(load_folder + '/baseline.npy')
    
    
    return {'cluster' : cluster,
        'all_delays' : delays,
        'cv' : [CVs, fit_cv],
        'hwhh' : [HWHHs, fit_hwhh],
        'rmax' : [Rmaxs, fit_rmaxs],
        'cv_array' : cv_array, 
        'full_array':full_array,
        'baseline' : baselines}
    

# --------------------------------------------------------------
# Refitting NKRs
# --------------------------------------------------------------  
'''
def fit_nkr(array) :
    x = np.linspace(0, 1, len(array))
    y = array[::-1]
    
    mod = Model(nakarushton)
    pars = Parameters()
    
    pars.add_many(('rmax',  np.max(y), True,  0.0,  100.),
            ('c50', .5, True,  0.001, 1.),
            ('b', y[0], True, y[0] * .5 + .001, y[0] * 3 + .002 ),
            ('n', 4, True,  1., 100.))
    
    out = mod.fit(y, pars, x=x, nan_policy='omit', fit_kws = {'maxfev' : 2000})
    return out.best_values, np.abs(1-out.residual.var() / np.var(y))
'''

def fit_nkr(array) :
    x = np.linspace(0, 1, len(array))
    y = array[::-1]
    
    mod = Model(nakarushton)
    pars = Parameters()
    
    pars.add_many(('rmax',  np.max(y), True,  0.0,  200.),
              ('c50', .5, True,  0.001, 1.),
              ('b', y[0], True, y[0] * .25 + .001, 1.), # this will make HWHH fit fails, but we're only doing cv
              ('n', 5., True,  1., 300.))
    
    out = mod.fit(y, pars, x=x, nan_policy='omit', max_nfev = 3000)
    return out.best_values, np.abs(1-out.residual.var() / np.var(y))

def nakarushton(x, rmax, c50, b, n):
    nkr = b + (rmax-b) * ((x**n) / (x**n + c50**n))
    return nkr

# --------------------------------------------------------------
# Plotting NKR
# -------------------------------------------------------------- 
def make_NKR_monoaxis(figsize, filename,
                      fit, raw, 
                      B_thetas, sharey = False, dtype = 'CV'):
    lw = 3
    s = 100
    fig, ax = plt.subplots(figsize=figsize, nrows = 1)
    
    # CV
    nkr = nakarushton(np.linspace(0, 1, 1000),
                    fit['rmax'], fit['c50'], fit['b'], fit['n'])
    
    if dtype == 'Rmax' :
        nkr = nkr[::-1]
        raw = raw[::-1]
    ax.plot(np.linspace(np.min(B_thetas), np.max(B_thetas), 1000),
            nkr,
            c=r'#3C5488FF', linewidth=lw, linestyle=(0, (3, 1)), zorder=5)
    ax.scatter(B_thetas, raw,
            facecolor=prm.colors, edgecolor='k', marker='o', s=s, zorder=20)
    
    ax.set_xticks(np.round(np.linspace(np.min(B_thetas), np.max(B_thetas),
                        3, endpoint = True), 1 ))

    
    if sharey and dtype == 'CV':
        ax.set_ylim(0., 1.)
        ax.set_yticks([0,.5, 1.])
    elif sharey and dtype == 'HWHH': 
        ax.set_ylim(0., 1.)
        ax.set_yticks([0., 25., 50.])
    elif sharey and dtype == 'HWHH': 
        ax.set_ylim(0., 1.)
        ax.set_yticks([0., 15., 30.])

    else :
        ax.set_ylim(np.min(raw) - np.min(raw) * .1,
                    np.max(raw) + np.max(raw) * .1,)
        
    # This is hard computed in the introduction notebook
    # Todo --> reload as numpy array before I get shamed on Twitter for declaring variables like this 
    cv_curve = [0.,0., 0.0026135727270204523, 0.0057877038576553685, 0.009690208927124133, 0.014640185595436117, 0.020417601991002066, 0.0269728956681059,
                0.03403086863369931, 0.041416385078155926, 0.04925638494139084, 0.05760349841107404, 0.06625823701255817, 0.07521972588477788, 0.08434919503779725,
                0.09379686802676734, 0.10359748357483789, 0.11378625792364405, 0.12420552127766449, 0.13512674211485098, 0.14632399000462026, 0.15787354758555805,
                0.16985650726133705, 0.182323931793001, 0.1951130630826412, 0.20819009284054757, 0.22177355921600117, 0.23574070846375073, 0.2501005025072105, 0.2648372446375009,
                0.27975235171988444, 0.2951226944414421, 0.31086294412078885, 0.3266420851406142, 0.34274895137324, 0.3589905761007487, 0.37519598353791883, 0.3913666534186441,
                0.40769315601946465, 0.4238504807230977, 0.439816107394586, 0.4554524829682177, 0.47064014388757347, 0.4858122701238168, 0.5007039115101781, 0.5148959078426101,
                0.5289495331291763, 0.5426001885127332, 0.5558817407989594, 0.568620290279629]

    ax.plot(np.linspace(np.min(B_thetas), np.max(B_thetas), len(cv_curve)),
        cv_curve, 
        c = 'k', alpha = .5)
    ax.set_xlabel(r'B$_\theta (°)$', y=.85, fontsize=16)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    fig.savefig('./figs/sn_NKR_%s_%s.pdf' %  (filename,dtype),
                bbox_inches='tight', dpi=200, transparent=True)

    plt.show(block = prm.block_plot)
