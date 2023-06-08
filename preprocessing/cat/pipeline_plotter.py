#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""


import numpy as np

import os, glob 
import imp
import shutil

from tqdm import tqdm

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from PyPDF2 import PdfMerger

from joblib import Parallel, delayed

import params as prm
import pipeline_mc as mc
import pipeline_fit as fit 


def create_ID_card():
    '''
    Creates a full report on the cluster, merging multiple matplotlib fig saved as _tmp
    '''
    print('Plotting and saving pdfs...')
    for folder in prm.folder_list : 
        folder_path = './processed/%s/' % folder
        clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
        
        clusters_iter = range(0, len(clusters_folders))
        Parallel(n_jobs = prm.n_cpu)(delayed(make_id)(i, clusters_folders = clusters_folders, folder_path = folder_path,
                folder = folder) for i in tqdm(clusters_iter))
   

# --------------------------------------------------------------
# 
# --------------------------------------------------------------
        
        
def make_id(i, clusters_folders, folder_path, folder) :
    
    cluster_folder = clusters_folders[i]
    cluster_path = folder_path + cluster_folder
    
    for i0, phase in enumerate(prm.phases) :
        pagenum = 0
        make_pg1(folder = folder, cluster_folder = cluster_path, phase = phase, pagenum = pagenum)
        pagenum += 1
        
        make_pg2(cluster_folder = cluster_path, phase = phase, pagenum = pagenum)
        pagenum += 1
        
        make_recap_psth(cluster_folder = cluster_path, phase = phase, pagenum = pagenum)
        pagenum +=1
        
        make_pg5(cluster_folder = cluster_path, phase = phase, pagenum = pagenum) 
        pagenum += 1
    
        for i1, btheta in enumerate(prm.B_thetas) : #n is the page number, btheta idx in the btheta index in the unique values
            make_PSTH_pages(cluster_folder = cluster_path, b_theta_idx = i1, n = i1+4, phase = phase, pagenum = pagenum)
            pagenum += 1
    
        pdfs = [cluster_path + '/tmp%s.pdf' % x for x in np.arange(0,pagenum)]
        merger = PdfMerger()
    
        for pdf in pdfs:
            merger.append(pdf)
        
        merger.write(cluster_path + 'ID%sphase_card.pdf' % i0)
        merger.close()
        
        for filename in glob.glob(cluster_path + '/tmp*'):
            os.remove(filename) 
            
        shutil.copyfile('%sID0phase_card.pdf' % (cluster_path),
                        '%s/recap.pdf' % (cluster_path))
            
    print('Done !')
              
    
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def make_pg1(folder, cluster_folder, phase, pagenum):
    '''
    Creates the first page, including cluster infos and waveform + spike stability
    '''
    cluster_info = get_var_from_file(cluster_folder + '/cluster_info.py')

    r_squareds = np.load(cluster_folder + '/%.3f_plot_neurometric_fit_reports.npy'% phase)

    spiketimes = np.load(cluster_folder + '/spiketimes.npy')

    unique_thetas = prm.thetas *  180 / np.pi
    unique_thetas = np.round(unique_thetas, 1)
    unique_bthetas = prm.B_thetas *  180 / np.pi
    
    colors = prm.colors
    
    # Page 1 tuning curve are stacked on the phase dimension
    fitted_TC = np.load(cluster_folder + '/%.3f_plot_neurometric_fitted_TC.npy' % phase)
    
    fig = plt.figure(figsize = (12,9))
    fig.tight_layout()
    gs = gridspec.GridSpec(2,2)
    axs1 = plt.subplot(gs[0, 0])
    axs2 = plt.subplot(gs[0, 1])
    axs3 = plt.subplot(gs[-1, 0])
    axs4 = plt.subplot(gs[-1, 1])
    
    # Waveform classification
    axs1.plot(unique_bthetas, r_squareds, c = 'k')
    

    # Spike density
    axs3.hist(spiketimes, int(np.max(spiketimes) /(.5 *prm.fs)), color = 'gray')
    
    axs1.set_title('r² for each btheta - max r² = %.2f' % np.max(r_squareds))
    axs1.set_ylabel('r²')
    axs1.set_xlabel(r'$B_\theta$')
    
    axs2.set_title('Average waveform and std')
    axs2.set_ylabel('Amplitude')
    axs2.set_xlabel('Time (sample)')
    
    axs3.set_title('Neuron activity over time of recording')
    axs3.set_ylabel('Firing rate (spk/s)')
    axs3.set_xlabel('Time (s)')
    
    for i, tc in enumerate(fitted_TC) : 
        axs4.plot(np.linspace(0, np.max(unique_thetas), len(tc)),
                  tc,
                  color = colors[i], label = '%.1f' % unique_bthetas[i])
    axs4.legend(title = r'$B_\theta$', loc = (1, .35))
    axs4.set_title('TC for each Btheta')

    axs4.set_ylabel('Spikes')
    axs4.set_xlabel('Angle °')


    plt.text(0.65, .95, 'Channel ID : %s' % cluster_info.channel_id, 
             fontsize = 15, transform = plt.gcf().transFigure)
    plt.text(0.65, .92, 'Channel depth : %s' % cluster_info.channel_depth,
             fontsize = 15, transform = plt.gcf().transFigure)
    
             
    fig.savefig(cluster_folder + '/tmp%s.pdf' % pagenum)
    plt.close(fig) # We don't want display

    
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def make_pg2(cluster_folder, phase, pagenum):
    '''
    Creates the scond page, with merged and non merged tuning curve, as well
    as the neurometric curve (HWHH, Cirvar, Rmaxs)
    '''

    unique_thetas = prm.thetas *  180 / np.pi
    unique_thetas = np.round(unique_thetas, 1)
    
    unique_bthetas = prm.B_thetas *  180 / np.pi
        
    fig = plt.figure(figsize = (11,17))
    fig.tight_layout()
    
    gs = gridspec.GridSpec(12,7)
    
    axs1 = plt.subplot(gs[0:2, :3]) # merged TC
    axs2 = plt.subplot(gs[0:2, 3:-1]) # all tc, as page 1
    axs3 = plt.subplot(gs[4:6, :3]) # phi curve
    axs4 = plt.subplot(gs[6:8, :3]) # cirvar curve
    axs5 = plt.subplot(gs[8:10, :3]) # r^² curve
    
    multiax = plt.subplot(gs[3:11, 3:-1]) # b theta 4 to

    colors = prm.colors

    # Merged TC plot
    mean_FR_per_theta = np.load(cluster_folder + '/%.3f_plot_MC_TC_merged_means.npy' % phase)
    r_squares = np.load(cluster_folder + '/%.3f_plot_neurometric_fit_reports.npy' % phase)
    cirvars = np.load(cluster_folder + '/%.3f_cirvar.npy' % phase)
    bthetas_fit = np.load(cluster_folder + '/%.3f_plot_neurometric_Btheta_fits.npy' % phase)
    cv_fit = np.load(cluster_folder + '/%.3f_cirvar_fit.npy' % phase, allow_pickle = True)
    cv_fit = cv_fit.item()
    phi_fit = np.load(cluster_folder + '/%.3f_phi_fit.npy' % phase, allow_pickle = True)
    phi_fit = phi_fit.item()
    r2_fit = np.load(cluster_folder + '/%.3f_r2_fit.npy' % phase)
   
    
    fitted_merge_curve = np.load(cluster_folder + '/%.3f_plot_neurometric_merged_fitted_TC.npy' % phase)
    btheta_merge_curve = np.load(cluster_folder + '/%.3f_plot_neurometric_merged_Btheta_fits.npy' % phase)
    r_squared_merge_curve = np.load(cluster_folder + '/%.3f_plot_neurometric_merged_fit_reports.npy' % phase)
    
    # Non merged TC
    mean_FR_per_btheta = np.load(cluster_folder + '/%.3f_plot_MC_TC_nonmerged_means.npy' % phase)
    fitted_TC = np.load(cluster_folder + '/%.3f_plot_neurometric_fitted_TC.npy' % phase)
    Rmaxs = np.max(fitted_TC, axis = 1)
    fit_rmax = np.load(cluster_folder + '/%.3f_rmax_fit.npy' % phase, allow_pickle = True)
    fit_rmax = fit_rmax.item()
    r2_rmax = np.load(cluster_folder + '/%.3f_rmax_fitr2.npy' % phase)

    max_FR = np.max(mean_FR_per_btheta)*1.1
    min_FR = np.min(mean_FR_per_btheta)*.9
    
    # Delay
    delta_t = np.load(cluster_folder +  '/optimal_delay.npy')


    xs = np.linspace(0, np.max(unique_thetas), len(fitted_merge_curve[0]))
    axs1.plot(xs, fitted_merge_curve[0])
    axs1.plot(unique_thetas, mean_FR_per_theta, '.k')
    axs1.set_xlabel(r'$\theta$' + '°')
    axs1.set_ylabel('n spikes')
    axs1.set_title('Tuning curve, averaged over all ' + r'$B_\theta$' +'\n' + r'$\varphi$ = %.2f' % btheta_merge_curve[0] + '    r² = %.2f' % r_squared_merge_curve[0],
     fontsize = 10)
    axs1.set_ylim(min_FR, max_FR)
    
    
    for i, tc in enumerate(fitted_TC) : 
        xs = np.linspace(0, np.max(unique_thetas), len(tc))
        axs2.plot(xs, tc, color = colors[i], label = '%.1f' % unique_bthetas[i])
    axs2.set_xlabel(r'$\theta$' + '°')
    axs2.set_ylabel('n spikes')
    axs2.yaxis.set_label_position('right')
    axs2.yaxis.tick_right()
    axs2.set_title('All tuning curves, fitting method : %s \n Spikes on [%.2f;%.2f]s interval'% (prm.fit_type,
                                                                                                 delta_t,
                                                                                                 delta_t + prm.delta_ts[-1]), fontsize = 12)
    axs2.legend(title = r'$B_\theta$', fontsize = 8)
    
    multiax.set_title('Wilcoxon raw OSI vs baseline OSI \n * < .05, ** < .005, *** < .0001', fontsize = 12)
    for i, btheta in enumerate(unique_bthetas) :
        tc = fitted_TC[i]
        
        xs = np.linspace(0, np.max(unique_thetas), len(fitted_TC[i]))
        
        multiax.plot(xs, tc+ i * 10, c = 'k', zorder = -i-1)
        multiax.scatter(unique_thetas, mean_FR_per_btheta[i] + i * 10, facecolor = colors[i], zorder = -.5*i ,edgecolor = 'k',
                        s = 10)
        multiax.fill_between(x = xs,
                          y1 = tc + i * 10,
                          y2 = np.full_like(xs, np.min(tc)) + i * 10,
                          color = colors[i], zorder = -i-1)
        

    multiax.set_yticks([])
    multiax.set_xlabel(r'$\theta$' + '°')
    
        
    # phi
    axs3.scatter(unique_bthetas, bthetas_fit, facecolor = r'#00a087', edgecolor = 'k', marker = 'o', s = 10, zorder = 20)
    nkr = fit.nakarushton(np.linspace(0, 1, 1000),
                      phi_fit['rmax'], phi_fit['c50'], phi_fit['b'] ,phi_fit['n'])
    axs3.plot(np.linspace(np.min(unique_bthetas), np.max(unique_bthetas), 1000),
              nkr,
              c = r'#00a087', linewidth = 2, linestyle = (0, (3,1)), zorder = 5)
    axs3.set_xticklabels([])
    axs3.set_title(r'$\varphi$',y = .85, fontsize = 14)
    axs3.text(x = 0, y = np.max(bthetas_fit) + .05 * np.max(bthetas_fit), 
              s = r'Low B$_\theta$                                        High B$_\theta$',
              fontsize = 12)

    # cirvar
    axs4.scatter(unique_bthetas, cirvars, facecolor = r'#2ea7c2', edgecolor = 'k', marker = 'o', s = 10, zorder = 20)
    nkr = fit.nakarushton(np.linspace(0, 1, 1000),
                      cv_fit['rmax'], cv_fit['c50'], cv_fit['b'] , cv_fit['n'])
    axs4.plot(np.linspace(np.min(unique_bthetas), np.max(unique_bthetas), 1000),
              nkr,
              c = r'#2ea7c2', linewidth = 2, linestyle = (0, (3,1)), zorder = 5)
    axs4.set_xticklabels([])
    axs4.set_title('CirVar', y = .85, fontsize = 14)
    
    # rmax
    axs5.scatter(unique_bthetas,Rmaxs, c = 'k', marker = 'o', s = 20)
    # relu = fit.ReLU_t(np.linspace(-1, 1, 1000),
    #                   slope = fit_rmax['slope'], threshold = fit_rmax['threshold'],
    #                   baseline = fit_rmax['baseline'])[::-1]
    # axs5.plot(np.linspace(np.min(unique_bthetas), np.max(unique_bthetas), 1000),
    #           relu,
    #           c = 'k', linewidth = 2, linestyle = (0, (3,1)), zorder = 5)
    nkr = fit.nakarushton(np.linspace(0, 1, 1000),
                          fit_rmax['rmax'], fit_rmax['c50'], fit_rmax['b'] , fit_rmax['n'])[::-1]
    axs5.plot(np.linspace(np.min(unique_bthetas), np.max(unique_bthetas), 1000),
              nkr,
              c = 'k', linewidth = 2, linestyle = (0, (3,1)), zorder = 5)
    axs5.set_title(r'$R_{max}$', y = .85, fontsize = 14)
    axs5.set_xlabel(r'B$_\theta$', y = .85, fontsize = 14)
    

    fig.savefig(cluster_folder + '/tmp%s.pdf' % pagenum, bbox_inches = 'tight')
    plt.close(fig) # We don't want display


# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def make_recap_psth(cluster_folder, phase, pagenum) :
    '''
    Creates the third page, with orth/pref psths
    '''
    
    nmmeans = np.load(cluster_folder + '/%.3f_plot_MC_TC_nonmerged_means.npy' % phase)
    cluster_info = get_var_from_file(cluster_folder + '/cluster_info.py')

    prefered_id = np.argmax(nmmeans[-1,:])
    orth_id = np.argmin(nmmeans[-1,:])
    
    n_bin = (prm.end_psth) - (prm.beg_psth) 
    n_bin*=1000
    n_bin/= prm.binsize
    
    colors = prm.colors
    
    fig = plt.figure(figsize = (11,17))
    fig.tight_layout()
    gs = gridspec.GridSpec(65,9)
    
    #Merged raster, pref
    axs1 = plt.subplot(gs[2:4, :4]) #
    PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_merged.npy' % phase, allow_pickle = True)
    if prm.substract_baseline :
        baseline = np.load(cluster_folder + '/baseline.npy')
    
    if prm.substract_baseline : 
        hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
    else :
        hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
                 
    theta = PSTH_list[prefered_id, :]
    for it_1, trial in enumerate(theta) :
        axs1.scatter(trial, np.full_like(trial, it_1), s =  .3,
                               color = 'C0')
    axs1.set_xlabel('PST (s)')
    axs1.set_ylabel('Trial')
    
    #Merged PSTH, pref
    axs1_0 = plt.subplot(gs[0:2, :4]) #
    if prm.substract_baseline :
        baseline = np.load(cluster_folder + '/baseline.npy')
    
    if prm.substract_baseline : 
        hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
    else :
        hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
                 
    max1_0, min1_0 = np.max(hists), np.min(hists)
    theta = PSTH_list[prefered_id, :]
    axs1_0.hist(np.concatenate(theta), np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)), color = 'C0', edgecolor = 'C0')
    axs1_0.set_ylabel('sp/bin')
    axs1_0.set_xticks([])
    
    
    # Merged raster, orth
    axs1 = plt.subplot(gs[2:4, 5:]) # merged PSTH
    if prm.substract_baseline :
        baseline = np.load(cluster_folder + '/baseline.npy')
    
    if prm.substract_baseline : 
        hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
    else :
        hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
                 
    theta = PSTH_list[orth_id, :]
    for it_1, trial in enumerate(theta) :
        axs1.scatter(trial, np.full_like(trial, it_1), s =  .3,
                               color = 'C0')
    axs1.set_yticklabels([])
    axs1.set_xlabel('PST (s)')
    axs1.set_ylabel(r'All B$_\theta$' + '\nmerged', rotation = 0, labelpad = 35, fontsize = 14)
    
    # Merged PSTH, orth
    axs1_1 = plt.subplot(gs[0:2, 5:]) #
    if prm.substract_baseline :
        baseline = np.load(cluster_folder + '/baseline.npy')
    
    if prm.substract_baseline : 
        hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
    else :
        hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
                 
    max1_1, min1_1 = np.max(hists), np.min(hists)
    theta = PSTH_list[orth_id, :]
    axs1_1.hist(np.concatenate(theta), np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)), color = 'C0', edgecolor = 'C0')
    axs1_1.set_yticklabels([])
    axs1_1.set_xticks([])
    
    min_y, max_y = np.min([min1_0, min1_1]), np.max([max1_0, max1_1]) +2
    axs1_0.set_ylim(min_y, max_y)
    axs1_1.set_ylim(min_y, max_y)
    axs1_0.plot((0.,.35), (max_y+3, max_y+3),
                clip_on = False, zorder = 100, c = 'k', linewidth = 2)
    axs1_1.plot((0.,.35), (max_y+3, max_y+3),
                clip_on = False, zorder = 100, c = 'k', linewidth = 2)
    axs1_0.text(.12, max_y + 4.2, 'stimulation', fontsize = 6)
    axs1_1.text(.12, max_y + 4.2, 'stimulation', fontsize = 6)
    
    
    # Pref orientation, raster
    y0 = 8
    pref_axs = []
    for nbt, _ in enumerate(prm.B_thetas) :
        ax = plt.subplot(gs[y0 + (5*nbt) : y0 + 2 + (5*nbt), :4])
        pref_axs.append(ax)
    
    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] #select only one specific btheta
        if prm.substract_baseline :
            baseline = np.load(cluster_folder + '/baseline.npy')
            
        if prm.substract_baseline : 
            hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
        else :
            hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
            
        theta = PSTH_list[prefered_id, :]
        for it_1, trial in enumerate(theta) :
            pref_axs[it_0].scatter(trial, np.full_like(trial, it_1), s =  .3,
                                   color = colors[it_0])
            
        if it_0 != len(prm.B_thetas) - 1 :
            pref_axs[it_0].set_xticklabels([])
            pref_axs[it_0].set_yticklabels([])
            pref_axs[it_0].set_yticks([])
        else :
            pref_axs[it_0].set_xlabel('PST (s)')
            pref_axs[it_0].set_ylabel('Trial')
        
    
    # Pref orientation, PSTH
    y0 = 6
    pref_axs = []
    for nbt, _ in enumerate(prm.B_thetas) :
        ax = plt.subplot(gs[y0 + (5*nbt) : y0 + 2 + (5*nbt), :4])
        pref_axs.append(ax)
    
    mins, maxs = [], []
    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] # select only one specific btheta
        if prm.substract_baseline :
            baseline = np.load(cluster_folder + '/baseline.npy')
        
        if prm.substract_baseline : 
            hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
        else :
            hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
                     
        theta = PSTH_list[prefered_id, :]
        pref_axs[it_0].hist(np.concatenate(theta), np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)),
                            color = colors[it_0], edgecolor = colors[it_0])   
        
        pref_axs[it_0].set_ylim(min_y, max_y)         
            
        if it_0 != len(prm.B_thetas) - 1 :
            pref_axs[it_0].set_xticks([])
            pref_axs[it_0].set_yticklabels([])
            pref_axs[it_0].set_yticks([])
        else :
            pref_axs[it_0].set_xticks([])
            pref_axs[it_0].set_ylabel('sp/bin')
        
        mins.append(np.min(hists))
        maxs.append(np.max(hists))
    
    for it_0, btheta in enumerate(prm.B_thetas) :
        pref_axs[it_0].set_ylim(np.min(mins), np.max(maxs) + 2)
        pref_axs[it_0].plot((0.,.35), (np.max(maxs)+3, np.max(maxs)+3),
                clip_on = False, zorder = 100, c = 'k', linewidth = 2) #adjust for time latency with .35 instead of .3
    
    
    # Orth orientation, raster
    y0 = 8
    pref_axs = []
    for nbt, _ in enumerate(prm.B_thetas) :
        ax = plt.subplot(gs[y0 + (5*nbt) : y0 + 2 + (5*nbt), 5:])
        pref_axs.append(ax)
    
    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] #select only one specific btheta
        if prm.substract_baseline :
            baseline = np.load(cluster_folder + '/baseline.npy')
            
        if prm.substract_baseline : 
            hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
        else :
            hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
            
        theta = PSTH_list[orth_id, :]
        for it_1, trial in enumerate(theta) :
            pref_axs[it_0].scatter(trial, np.full_like(trial, it_1), s =  .3,
                                   color = colors[it_0])
            
            
        if it_0 != len(prm.B_thetas) - 1 :
            pref_axs[it_0].set_xticklabels([])
            pref_axs[it_0].set_yticklabels([])
        else :
            pref_axs[it_0].set_xlabel('PST (s)')
            pref_axs[it_0].set_yticklabels([])

    # Orth orientation, PSTH
    y0 = 6
    pref_axs = []
    for nbt, _ in enumerate(prm.B_thetas) :
        ax = plt.subplot(gs[y0 + (5*nbt) : y0 + 2 + (5*nbt), 5:])
        pref_axs.append(ax)

    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] #select only one specific btheta
        if prm.substract_baseline :
            baseline = np.load(cluster_folder + '/baseline.npy')
        
        if prm.substract_baseline : 
            hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
        else :
            hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
                     
        theta = PSTH_list[orth_id, :]
        pref_axs[it_0].hist(np.concatenate(theta), np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)),
                            color = colors[it_0], edgecolor = colors[it_0])   
        pref_axs[it_0].set_ylim(min_y, max_y)         
            

        pref_axs[it_0].set_xticks([])
        pref_axs[it_0].set_yticklabels([])
        
        pref_axs[it_0].set_ylabel(r'B$_\theta$ =%.1f°' % (btheta * 180 / np.pi) , rotation = 0, labelpad = 35, fontsize = 14)
        pref_axs[it_0].yaxis.set_label_coords(-0.15,-.38)

    for it_0, btheta in enumerate(prm.B_thetas) :
        pref_axs[it_0].set_ylim(np.min(mins), np.max(maxs) + 2)
        pref_axs[it_0].plot((0.,.35), (np.max(maxs)+3, np.max(maxs)+3),
                clip_on = False, zorder = 100, c = 'k', linewidth = 2) #adjust for time latency with .35 instead of .3

        
    plt.suptitle('Pref. orientation             (%sms bin size)          Orth. orientation' % prm.binsize,  y = .9, fontsize = 15 )
    
    fig.savefig(cluster_folder + '/tmp%s.pdf' % pagenum, bbox_inches = 'tight')
    plt.close(fig)
    
    

# --------------------------------------------------------------
# 
# --------------------------------------------------------------
def make_fit_psth(cluster_folder, phase, pagenum) :
    '''
    Creates the fourth page, with orth/pref psth fits (no raster)
    '''
    
    # Reloading
    nmmeans = np.load(cluster_folder + '/%.3f_plot_MC_TC_nonmerged_means.npy' % phase)
    cluster_info = get_var_from_file(cluster_folder + '/cluster_info.py')
    
    PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_merged.npy' % phase, allow_pickle = True)
    
    PSTH_models_pref = np.load(cluster_folder + '/%.3f_psth_fit_models_pref.npy' % phase, allow_pickle = True)
    PSTH_models_pref_params = np.load(cluster_folder + '/%.3f_psth_fit_params_pref.npy' % phase, allow_pickle = True)
    
    PSTH_models_orth = np.load(cluster_folder + '/%.3f_psth_fit_models_orth.npy' % phase, allow_pickle = True)
    PSTH_models_orth_params = np.load(cluster_folder + '/%.3f_psth_fit_params_orth.npy' % phase, allow_pickle = True)

    # Recomputing and aliasing
    prefered_id = np.argmax(nmmeans[-1,:])
    orth_id = np.argmin(nmmeans[-1,:])
    
    n_bin = (prm.end_psth) - (prm.beg_psth) 
    n_bin*=1000
    n_bin/= prm.binsize
    
    colors = prm.colors
    
    
    # Figure
    fig = plt.figure(figsize = (11,17))
    fig.tight_layout()
    gs = gridspec.GridSpec(65,9)
    
    # Pref orientation, PSTH    
    y0 = 6
    pref_axs = []
    for nbt, _ in enumerate(prm.B_thetas) :
        ax = plt.subplot(gs[y0 + (5*nbt) : y0 + 2 + (5*nbt), :4])
        pref_axs.append(ax)
    
    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] #select only one specific btheta
                             
        theta = np.concatenate(PSTH_list[prefered_id, :])
        hist = np.histogram(theta, np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)))[0]
        hist = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
        x = np.linspace(prm.beg_psth, prm.end_psth, len(hist))
        pref_axs[it_0].bar(x = x, height = hist, width = x[1] - x[0],
                           facecolor = colors[it_0], edgecolor = colors[it_0])
        pref_axs[it_0].plot(x, PSTH_models_pref[it_0])
        
        best_str = ['%s = %.3f ; ' % (key,val) for key,val in PSTH_models_pref_params[it_0].items()]
        best_str = ''.join(best_str)
        best_str += 'ttp = %.2fs' % (x[np.argmax(PSTH_models_pref[it_0])])
        best_str = best_str.replace('; sigma', ';\nsigma')
        pref_axs[it_0].text(x = 0, y = -.95, s = best_str, transform = pref_axs[it_0].transAxes,
                            fontsize = 8)
        
        pref_axs[it_0].set_ylim(0, 1.1)         
            
        if it_0 != len(prm.B_thetas) - 1 :
            pref_axs[it_0].set_xticklabels([])
            pref_axs[it_0].set_yticklabels([])
            pref_axs[it_0].set_yticks([])
        else :
            pref_axs[it_0].set_xticklabels([])
            pref_axs[it_0].set_ylabel('norm.\nbins')
            #pref_axs[it_0].set_xlabel('PST (s)')

    for it_0, btheta in enumerate(prm.B_thetas) :
        pref_axs[it_0].plot((0.,.35), (1.1, 1.1),
                clip_on = False, zorder = 100, c = 'k', linewidth = 2) #adjust for time latency with .35 instead of .3
    


    # Orth orientation, PSTH
    y0 = 6
    pref_axs = []
    for nbt, _ in enumerate(prm.B_thetas) :
        ax = plt.subplot(gs[y0 + (5*nbt) : y0 + 2 + (5*nbt), 5:])
        pref_axs.append(ax)

    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] #select only one specific btheta

                 
        theta = np.concatenate(PSTH_list[orth_id, :])
        hist = np.histogram(theta, np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)))[0]
        hist = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
        x = np.linspace(prm.beg_psth, prm.end_psth, len(hist))
        pref_axs[it_0].bar(x = x, height = hist, width = x[1] - x[0],
                           facecolor = colors[it_0], edgecolor = colors[it_0])
        pref_axs[it_0].plot(x,PSTH_models_orth[it_0])
        
        best_str = ['%s = %.3f ; ' % (key,val) for key,val in PSTH_models_orth_params[it_0].items()]
        best_str = ''.join(best_str)
        best_str += 'ttp = %.2fs' % (x[np.argmax(PSTH_models_orth[it_0])])
        best_str = best_str.replace('; sigma', ';\nsigma')
        pref_axs[it_0].text(x = 0, y = -.95, s = best_str, transform = pref_axs[it_0].transAxes,
                            fontsize = 8)
        
        pref_axs[it_0].set_ylim(0, 1.1) 

        pref_axs[it_0].set_ylabel(r'B$_\theta$ =%.1f°' % (btheta * 180 / np.pi) , rotation = 0, labelpad = 35, fontsize = 14)
        pref_axs[it_0].yaxis.set_label_coords(-0.15,0.)
        
        if it_0 != len(prm.B_thetas) - 1 :
            pref_axs[it_0].set_xticklabels([])
            pref_axs[it_0].set_yticklabels([])
            pref_axs[it_0].set_yticks([])
        else :
            pref_axs[it_0].set_xticklabels([])
            pref_axs[it_0].set_yticks([])

    for it_0, btheta in enumerate(prm.B_thetas) :
        pref_axs[it_0].plot((0.,.35), (1.1, 1.1),
                clip_on = False, zorder = 100, c = 'k', linewidth = 2) #adjust for time latency with .35 instead of .3


    plt.suptitle('Pref. orientation             (%sms bin size)          Orth. orientation' % prm.binsize,  y = .83, fontsize = 15 )
    fig.savefig(cluster_folder + '/tmp%s.pdf' % pagenum, bbox_inches = 'tight')
    plt.close(fig)
    
    
def make_fit_psth_params(cluster_folder, phase, pagenum) :
    '''
    Plot the variations of the PSTH model parameters
    '''
    
    # Reloading
    nmmeans = np.load(cluster_folder + '/%.3f_plot_MC_TC_nonmerged_means.npy' % phase)
    cluster_info = get_var_from_file(cluster_folder + '/cluster_info.py')
    PSTH_models_pref = np.load(cluster_folder + '/%.3f_psth_fit_models_pref.npy' % phase, allow_pickle = True)
    PSTH_models_pref_params = np.load(cluster_folder + '/%.3f_psth_fit_params_pref.npy' % phase, allow_pickle = True)
    
    PSTH_models_orth = np.load(cluster_folder + '/%.3f_psth_fit_models_orth.npy' % phase, allow_pickle = True)
    PSTH_models_orth_params = np.load(cluster_folder + '/%.3f_psth_fit_params_orth.npy' % phase, allow_pickle = True)
    
    prefered_id = np.argmax(nmmeans[-1,:])
    orth_id = np.argmin(nmmeans[-1,:])
    n_bin = (prm.end_psth) - (prm.beg_psth) 
    n_bin*=1000
    n_bin/= prm.binsize
    
    fig, axs = plt.subplots(ncols = 2, nrows = len(PSTH_models_pref_params[0]) + 1, figsize = (9,15))
    fig.tight_layout()
    
    # Pref
    i0 = 0
    for k,v in PSTH_models_pref_params[0].items() :
        params_vals = [d[k] for d in PSTH_models_pref_params]
        
        axs[i0, 0].plot(prm.B_thetas * 180 / np.pi, params_vals)
        axs[i0, 0].set_title(k, fontsize = 14)
        axs[i0, 0].spines['right'].set_visible(False)
        axs[i0, 0].spines['top'].set_visible(False)
        axs[i0, 0].set_xticklabels([])
        
        i0 +=1
    
    ttps = []
    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] #select only one specific btheta

        theta = np.concatenate(PSTH_list[prefered_id, :])
        hist = np.histogram(theta, np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)))[0]
        hist = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
        x = np.linspace(prm.beg_psth, prm.end_psth, len(hist))
        ttps.append(x[np.argmax(PSTH_models_pref[it_0])])
    axs[-1, 0].plot(prm.B_thetas * 180 / np.pi, ttps)
    axs[-1, 0].set_title('Time to peak (s)', fontsize = 14)
    axs[-1, 0].spines['right'].set_visible(False)
    axs[-1, 0].spines['top'].set_visible(False)
    axs[-1, 0].set_xlabel(r'$B_\theta$')
    
    
    # Orth   
    i0 = 0
    for k,v in PSTH_models_orth_params[0].items() :
        params_vals = [d[k] for d in PSTH_models_orth_params]
        
        axs[i0, 1].plot(prm.B_thetas * 180 / np.pi, params_vals)
        axs[i0, 1].set_title(k, fontsize = 14)
        axs[i0, 1].spines['right'].set_visible(False)
        axs[i0, 1].spines['top'].set_visible(False)
        axs[i0, 1].set_xticklabels([])
            
        i0 +=1
        
    ttps = []
    for it_0, btheta in enumerate(prm.B_thetas) :
        PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
        PSTH_list = PSTH_list[it_0, : , :] #select only one specific btheta

        theta = np.concatenate(PSTH_list[orth_id, :])
        hist = np.histogram(theta, np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)))[0]
        hist = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
        x = np.linspace(prm.beg_psth, prm.end_psth, len(hist))
        ttps.append(x[np.argmax(PSTH_models_orth[it_0])])
    axs[-1, 1].plot(prm.B_thetas * 180 / np.pi, ttps)
    axs[-1, 1].set_title('Time to peak (s)', fontsize = 14)
    axs[-1, 1].spines['right'].set_visible(False)
    axs[-1, 1].spines['top'].set_visible(False)
    axs[-1, 1].set_xlabel(r'$B_\theta$')
        
    plt.suptitle('Pref. orientation                            Orth. orientation',  y = 1.02, fontsize = 15 )
    
    fig.savefig(cluster_folder + '/tmp%s.pdf' % pagenum, bbox_inches = 'tight')
    plt.close(fig)
    
# --------------------------------------------------------------
# 
# --------------------------------------------------------------    
def make_pg5(cluster_folder, phase, pagenum) :
    '''
    Creates the fifth page, with PSTH merged
    '''
    
    PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_merged.npy' % phase, allow_pickle = True)
    if prm.substract_baseline :
        baseline = np.load(cluster_folder + '/baseline.npy')
    
    n_bin = (prm.end_psth) - (prm.beg_psth) 
    n_bin*=1000
    n_bin/= prm.binsize
    
    ys = np.linspace(.85, .13, len(PSTH_list))
    colors = plt.cm.viridis(np.linspace(0, .8, len(PSTH_list)))
    
    fig, ax = plt.subplots(len(PSTH_list), 2, sharex = 'col', figsize = (12, 9))
    
    if prm.substract_baseline : 
        hists = [np.histogram(np.concatenate(theta)-baseline, int(n_bin))[0] for theta in PSTH_list]
    else :
        hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
        
    min_hist = np.min(hists)
    max_hist = np.max(hists)
    
    for it_0, theta in enumerate(PSTH_list) : 
        for it_1, trial in enumerate(theta) :
            ax[it_0][0].scatter(trial, np.full_like(trial, it_1), s =  .3,
                                color = colors[it_0])
            
        ax[it_0][0].axvline(0, c = 'gray')
        ax[it_0][1].axvline(0, c = 'gray')
        ax[it_0][1].hist(np.concatenate(theta), np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)), color = colors[it_0], edgecolor = colors[it_0])
        
        plt.text(0.02, ys[it_0], r'$\theta$' + '=%.1f°' % (prm.thetas[it_0]*180/np.pi), fontsize = 9, transform = plt.gcf().transFigure, 
                 color = colors[it_0])
        
        ax[it_0][0].set_xlim(prm.beg_psth, prm.end_psth)
        ax[it_0][0].set_ylim(0, len(theta))
        ax[it_0][1].set_xlim(prm.beg_psth, prm.end_psth)
        ax[it_0][1].set_ylim(min_hist, max_hist)
        
        if it_0 == len(PSTH_list)-1 :
                ax[it_0][0].set_xlabel('PST (s)')
                ax[it_0][0].set_ylabel('Trial')
                ax[it_0][1].set_xlabel('PST (s)')
                ax[it_0][1].set_ylabel('sp/bin')
        
    plt.suptitle('PSTH, All ' +  r'$B_\theta$' + ' merged\n Histogram with %sms bin size' % prm.binsize,  y = .95, fontsize = 15, )
    
    fig.savefig(cluster_folder + '/tmp%s.pdf' % pagenum, bbox_inches = 'tight')
    plt.close(fig)
  
  
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def make_PSTH_pages(cluster_folder,b_theta_idx, n, phase, pagenum):
    '''
    Creates the nth page, with PSTH non merged for decreasing Btheta
    '''
    
    # ------------
    # PSTH
    # ------------

    PSTH_list = np.load(cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, allow_pickle = True)
    PSTH_list = PSTH_list[b_theta_idx, : , :] #select only one specific btheta
    
    n_bin = (prm.end_psth) - (prm.beg_psth) 
    n_bin*=1000
    n_bin/= prm.binsize
    
    hists = [np.histogram(np.concatenate(theta), int(n_bin))[0] for theta in PSTH_list]
    min_hist = np.min(hists)
    max_hist = np.max(hists)
 
    ys = np.linspace(.85, .13, len(PSTH_list))
    colors = plt.cm.viridis(np.linspace(0, .8, len(PSTH_list)))
    
    fig, ax = plt.subplots(len(PSTH_list), 2, sharex = 'col', figsize = (12, 9))

    for it_0, theta in enumerate(PSTH_list) : 
        for it_1, trial in enumerate(theta) :
            ax[it_0][0].scatter(trial, np.full_like(trial, it_1), s =  .3,
                                color = colors[it_0])
            
        
        ax[it_0][0].axvline(0, c = 'gray')
        ax[it_0][1].axvline(0, c = 'gray')
        ax[it_0][1].hist(np.concatenate(theta), np.linspace(prm.beg_psth, prm.end_psth, int(n_bin)), color = colors[it_0],edgecolor = colors[it_0])
        
        plt.text(0.02, ys[it_0], r'$\theta$' + '=%.1f°' % (prm.thetas[it_0] * 180 / np.pi), fontsize = 9, transform = plt.gcf().transFigure, 
                 color = colors[it_0])
        
        ax[it_0][0].set_xlim(prm.beg_psth, prm.end_psth)
        ax[it_0][0].set_ylim(0, len(theta))
        ax[it_0][1].set_xlim(prm.beg_psth, prm.end_psth)
        ax[it_0][1].set_ylim(min_hist, max_hist+1)
        
        if it_0 == len(PSTH_list)-1 :
                ax[it_0][0].set_xlabel('PST (s)')
                ax[it_0][0].set_ylabel('Trial')
                ax[it_0][1].set_xlabel('PST (s)')
                ax[it_0][1].set_ylabel('sp/bin')
        
    plt.suptitle(r'$B_\theta$' + ' = %.1f°' % (prm.B_thetas[b_theta_idx] * 180 / np.pi) + '\nHistogram with %sms bin size' % prm.binsize,
                 y = .95, fontsize = 15, )
    
    fig.savefig(cluster_folder + '/tmp%s.pdf' % pagenum, bbox_inches = 'tight')
    plt.close(fig) # We don't want display
       
  
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def get_var_from_file(filename):
    f = open(filename)
    cluster_info = imp.load_source('cluster_info', filename)
    f.close()
    
    return cluster_info
