#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""
import itertools
import os

import numpy as np

import scipy.ndimage as snd

from tqdm import tqdm

import MotionClouds as mc
import params as prm


# --------------------------------------------------------------
# 
# --------------------------------------------------------------

def mc_analysis() :
    '''
    Iterates through folders, linking spiketimes, stimulation info and timing together.
    Then, reloads this array and performs :
        Baseline removal, from pre-trig spikes
        Raw data extraction for tuning curve
        Circular Variance 
        Peristimulus time histogram 
        Averaging both directions in to one
        Statistics on the baseline OSI vs evoked OSI
        
    Fit are then done in pipeline_fit.py
    '''
    
    print('# Running MotionClouds analysis #')
    
    for folder in prm.folder_list :
        sync_sequences(folder)
        
        baseline_removal(folder)
        
        ori_selec(folder)
        cirvar(folder)
        psth(folder)
        
        if prm.stack_TC == False :
            choose_TC(folder)
            
        
        print('# MotionClouds analysis done ! #\n')


# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def sync_sequences(folder) :
    '''
    Starts by regenerating the infos in the stimulation sequences, then matching it with
    the stimulation times extracted by the photodiode.
    Then, for each cluster in the folder, saves a npy array that for each trial contains
    beg time, end time, theta, btheta, FR and type of stimulation 
    '''
    print('Synchronizing sequences...')

    # Regenerates the sequence used in stimulations
    np.random.seed(prm.seed)
    MC_sequence = list(itertools.product(prm.thetas, prm.B_thetas, prm.phases))
    np.random.shuffle(MC_sequence)
    
    # Regenerates the first sequence
    generated_MC_sequences = []
    for seq in tqdm(MC_sequence, 'Generating MC for RNG', position = 0, leave = True):
        theta = seq[0]
        btheta = seq[1]
        phase = seq[2]
        stimtype = 'MC'
        im_0 = generate_cloud(theta=theta, b_theta=btheta, contrast=1,
                        N_X=512, N_Y=512, seed=prm.seed, transition=False,
                        phase = phase)[:, :, 0]
        im_1 = generate_cloud(theta=theta, b_theta=btheta, contrast=1,
                            N_X=512, N_Y=512, seed=prm.seed, transition=True,
                            phase = phase)[:, :, 0]

        dico = {'theta': theta, 'btheta': btheta, 'phase' : phase, 'stimtype': stimtype,
                'im_0': im_0, 'im_1': im_1}
        generated_MC_sequences.append(dico)
    
    # Reshuffles the sequence as many time as it's been repeated during stimulations
    full_sequence, verif_list = [], []
    for repeat in range(prm.repetition) :
        np.random.shuffle(generated_MC_sequences)
        verif_list.append(generated_MC_sequences[0]['theta'])
        full_sequence += generated_MC_sequences
        
    print('\nVerification list :')
    print(verif_list)
    
    if prm.verbose : print('Stimulation sequence of %sx%s = %s trials'
                    % (prm.repetition, len(generated_MC_sequences), len(full_sequence)))
    
    # And now to merge the content and the times of the sequence
    # Getting sequence times
    sequences_times = np.load('./processed/%s/sequences_times.npy' % folder)
    # Synchronizing both
    if len(sequences_times) == len(full_sequence) :
        print('Length of extracted sequences matches length of sequences infos\n')
        
        sequences_list = [] 
        for i, sequence in enumerate(sequences_times):
            seq_dict = {'sequence_beg' : sequences_times[i][0],
                        'sequence_end' : sequences_times[i][1],
                        'sequence_theta' : full_sequence[i]['theta'],
                        'sequence_btheta' : full_sequence[i]['btheta'],
                        'sequence_phase' : full_sequence[i]['phase'],
                        'stimtype' : full_sequence[i]['stimtype']}
            sequences_list.append(seq_dict)
    
    # Something went wrong in the param entered or in the photodiode detection
    else :
        print('Non matching dimension between photodiode and sequence infos')
        print('Length of sequences from photodiode = %d ' % (len(sequences_times)))
        print('Length of sequences from sequence infos parameters = %d ' % (len(full_sequence)))
        raise ValueError('len error at %s' % folder)
        
    # Loading clusters to add firing rates
    folder_path = './processed/%s/' % folder
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    if prm.verbose : print('Maximizing variance with Delta T...')
    for cluster_folder in tqdm (clusters_folders) : 
        spiketimes = np.load(folder_path + cluster_folder + '/spiketimes.npy')
        variances = []
        for delta_t in prm.delta_ts :
            meanresponse = []
            for sequence in sequences_list :
                #if sequence['sequence_btheta'] == 0.0 :
                beg = sequence['sequence_beg'] + (delta_t * prm.fs)
                end = beg + (prm.delta_ts[-1] * prm.fs)
                spikes_idx = np.where((spiketimes > beg ) & (spiketimes < end))[0]
                spikes = spiketimes[spikes_idx]
                meanresponse.append(len(spikes))
            variances.append(np.var(meanresponse))
        np.save(folder_path + cluster_folder + '/deltaT_variances.npy', variances)
        np.save(folder_path + cluster_folder + '/optimal_delay.npy', prm.delta_ts[np.argmax(variances)])
            
            
    if prm.verbose : print('Synchronizing with spiketimes...')
    for cluster_folder in tqdm(clusters_folders) :
        spiketimes = np.load(folder_path + cluster_folder + '/spiketimes.npy')
        delta_t = np.load(folder_path + cluster_folder + '/optimal_delay.npy')
        
        sequences_list_with_FR = []
        for i, sequence in enumerate(sequences_list):
            beg = sequence['sequence_beg'] + (delta_t * prm.fs)
            end = beg + (prm.delta_ts[-1] * prm.fs)
            bsl_beg = sequence['sequence_beg'] + (prm.baseline_beg * prm.fs)
            bsl_end = sequence['sequence_beg'] + (prm.baseline_end * prm.fs)
            
            spiketimes_in_seq = np.where((spiketimes > beg) & (spiketimes < end))[0]
            spiketimes_in_bsl = np.where((spiketimes > bsl_beg) & (spiketimes < bsl_end))[0]
            new_seq_dict = {'sequence_beg' : sequence['sequence_beg'],
                            'sequence_end' : sequence['sequence_end'],
                            'sequence_theta' : sequence['sequence_theta'],
                            'sequence_btheta' : sequence['sequence_btheta'],
                            'sequence_phase' : sequence['sequence_phase'],
                            'stimtype' : sequence['stimtype'],
                            'spiketimes' : spiketimes[spiketimes_in_seq],
                            'tot_spikes' : len(spiketimes_in_seq),
                            'spiketimes_bsl' : spiketimes[spiketimes_in_bsl],
                            'tot_spikes_bsl' : len(spiketimes_in_bsl)}
            
            sequences_list_with_FR.append(new_seq_dict)

        spiketime_density = [sequence['tot_spikes'] for sequence in sequences_list_with_FR]

        np.save(folder_path + cluster_folder + '/plot_spiketime_density.npy', spiketime_density)
        np.save(folder_path + cluster_folder + '/sequences_contents.npy', sequences_list_with_FR)
    
    
    
# --------------------------------------------------------------
# 
# --------------------------------------------------------------        
def ori_selec(folder) :
    '''
    Runs orientation selectivity analysis on each cluster subfolder in the folder
    saves directly as arrays for merged and non merged tuning curves of the cluster
    
    The trick to do phase or not here is to iterate by phase, but only be selective to 
    phase if the prm.stack_TC boolean is False. Both files are then duplicates and one
    can load either of them to obtain the same result
    '''
    print('\nAnalyzing orientation selectivity...')
    folder_path = './processed/%s/' % folder
    
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    for cluster_folder in tqdm(clusters_folders) :
        sequences_contents = np.load(folder_path + cluster_folder + '/sequences_contents.npy',
                                    allow_pickle = True)
        bsl = np.load(folder_path + cluster_folder + '/baseline.npy')

        #-----------------
        # Bthetas Merged
        #-----------------
        # Iterates through phases
        for phase in prm.phases :
            TC_list = []
            
            #Iterates through all possible thetas
            for u_theta in prm.thetas :
                
                # And through sequences to find them
                spikes_per_theta = []
                for seq in sequences_contents :
                    if seq['sequence_theta'] == u_theta and seq['stimtype'] == 'MC' and seq['sequence_phase'] == phase and not prm.stack_TC : 
                        spikes_per_theta.append(seq['tot_spikes']- np.mean(bsl))
                    
                    elif seq['sequence_theta'] == u_theta and seq['stimtype'] == 'MC' and prm.stack_TC :     
                        spikes_per_theta.append(seq['tot_spikes']- np.mean(bsl))
                        
                TC_list.append(spikes_per_theta)
                
            means, stds = [], []
            for theta in TC_list : 
                means.append(np.mean(theta))
                stds.append(np.std(theta))
                
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_TC_merged_means.npy' % phase, means)
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_TC_merged_stds.npy' % phase, stds)
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_TC_merged_all.npy' % phase, TC_list)
                    
                
        #--------------------
        # Bthetas Not merged
        #--------------------
        for phase in prm.phases  :
            TC_list_btheta = []
            for ibt, u_btheta in enumerate(prm.B_thetas):
                
                # And all the thetas
                TC_list_theta =[] 
                for u_theta in prm.thetas :
                    
                    spikes_per_thetabtheta = []
                    for seq in sequences_contents :
                        if seq['sequence_theta'] == u_theta and seq['sequence_btheta'] == u_btheta and seq['sequence_phase'] == phase and seq['stimtype'] == 'MC' and not prm.stack_TC:
                            spikes_per_thetabtheta.append(seq['tot_spikes'] - bsl[ibt])
                        
                        elif seq['sequence_theta'] == u_theta and seq['sequence_btheta'] == u_btheta and seq['stimtype'] == 'MC' and prm.stack_TC:
                            spikes_per_thetabtheta.append(seq['tot_spikes'] - bsl[ibt])
                            
                    TC_list_theta.append(spikes_per_thetabtheta)
                TC_list_btheta.append(TC_list_theta)
    
            all_means, all_stds = [], []
            for i in range(len(TC_list_btheta)):
                means = np.mean(TC_list_btheta[i], axis = 1)
                stds = np.std(TC_list_btheta[i], axis = 1)
                
                all_means.append(means)
                all_stds.append(stds)
            
            #saves a (b_theta, thetas) shaped arrays
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_TC_nonmerged_means.npy'% phase, all_means)
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_TC_nonmerged_stds.npy'% phase, all_stds)
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_TC_nonmerged_all.npy'% phase, TC_list_btheta)
            
    
# --------------------------------------------------------------
# 
# --------------------------------------------------------------  
def cirvar(folder) :
    '''
    Computes Circular variance (see ringach paper) on the mean for each btheta
    '''

    print('\nCalculating Circular Variance...')
    
    folder_path = './processed/%s/' % folder
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    for cluster_folder in tqdm(clusters_folders) :
        for phase in prm.phases :
            fr_means = np.load(folder_path+cluster_folder+'/%.3f_plot_MC_TC_nonmerged_means.npy' % phase)
            baseline = np.load(folder_path + cluster_folder + '/baseline.npy')
            
            cvs = []
            for i, btheta in enumerate(prm.B_thetas) :
                arr = fr_means[i, :] - baseline[i] # Maybe baseline isn't needed because it's already removed from the nonmerged_means array
                R = np.sum( (arr * np.exp(2j*prm.thetas)) / np.sum(arr) )
                cirvar = 1 - np.abs(np.real(R))
                cvs.append(cirvar)
            
            np.save(folder_path + cluster_folder + '/%.3f_cirvar.npy' % phase, cvs)
            
            # fit the cv curve with a naka rushton
            # arr_fit, r2 = fit_nkr(cvs)
            # np.save(folder_path + cluster_folder + '/%.3f_cirvar_fit.npy' % phase, arr_fit)
            # np.save(folder_path + cluster_folder + '/%.3f_cirvar_fitr2.npy' % phase, r2)
            
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def psth(folder) :
    '''
    PSTH, see TC function for the direction trick and comments
    Baseline is removed in plotting, as this doesnt affect any calculation
    '''
    
    print('\nAnalyzing PSTH...')
    
    folder_path = './processed/%s/' % folder
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    beg_psth = prm.beg_psth * prm.fs
    end_psth = prm.end_psth * prm.fs
    
    n_bin = (end_psth/prm.fs) - (beg_psth/prm.fs)
    n_bin *= 1000
    n_bin /= prm.binsize
    
    
    for cluster_folder in tqdm(clusters_folders) :
        sequences_contents = np.load(folder_path + cluster_folder + '/sequences_contents.npy',
                                    allow_pickle = True)
        spiketimes = np.load(folder_path + cluster_folder + '/spiketimes.npy')

        #-----------------
        # Bthetas Merged
        #-----------------
        for phase in prm.phases :
            PSTH_list = []
            for u_theta in prm.thetas :
                spikes_per_theta = []
                for seq in sequences_contents  :
                    if seq['sequence_theta'] == u_theta and seq['stimtype'] == 'MC' and seq['sequence_phase'] == phase and not prm.stack_TC :
                        near_sequence_beg = np.where((spiketimes > seq['sequence_beg'] + beg_psth) & (spiketimes < seq['sequence_beg']  + end_psth))[0]
                        spikes_per_theta.append( (spiketimes[near_sequence_beg]/prm.fs) - (seq['sequence_beg']/prm.fs))
                    elif seq['sequence_theta'] == u_theta and seq['stimtype'] == 'MC' and prm.stack_TC :
                        near_sequence_beg = np.where((spiketimes > seq['sequence_beg'] + beg_psth) & (spiketimes < seq['sequence_beg']  + end_psth))[0]
                        spikes_per_theta.append( (spiketimes[near_sequence_beg]/prm.fs) - (seq['sequence_beg']/prm.fs))
                PSTH_list.append(spikes_per_theta)
                
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_PSTH_merged.npy' % phase, PSTH_list)
         
        #--------------------
        # Bthetas Not merged
        #--------------------
        for phase in prm.phases :
            PSTH_list_btheta = []
            for u_btheta in prm.B_thetas :
                PSTH_list_theta = []
                for u_theta in prm.thetas : 
                    spikes_per_thetabtheta = []
                    for seq in sequences_contents :
                        if seq['sequence_theta'] == u_theta and seq['sequence_btheta'] == u_btheta and seq['stimtype'] == 'MC' and seq['sequence_phase'] == phase and not prm.stack_TC  :
                            near_sequence_beg = np.where((spiketimes > seq['sequence_beg'] + beg_psth) & (spiketimes < seq['sequence_beg']  + end_psth))[0]
                            spikes_per_thetabtheta.append( (spiketimes[near_sequence_beg]/prm.fs) - (seq['sequence_beg']/prm.fs))
                        elif seq['sequence_theta'] == u_theta and seq['sequence_btheta'] == u_btheta and seq['stimtype'] == 'MC' and prm.stack_TC  :
                            near_sequence_beg = np.where((spiketimes > seq['sequence_beg'] + beg_psth) & (spiketimes < seq['sequence_beg']  + end_psth))[0]
                            spikes_per_thetabtheta.append( (spiketimes[near_sequence_beg]/prm.fs) - (seq['sequence_beg']/prm.fs))
                    PSTH_list_theta.append(spikes_per_thetabtheta)
                PSTH_list_btheta.append(PSTH_list_theta)
                
            np.save(folder_path + cluster_folder + '/%.3f_plot_MC_PSTH_nonmerged.npy' % phase, PSTH_list_btheta)



## --------------------------------------------------------------
## 
## --------------------------------------------------------------
    
def baseline_removal(folder) :
    '''
    Generates a baseline.npy file, which gives an average baseline value
    to substract from histograms and tuning curves.
    '''

    folder_path = './processed/%s/' % folder
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    print('\nSubstracting baseline...')
    
    for cluster_folder in tqdm(clusters_folders) :
        #if verbose : print('analyzing ./processed/%s/%s' % (folder, cluster_folder))
        
        sequences_contents = np.load(folder_path + cluster_folder + '/sequences_contents.npy', allow_pickle = True)
        
        baseline_per_bt = np.zeros(len(prm.B_thetas))
        count_bsl = 0
        for ibt, bt in enumerate(prm.B_thetas) : 
            for seq in sequences_contents :
                if seq['sequence_btheta'] == bt :
                    baseline_per_bt[ibt] += seq['tot_spikes_bsl']
                    count_bsl +=1
            baseline_per_bt[ibt] /= count_bsl
            
        np.save(folder_path + cluster_folder + '/baseline.npy', baseline_per_bt)
        
        
## --------------------------------------------------------------
## 
## --------------------------------------------------------------
    
def choose_TC(folder) :
    '''
    Choose the best tuning curve, as determined by the one which max spikes is most important
    Not implemented in the plotter, need to go manually searching for it if you only want to work on 180 deg
    '''

    folder_path = './processed/%s/' % folder
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    print('\nPerforming direction correction...')
    
    for cluster_folder in tqdm(clusters_folders) :
        nonmerged_TC_p1 = np.load(folder_path + cluster_folder + '/0.000_plot_MC_TC_nonmerged_means.npy')
        nonmerged_TC_p2 = np.load(folder_path + cluster_folder + '/0.025_plot_MC_TC_nonmerged_means.npy')

        nonmerged_phases = []
        for i0 in range(0, len(prm.B_thetas)) :
            p1_max = np.max(nonmerged_TC_p1[i0, :])
            p2_max = np.max(nonmerged_TC_p2[i0, :])
            best_nonmerged_phase = np.argmax([p1_max, p2_max])
            nonmerged_phases.append(best_nonmerged_phase)
        val, counts = np.unique(nonmerged_phases, return_counts = True)
        preferred_phase = ['0.000', '0.025'][np.argmax(counts)]
        
        np.save(folder_path + cluster_folder + '/preferred_phase.npy', preferred_phase)
        
        
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def generate_cloud(theta, b_theta, phase,
                   N_X, N_Y, seed, contrast=1.,
                   transition=False):

    fx, fy, ft = mc.get_grids(N_X, N_Y, 1)
    
    disk = mc.frequency_radius(fx, fy, ft) < .5
    
    if b_theta == 0 : 
        mc_i = mc.envelope_gabor(fx, fy, ft,
                                 V_X=0., V_Y=0., B_V=0.,
                                 sf_0=0.1232558, B_sf=0.1232558,
                                 theta=0, B_theta=b_theta)
    else : 
        mc_i = mc.envelope_gabor(fx, fy, ft,
                                 V_X=0., V_Y=0., B_V=0.,
                                 sf_0=0.1232558, B_sf=0.1232558,
                                 theta=theta, B_theta=b_theta)
        
    im_ = np.zeros((N_X, N_Y, 1))
    im_ += mc.rectif(mc.random_cloud(mc_i*np.exp(1j * phase), seed=seed),
                     contrast=2)
        
    im_ *= disk  # masking outside the disk
    im_ += -.5
    im_ += .5*(1-disk)  # gray outside the disk
    
    if b_theta == 0 : im_ = snd.rotate(im_, theta * 180 / np.pi, reshape = False)
    if transition:
        im_[10:40, 10:40] = 1
    else:
        im_[10:40, 10:40]= -.35

    return im_
