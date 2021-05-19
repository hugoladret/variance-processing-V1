#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
import pipeline_params as prm

def export_sequences_times() :
    '''
    Loops through the folders and export photodiode times
    '''
    print('# Extracting sequences times from photodiode #')
          
    for folder in prm.folder_list :
        if prm.verbose : print('Extracting from %s folder' % folder)
        signal = np.fromfile('./pipelines/%s/photodiode.bin' % folder, np.int16)
         
        beg = int(len(signal)/prm.beg_index)
        end = len(signal) - (prm.end_index*prm.fs)
        flash_height_level = np.percentile(signal, prm.flash_height_percentile)
        baseline_height_level = np.percentile(signal, prm.baseline_height_percentile)
        
        sequences_times = get_peak_times(signal = signal, beg_index = beg, end_index = end,
                   flash_height = flash_height_level, baseline_height = baseline_height_level,
                   folder = folder)
        
        if prm.debug_plot : plot_sequences_length(sequences_times,prm.fs, folder)
        
        np.save('./results/%s/sequences_times.npy' % folder, sequences_times)
        
    print('# Photodiode analysis done !#\n')
 

# --------------------------------------------------------------
# 
# --------------------------------------------------------------

def get_peak_times(signal, beg_index, end_index,
                   flash_height, baseline_height,
                   folder):
    '''
    Plots the [0:beg_index] and [end_index:-1] signal of the photodiode
    with detected sequences. Beg and end index are purely cosmetic.
    And returns the peaks
    '''
    end_index = int(end_index)

    #get the chunks above flash_height
    chunk_list = custom_peak(signal, width = prm.width, height = flash_height)
    
    #i %2 == 0 is a stim, otherwise it's a grey screen
    sequences_times = []
    mean_duration = []
    for i, beg in enumerate(chunk_list) :
        try :
            stim_beg = chunk_list[i][0]
            stim_end = chunk_list[i+1][0]
    
            sequences_times.append((stim_beg, stim_end))
            mean_duration.append(stim_end-stim_beg)
        except IndexError :
            stim_beg = chunk_list[i][0]
            stim_end = chunk_list[i][0] + np.mean(mean_duration)
            sequences_times.append((stim_beg, stim_end))
            
    stim_duration = np.mean(mean_duration)
    print('Found mean stimulation duration of %.3f s' % (stim_duration/prm.fs))
    
    sequences_times2 = sequences_times
       
    if prm.debug_plot : 
        fig, ax = plt.subplots(1, 2, sharex= 'col', sharey = 'row',
                               figsize = (12,6))
        fig.tight_layout()
        
        ax[0].set_title('Beginning of the photodiode signal')
        ax[0].set_ylabel('Absolute value of signal')
        ax[0].set_xlabel(' Time (points)')
        ax[0].plot(signal[0:beg_index], c = 'grey')
        for i,sequence_time in enumerate(sequences_times2) :
            if sequence_time[1] <= beg_index :
                if i%2 == 0 : y = flash_height
                else : y = flash_height-25
                ax[0].plot((sequence_time[0], sequence_time[1]),
                              (y, y), c = 'r')
                
        ax[1].set_title('End of the photodiode signal')
        ax[1].plot(signal[end_index:-1], c = 'grey')
        for i,sequence_time in enumerate(sequences_times2) :
            if sequence_time[1] >= end_index :
                if i%2 == 0 : y = flash_height
                else : y = flash_height-25
                ax[1].plot((sequence_time[0]-end_index, sequence_time[1]-end_index),
                              (y, y), c = 'r')
        plt.suptitle(folder)
        fig.savefig('./results/%s/photodiode.pdf' % folder, format = 'pdf', bbox_inches = 'tight')
        
        plt.show()
    
    #Debug display in case a sequence is very large or very small
    for i, seq_time in enumerate(sequences_times2) :
        if (seq_time[1] - seq_time[0]) >= stim_duration * 1.6 or (seq_time[1] - seq_time[0]) <= stim_duration  /2 :
            print('Found an anormally short or long stimulation at stimulations %s' % i )
            print('Location : from %.3f to % .3f ' % (sequences_times2[i][0],sequences_times2[i][1]))
            
            if prm.debug_plot :
                fig, ax = plt.subplots(figsize = (12,6))
                fig.tight_layout()
                ax.set_title('Location of anomalous stimulation duration -- If you cant see anything, its because the anomaly is a microsequence')
                ax.set_ylabel('Absolute value of signal')
                ax.set_xlabel(' Time (points)')
                
                ax.plot(signal[int(seq_time[0]-20000): int(seq_time[1]+20000)], c = 'grey')
                ax.plot((seq_time[0]-seq_time[0]+20000, seq_time[1]-seq_time[0]+20000),
                                  (flash_height, flash_height), c = 'r')
                
                plt.show()
            
    # Manual fix for photodiodes, using the folder name
    sequences_times3, _ = fix_photodiode(sequences_times2 = sequences_times2, stim_duration = stim_duration, folder = folder)

    return sequences_times3
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def grouper(iterable, width):
    '''
    Black-magic powered iterator from :https://stackoverflow.com/questions/15800895/finding-clusters-of-numbers-in-a-list
    iterates through the list and generates n:[cluster] elements
    '''
    prev = None
    group = []
    for item in iterable:
        if not prev or item - prev <= width:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group

# --------------------------------------------------------------
# 
# --------------------------------------------------------------
        
def custom_peak(signal, width, height) :
    '''
    Wrapper for grouper, return a list of [(beg,end)] plateaux
    '''
    iterable = np.where(signal > height)[0]
    plateaux = list(enumerate(grouper(iterable, width), 1))
    
    real_peaks = []
    for plateau in plateaux :
        min_plat = plateau[1][0]
        max_plat = plateau[1][-1]
        real_peaks.append((min_plat, max_plat))
        
    return real_peaks

# --------------------------------------------------------------
# 
# --------------------------------------------------------------
       
def plot_sequences_length(sequences_times, fs, folder) :
    fig, ax = plt.subplots(figsize = (10,5))
    sequence_lengths = []
    for sequence in sequences_times :
        sequence_lengths.append((sequence[1]-sequence[0])/fs)
    ax.plot(sequence_lengths)
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Trial #')
    ax.set_title('Duration of trials, std = %.3f, mean = %.3f' % (np.std(sequence_lengths), np.mean(sequence_lengths)))
    
    fig.savefig('./results/%s/photodiode_stability.pdf' % folder,
                format = 'pdf', bbox_inches = 'tight')
    plt.show()
        
    
# --------------------------------------------------------------
# 
# --------------------------------------------------------------
def fix_photodiode(sequences_times2, stim_duration, folder) :
    fixed = False 
    
    if folder == 'Mary_B005' :
        '''
        Missplaced photodiode, splitting the sequence in two does the trick
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 57
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        fixed = True
        
    elif folder == 'Mary_B006' :
        '''
        Who the fuck walks in front of a screen during an experiment ?
        All the artifacts are from a shadow in front of the photodiode
        The indexes weren't the right ones from the debug output, and had to check them
        directly from matplotlib.. bug worth investigating
        '''
        print('A manual fix has been developped for this recording, fixing...')
        mi_idxs = [146, 434, 981, 1164, 1436, 2162, 2428, 2643, 3116, 4411, 4786,
                        5413, 5580]
        for missing_idx in mi_idxs :
            # Fixing previous element
            previous_end = sequences_times2[missing_idx][1].copy()
            new_start = sequences_times2[missing_idx][0]
            new_end = sequences_times2[missing_idx][0] + stim_duration
            sequences_times2[missing_idx] = (new_start, new_end)
            # Adding a new element
            content = (previous_end-stim_duration, previous_end)
            sequences_times2.insert(missing_idx, content)
        fixed = True
        
    elif folder == 'Mary_E004' :
        '''
        Same as Mary_B005
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 1102
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        fixed = True
        
    elif folder == 'Mary_E005' :
        '''
        Very odd, looks like luminance temporarily decreased on the photodiode for 6 stims
        So we need to add 6 stims after the interval
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 330
        new_start = sequences_times2[missing_idx][0] 
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        for i in range(1, 6) : 
            # Fixing previous element
            new_start = sequences_times2[missing_idx][0] + i*stim_duration
            new_end = sequences_times2[missing_idx][0] + (i*stim_duration) + stim_duration
            # Adding a new element
            content = (new_start, new_end)
            sequences_times2.insert(missing_idx+i,content )
        fixed = True
        
    elif folder == 'Duchess_D005' :
        '''
        This one experiment misses a photodiode flash right in the middle of the sequences
        so we re-add the missed flash
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 9323
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        
        fixed = True
        
    elif folder == 'Duchess_H006' :
        '''
        The first error is an absence of white flash, so only need to replace a sequence
        The second I think is only a timelag
        '''
        print('A manual fix has been developped for this recording, fixing...')
                
        missing_idx = 1285
        #Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        
        fixed = True
        
    elif folder == 'Duchess_I004' :
        '''
        The first error is an absence of white flash, so only need to replace a sequence
        The second is an error due to SOMEONE not stopping the recording
        NOT WORKING
        '''
        print('A manual fix has been developped for this recording, fixing...')
                
        missing_idx = 450
        #Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        
        missing_idx = 28797
        #Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        
        fixed = True
        
    elif folder == 'Steven_F001' :
        '''
        Missplaced photodiode, splitting the sequence in two does the trick
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 1979
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        fixed = True
        
    elif folder == 'Steven_I002' :
        '''
        Missplaced photodiode, splitting the sequence in two does the trick
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 90
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        fixed = True
        
    elif folder == 'Steven_Z001' :
        '''
        Missplaced photodiode, splitting the sequence in two does the trick
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 2619
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        fixed = True
        
    elif folder == 'Steven_AB01' :
        '''
        Missplaced photodiode, splitting the sequence in two does the trick
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 5
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        fixed = True
        
    elif folder == 'Steven_AK001' :
        '''
        Missplaced photodiode, splitting the sequence in two does the trick
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idx = 516
        # Fixing previous element
        new_start = sequences_times2[missing_idx][0]
        new_end = sequences_times2[missing_idx][0] + stim_duration
        sequences_times2[missing_idx] = (new_start, new_end)
        # Adding a new element
        content = (sequences_times2[missing_idx][0]+1, sequences_times2[missing_idx][0]+stim_duration)
        sequences_times2.insert(missing_idx,content )
        fixed = True
        
    elif folder == 'Steven_M001' :
        '''
        Seems like the photodiode signal is too high and we detect two non signals
        '''
        print('A manual fix has been developped for this recording, fixing...')
        missing_idxs = [5, 62]
        
        for idx in missing_idxs :
            del sequences_times2[idx]
        fixed = True
        
    return sequences_times2, fixed