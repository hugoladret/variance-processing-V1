#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo

Waveform analysis based on https://www.cell.com/neuron/fulltext/S0896-6273(09)00720-X?
Two points only from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhz149/5549031 (Bruno)
Todo : use sklearn silhouette to give coefficient https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
"""

import sys
import os 
import fileinput
from tqdm import tqdm
import imp

from sklearn import cluster
import numpy as np
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import pipeline_params as prm



def waveform_analysis() :
    '''
    Get waveforms for all clusters in folder_list's subfolders,
    characterizes them and classifies them with a KMean
    the output identity is saved in the neuron subfolder
    '''
    
    print('# Running waveform analysis #')
          
    # get the classif points using spiketimes and raw signals
    caracterise_waveforms()
    
    # perform the k-mean clustering
    if prm.verbose : print('Running K-means')
    kmean_waveforms()
    
    print('# Waveform analysis complete ! #\n')

# --------------------------------------------------------------
#
# --------------------------------------------------------------
    
def caracterise_waveforms():
    '''
    Using spiketimes under /results/ and raw data under /pipelines/
    caracterises the waveforms and saves everything as .npy files
    '''
    
    if prm.verbose : print('Getting classification points from waveforms')
    
    # iterate through cluster groups (= former pipeline folders)
    for folder in prm.folder_list :
        
        folder_path = './results/%s/' % folder
        raw_file_path = './pipelines/%s/converted_data.bin' % folder
        
        if prm.verbose : print('Loading raw data from %s' % raw_file_path )
        
        # Raw data
        raw_array = np.fromfile(raw_file_path, dtype = np.int16)
        raw_array = np.reshape(raw_array, (-1, prm.n_chan))
        
        # iterate through clusters
        clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
        for cluster_folder in tqdm(clusters_folders) :
            
            subfolder_path = folder_path + cluster_folder + '/'
            sys.path.insert(0, subfolder_path)
            
            cluster_info = get_var_from_file(folder_path + cluster_folder + '/cluster_info.py')
            
            spiketimes = np.load(subfolder_path+'spiketimes.npy')
            y = butter_bandpass_filter(raw_array[:, cluster_info.channel_id], 
                                       prm.lowcut, prm.highcut, prm.fs, prm.order)
            
            waveform_list = []
            for spiketime in spiketimes[10:-1] :
                beg = int(spiketime-(prm.window_size/2))
                end = int(spiketime+(prm.window_size*2))
                waveform_list.append(y[beg:end])
                
            mean_waveform = np.mean(waveform_list, axis = 0)
            interp_xs = np.arange(0, len(mean_waveform))
            interp_waveform = interp1d(interp_xs, mean_waveform)
            oversampled_xs = np.linspace(0, len(mean_waveform)-1, prm.interp_points, endpoint = False)
            splined_waveform_mean = interp_waveform(oversampled_xs)
            
            std_waveform = np.std(waveform_list, axis = 0)
            interp_xs = np.arange(0, len(std_waveform))
            interp_waveform = interp1d(interp_xs, std_waveform)
            oversampled_xs = np.linspace(0, len(std_waveform)-1, prm.interp_points, endpoint = False)
            splined_waveform_std = interp_waveform(oversampled_xs)
            
            try :
                classif_points = get_classif_points(splined_waveform_mean)
                np.save(subfolder_path + 'waveform_mean.npy', splined_waveform_mean)
                np.save(subfolder_path + 'waveform_std.npy', splined_waveform_std)
                np.save(subfolder_path + 'waveform_classif_points.npy', classif_points)
                
            except ValueError :
                print('Error in getting classif points, no spike-like waveform found for %s' % cluster_folder) 
                np.save(subfolder_path + 'waveform_mean.npy', splined_waveform_mean)
                np.save(subfolder_path + 'waveform_std.npy', splined_waveform_std)
                
                
            

# --------------------------------------------------------------
#
# --------------------------------------------------------------
            
def kmean_waveforms():
    '''
    Perform k-mean clustering from the available waveform caracterisation points in /results
    DO NOT FLIP THE ORDER OF FIRST AND SECOND CLASS TUPLES 
    '''
    unscaled_to_ms = prm.window_size / prm.fs
    unscaled_to_ms *= 1000
    
    all_carac_points = []
    path_to_carac_points = [] #use to write the kmeans info
    for folder in prm.folder_list :
        
        folder_path = './results/%s/' % folder
        clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
        
        for cluster_folder in clusters_folders :
            
            try :
                subfolder_path = folder_path + cluster_folder + '/'
                carac_points = np.load(subfolder_path + 'waveform_classif_points.npy', allow_pickle = True)
                all_carac_points.append(( (carac_points[0]['halfwidth'][0] * unscaled_to_ms) / prm.interp_points ,
                                         (carac_points[0]['troughtopeak'][0] * unscaled_to_ms) / prm.interp_points ))
                path_to_carac_points.append(subfolder_path)
            except FileNotFoundError : #in case the waveform couldn't be classified previously, not included in analysis
                pass
         
    kmeans = cluster.KMeans(n_clusters = prm.n_clusters, init = prm.k_init,
                            n_init = 10, max_iter=1000).fit(all_carac_points)
    
    first_class_tuples = []
    second_class_tuples = []
    
    #max_id = [i for i, tupl in enumerate(all_carac_points) if tupl[0]==np.max(all_carac_points, axis = 0)[0] and tupl[1]==np.max(all_carac_points, axis = 0)[1]]
    max_id = [i for i, tupl in enumerate(all_carac_points) if tupl[0]==np.max(all_carac_points, axis = 0)[0]][0]

    label_max_id = kmeans.labels_[max_id] # the label of the neuron maximizing trough to peak and half width, meaning this label is excitatory

    for i in range(len(kmeans.labels_)) :
        if kmeans.labels_[i] == label_max_id :
            first_class_tuples.append((all_carac_points[i][1], all_carac_points[i][0]))
            replace_if_exist(path_to_carac_points[i] + '/cluster_info.py',
                             'putative_type', 'putative_type = "exc"\n')
        else :
            second_class_tuples.append((all_carac_points[i][1], all_carac_points[i][0]))
            replace_if_exist(path_to_carac_points[i] + '/cluster_info.py',
                             'putative_type', 'putative_type = "inh"\n')

    xs1, ys1 = [], []
    for i in first_class_tuples :
            xs1.append(i[0])
            ys1.append(i[1])
            
    xs2, ys2 = [], []
    for i in second_class_tuples:
            xs2.append(i[0])
            ys2.append(i[1])
    
    if prm.debug_plot :
        plot_KMeans(xs1, ys1,
                    xs2, ys2)

# --------------------------------------------------------------
#
# --------------------------------------------------------------
        
def plot_KMeans(xs1, ys1,
                xs2, ys2) :
    '''
    Plots Kmeans classification of all the clusters
    '''
    fig, ax = plt.subplots(figsize = (8,8))

    ax.scatter(xs1, ys1,
                   c = 'r', label = 'Putative regular spiking (exc)')
    ax.scatter(xs2, ys2,
               c = 'b', label = 'Putative fast spiking (inh)')
    

    
    ax.set_xlabel('Through to peak (ms)')
    ax.set_ylabel('Half width (ms)')

    plt.legend()
    plt.grid()
    fig.savefig('./results/Kmeans.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()
    
# --------------------------------------------------------------
#
# --------------------------------------------------------------
            
def plot_waveforms():
    '''
    Plots the caracterisation points and mean waveform of all clusters
    '''
    
    all_mean_waveforms, all_classif_points, all_std_waveforms = [],[],[]
    title_list = []
    
    #it's that good old double iteration with list generator to reload .npy data
    for folder in prm.folder_list :
        
        folder_path = './results/%s/' % folder
        clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
        
        for cluster_folder in clusters_folders :
            
            subfolder_path = folder_path + cluster_folder + '/'
            
            mean_waveform = np.load(subfolder_path + 'waveform_mean.npy')
            all_mean_waveforms.append(mean_waveform)
            
            std_waveform = np.load(subfolder_path + 'waveform_std.npy')
            all_std_waveforms.append(std_waveform)
                        
            classif_points = np.load(subfolder_path + 'waveform_classif_points.npy', allow_pickle = True)
            all_classif_points.append(classif_points)
            
            title_list.append('/%s/%s' % (folder, cluster_folder))
    
    fig, ax = plt.subplots(len(all_mean_waveforms), 2, sharex= 'col', sharey = 'row',
                           figsize = (12,8))
    fig.tight_layout()
    
    # plotting, indices have been carefully triple checked
    for i in range(len(all_mean_waveforms)) :
        for j in range(2):
            if j == 0 : #plot the classif points
                ax[i,j].axhline(0, c = 'gray', linewidth = 2, alpha = .8)

                ax[i,j].plot((all_classif_points[i][0]['half_width_0_time'], all_classif_points[i][0]['half_width_1_time']),
                              (all_classif_points[i][0]['half_width_0_amp'], all_classif_points[i][0]['half_width_0_amp']),
                               c = 'k', linestyle = '--')
                # The first negative peak
                ax[i,j].plot((all_classif_points[i][0]['first_time'], all_classif_points[i][0]['first_time']),
                              (0, all_classif_points[i][0]['first_amp']),
                              c = 'k', linestyle = '--')
                 # The positive peak
                ax[i,j].plot((all_classif_points[i][0]['max_time'], all_classif_points[i][0]['max_time']),
                              (0, all_classif_points[i][0]['max_amp']),
                              c = 'k', linestyle = '--')
                # The trough to peak
                ax[i,j].plot((all_classif_points[i][0]['first_time'], all_classif_points[i][0]['max_time']),
                              (all_classif_points[i][0]['first_amp'], all_classif_points[i][0]['first_amp']),
                              c = 'k', linestyle = '--')
                # The waveform
                ax[i,j].plot(all_mean_waveforms[i])
                if i == 0 : ax[i,j].set_title(title_list[i] + ' - classification points')
                else : ax[i,j].set_title(title_list[i])
            elif j == 1 : #plot the mean waveforms
                mean_waveform = all_mean_waveforms[i]
                std_waveform = all_std_waveforms[i]
                ax[i,j].plot(mean_waveform, c = 'w')
                ax[i,j].fill_between(np.arange(0, len(mean_waveform)),
                                     mean_waveform - std_waveform,
                                     mean_waveform + std_waveform)
                if i == 0 : ax[i,j].set_title(title_list[i] + ' - mean and std waveform')
                else : ax[i,j].set_title(title_list[i])
                
    fig.savefig('./results/waveforms.pdf', format = 'pdf', bbox_inches = 'tight') 
    plt.show()
    
# --------------------------------------------------------------
#
# --------------------------------------------------------------
  
def get_classif_points(mean_waveform) :
    '''
    Returns the half width, trough to peak, peak amplitude asymetry for PCA classif
    as done in https://www.cell.com/neuron/fulltext/S0896-6273(09)00720-X?
    '''   

    # Positive peak
    max_amp = np.max(mean_waveform)
    max_amp_time = np.where(mean_waveform == max_amp)[0]
    middle = max_amp_time[0]
    
    # First negative peak
    first_min_amp = np.min(mean_waveform[:middle])
    first_min_time = np.where(mean_waveform == first_min_amp)[0]
    
    # Time between first negative peak and positive peak
    trough_to_peak = max_amp_time - first_min_time
    
    # Half width of the negative peak
    first_half_peak = find_nearest(mean_waveform[:first_min_time[0]], first_min_amp/2)
    second_half_peak = find_nearest(mean_waveform[first_min_time[0]:max_amp_time[0]], first_min_amp/2)
    first_half_peak_time = np.where(mean_waveform == first_half_peak)[0]
    second_half_peak_time = np.where(mean_waveform == second_half_peak)[0]
    half_width = second_half_peak_time - first_half_peak_time

    return[{'halfwidth' : half_width, 'troughtopeak' : trough_to_peak,
           'first_amp' : first_min_amp, 'first_time' : first_min_time,
           'max_amp' : max_amp, 'max_time' : max_amp_time,
           'half_width_0_amp' : first_half_peak, 'half_width_0_time' : first_half_peak_time,
           'half_width_1_amp' : second_half_peak, 'half_width_1_time' : second_half_peak_time}]
             
# --------------------------------------------------------------
#
# --------------------------------------------------------------
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --------------------------------------------------------------
#
# --------------------------------------------------------------
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
 
# --------------------------------------------------------------
#
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

# --------------------------------------------------------------
# 
# --------------------------------------------------------------
    
def get_var_from_file(filename):
    f = open(filename)
    cluster_info = imp.load_source('cluster_info', filename)
    f.close()
    
    return cluster_info