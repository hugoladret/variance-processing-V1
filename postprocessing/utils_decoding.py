#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import numpy as np 
from joblib import Parallel, delayed
from sklearn import preprocessing 
from lmfit import Model, Parameters

from tqdm import tqdm 

# --------------------------------------------------------------
# Fetching the data from the disk
# -------------------------------------------------------------- 
def load_neuron_data(cluster_path):
    spiketimes = np.load(prm.grouping_path + cluster_path + '/spiketimes.npy')
    spiketimes = spiketimes / prm.fs
    seq_contents = np.load(prm.grouping_path + cluster_path + '/sequences_contents.npy', allow_pickle = True)

    return {'cluster_path' : cluster_path,
            'spiketimes' : spiketimes,
            'seq_contents' : seq_contents}

# --------------------------------------------------------------
# Filtering the fetched data based on decoding type
# -------------------------------------------------------------- 
def filter_neuron_data(data,
                    data_type, target_btheta, target_theta,
                    min_t, max_t):
    
    spiketimes = data['spiketimes']
    seq_contents = data['seq_contents']
    
    if data_type == 'one_bt':
        spikes = [len(np.where((spiketimes > (seq['sequence_beg'] / prm.fs) + min_t) &
                    (spiketimes < (seq['sequence_beg'] / prm.fs) + max_t))[0])
                    for seq in seq_contents
                    if seq['sequence_btheta'] == target_btheta]

    elif data_type == 'bt_decoding_one_t' :
        spikes = [len(np.where((spiketimes > (seq['sequence_beg'] / prm.fs) + min_t) &
                    (spiketimes < (seq['sequence_beg'] / prm.fs) + max_t))[0])
                    for seq in seq_contents
                    if seq['sequence_theta'] == target_theta]

    elif data_type in ['all_bt',  'all_t_bt', 'bt_decoding']:
        spikes = [len(np.where((spiketimes > (seq['sequence_beg'] / prm.fs) + min_t) &
                    (spiketimes < (seq['sequence_beg'] / prm.fs) + max_t))[0])
                    for seq in seq_contents]

    return {'cluster_path' : data['cluster_path'],
            'spikes' : spikes}

# --------------------------------------------------------------
# Loading the data and filtering it using joblib's parallel loops
# Note that this return out_data in the same order as target_clusters
# So you don't need the id as previously
# -------------------------------------------------------------- 
def par_load_data(timesteps, target_clusters, 
                data_type, target_btheta, target_theta,
                disable_tqdm = False):
    
    # Load the data from the disk
    loaded_data = Parallel(n_jobs = -1)(delayed(load_neuron_data)(cluster_path) for cluster_path in tqdm(target_clusters, desc = 'Loading data', disable = disable_tqdm))
    
    # Filter the data to get correct timestep and spikes
    if data_type == 'one_bt' : # depends on the number of repetitions, thus on the decoding type
        last_shape = 30*12
    elif data_type == 'bt_decoding' :
        last_shape = 30*8*12
    elif data_type == 'all_t_bt' :
        last_shape = 30*8*12 # TODO is this correct ?
        
    out_data = np.zeros((len(timesteps), len(target_clusters), last_shape))
    for it, timestep in tqdm(enumerate(timesteps), total = len(timesteps), desc = 'Filtering data', disable = disable_tqdm):
        max_t = timestep + prm.win_size
        min_t = timestep 
        
        filtered_data = Parallel(n_jobs = -1)(delayed(filter_neuron_data)(data,
                                                                        data_type, target_btheta, target_theta,
                                                                        min_t, max_t) for data in loaded_data)

        if np.asarray(filtered_data[0]['spikes']).shape[0] != last_shape:
            print('Warning: the number of reptitions %s does not match last shape %s' %(np.asarray(filtered_data[0]['spikes']).shape, last_shape))
            
        # And reformat to array to follow the same ordering as target_clusters
        for ineuron, data in enumerate(filtered_data):
            '''print(data['spikes'])
            print(data['spikes'].shape)'''
            idx_match = np.where(data['cluster_path'] == target_clusters)[0]
            out_data[it, idx_match] = data['spikes']
            
    out_data = np.swapaxes(out_data, 1, -1) # swap the reptition and neuron axis, otherwise bug
            
    # Now we get the labels, which is the easy part
    seq_contents_example = np.load(prm.grouping_path + target_clusters[0] + '/sequences_contents.npy', allow_pickle = True)
    if data_type == 'one_bt':
        labels = ['T%.3f'% seq['sequence_theta'] for seq in seq_contents_example
                if seq['sequence_btheta'] == target_btheta]

    elif data_type == 'all_t_bt':
        labels = ['BT%.3fT%.3f'%(seq['sequence_btheta'],seq['sequence_theta']) 
                for seq in seq_contents_example]

    elif data_type == 'bt_decoding' :
        labels = ['BT%.3f'% seq['sequence_btheta'] for seq in seq_contents_example]
        
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    
    return out_data, le.transform(labels), le
        

# --------------------------------------------------------------
# Fitting the timecourse of population decoding
# -------------------------------------------------------------- 
def log_function(x, L, k0, k1, t0, t1, bsl0, bsl1):
    # x : array ; L : max ; k = hardness ; t0 = time midpoint ; bsl baseline
    pt1 = L / (1. + np.exp(-k0 * (x-t0))) + bsl0
    pt2 = -(L / (1. + np.exp(-k1 * (x-t1))) + bsl1)
    return pt1 + pt2

def fit_accum(array) :
    x = np.linspace(0, len(array), len(array), endpoint = False)
    y = array
    
    mod = Model(log_function)
    pars = Parameters()
    pars.add_many(('L', np.max(y), True, 0.0, np.max(y)*1.5),
                  ('k0', 0.5, True, 0.0, 1),
                  ('k1', 0.5, True, 0.25, 1),
                  ('t0', 17, True, 1, 50),
                  ('t1', 50, True, 40, 70),
                  ('bsl0', np.min(y), True, 0.0, np.max(y)/2),
                  ('bsl1', np.min(y), True, 0.0, np.max(y)/2)
                 )
    out = mod.fit(y, pars, x = x, nan_policy = 'omit', max_nfev = 5000)
    
    return out.best_values, np.abs(1-out.residual.var()/np.var(y))

def fit_all(all_params, all_means):
    titles = {'L' : 'Max. accuracy',
            'k0' : r'Steepness',
            'k1' : r'Steepness$_{decreasing}$',
            't0' : r'$\tau_1$ (ms)',
            't1' : r'$t_{decreasing}$'}
    params = titles.copy()
    for i, key in enumerate(titles.keys()) :
        param_list = []
        for i1, param in enumerate(all_params) :
            if key == 'L' :
                # it's easier to just get max accuracy this way instead of adding base+max
                param_list.append(np.max(all_means[i1]))
            else :
                param_list.append(param[key])
            
        params[key] = param_list
    return params, titles

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

