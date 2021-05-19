#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo

Contains the functions for :
    Kwik conversion from kwd to bin and from kwd to npy
    npy conversion to bin
    KwikTool loader

"""

import numpy as np
import h5py
import os
import sys
from tqdm import tqdm

def kwd_to_file(kwik_path, experiment_name, pipeline_name,
               channel_map, photodiode_index,
               output_type = 'bin', iterator = 0,
               do_ch_map = True, do_photodiode = True,
               verbose = True):
    '''
    Directly converts a kwik file (.raw.kwd extension) to a file for the pipeline,
    whether a bin in case no recording merging is needed or a npy in case further merging is needed
    
    Array types in case of a numpy output : int16 photodiode, float64 timestamps, int16 data
    
    Args :
        -kwik_path STR : the relative path to the experiment1_100.raw.kwd file
        -experiment_name STR : the name of the experiment, ex : "A005_a17", 
            used to identify subsequent files, NOT to load raw files
        -pipeline_name STR : the name of the current pipeline
        -channel_map ARR = channel map of the channels to extract from the kwd file
        -photodiode_index INT = index of the photodiode in the raw data, usually an analog in the bottom rows
        -output_type STR = 'bin', 'npy', the type of conversion to use
        -iterator INT = the # of the file, used in the name saving
        -verbose BOOL : whether the pipeline gets the right to talk
    '''
    print('# Converting a kwd file to a >%s< file #' % output_type)
    # -------------------------------------------------------------------------
    if verbose : print('Loading raw.kwd file into memory ...')
    
    dataset = kwk_load(kwik_path, 'all')
    
    data = dataset['data']['0']
    channels = dataset['channel_bit_volts']['0']
    timestamps = dataset['timestamps']['0']
    
    photodiode_data = data[:, photodiode_index]
    
    
    for chan in tqdm(channel_map) : 
        chan_data = data[: , chan]

        np.save('./pipelines/%s/%s.npy' % (pipeline_name, chan), chan_data)  
        
    del data
    os.remove(kwik_path)
    
    if verbose : print('Concatenating channels into a single array ...')
    data2 = np.load('./pipelines/%s/%s.npy' % (pipeline_name, channel_map[0]))
    for chan in tqdm(channel_map[1:]):
        chan_data = np.load('./pipelines/%s/%s.npy' % (pipeline_name, chan))
        data2 = np.vstack((data2, chan_data))
    data2 = data2.swapaxes(0,1)
        

    if verbose : print('Done ! Found %.3f seconds of recording in array of shape %s. '% (timestamps.max() -timestamps.min(), data2.shape))

    # Removes the temporary npy array
    mainpath = './pipelines/%s/' % pipeline_name   
    main_files = [file for file in os.listdir(mainpath) if os.path.isfile(os.path.join(mainpath, file))]
    for main_file in main_files :
        if '.npy' in main_file :
            os.remove(mainpath + main_file)
            
    
            
    # -------------------------------------------------------------------------
    # BIN CONVERSION
    # -------------------------------------------------------------------------
    if output_type == 'bin' :
        if verbose : print('Saving numpy array as an int16 binary file ...')
        data2.tofile('./pipelines/%s/mydata_%s.bin' % (pipeline_name,iterator))
        photodiode_data.tofile('./pipelines/%s/bins/phtdiode_%s.bin' % (pipeline_name,iterator))
        timestamps.tofile('./pipelines/%s/bins/timestamps_%s.bin' % (pipeline_name,iterator))
    
        del data2

        print('Conversion from raw.kwd to int16 binary file successfully completed !\n')
        
        with open('./pipelines/%s/debugfile.txt' %pipeline_name, 'a') as file:
            file.write('|-| File %s |-| \n' % iterator)
            file.write('| nbr_channels = %s | \n'%channels.shape[0])
            file.write('| timestamps_min = %s | \n' % timestamps.min())
            file.write('| timestamps_max = %s | \n' % timestamps.max())
                
    # -------------------------------------------------------------------------
    # NPY CONVERSION
    # -------------------------------------------------------------------------
    elif output_type == 'npy' :
        if verbose : print('Saving numpy array as a >.npy file< ...')
        np.save('./pipelines/%s/%s.npy' % (pipeline_name, experiment_name), data2)
        np.save('./pipelines/%s/%s_phtdiode.npy' % (pipeline_name, experiment_name), photodiode_data)
        np.save('./pipelines/%s/%s_timestamps.npy' % (pipeline_name, experiment_name), timestamps)

        del data2

        if verbose : print('Sanity check passed.')
        print('Conversion from raw.kwd to npy file successfully completed !\n')
    
        with open('./pipelines/%s/debugfile.txt' %pipeline_name, 'a') as file:
            file.write('|-| File %s |-| \n' % iterator)
            file.write('| nbr_channels = %s | \n'%channels.shape[0])
            file.write('| timestamps_min = %s | \n' % timestamps.min())
            file.write('| timestamps_max = %s | \n' % timestamps.max())

        
    else :
        print('Output format not implemented.')
        sys.exit()
    
        
# --------------------------------------------------------------
# @author: Josh Siegle
# https://github.com/klusta-team/kwiklib
# --------------------------------------------------------------
        
def kwk_load(filename, dataset=0):
    f = h5py.File(filename, 'r')
    
    if filename[-4:] == '.kwd':
        data = {}
        
        if dataset == 'all':
            data['info'] = {Rec: f['recordings'][Rec].attrs 
                            for Rec in f['recordings'].keys()}
            
            data['data'] = {Rec: f['recordings'][Rec]['data']
                            for Rec in f['recordings'].keys()}
            
            R = list(f['recordings'])[0]
            if 'channel_bit_volts' in f['recordings'][R]\
                                       ['application_data'].keys():
                data['channel_bit_volts'] = {Rec: f['recordings'][Rec]\
                                                   ['application_data']\
                                                   ['channel_bit_volts']
                                             for Rec in f['recordings'].keys()}
            else:
                # Old OE versions do not have channel_bit_volts info.
                # Assuming bit volt = 0.195 (Intan headstages). 
                # Keep in mind that analog inputs have a different value!
                # In out system it is 0.00015258789
                data['channel_bit_volts'] = {Rec: [0.195]*len(
                                                 data['data'][Rec][1, :]
                                                             )
                                             for Rec in f['recordings'].keys()}
                
            
            data['timestamps'] = {Rec: ((
                                        np.arange(0,data['data'][Rec].shape[0])
                                        + data['info'][Rec]['start_time'])
                                       / data['info'][Rec]['sample_rate'])
                                       for Rec in f['recordings']}
        
        else:
            data['info'] = f['recordings'][str(dataset)].attrs
            data['channel_bit_volts'] = f['recordings'][str(dataset)]\
                                         ['application_data']\
                                         ['channel_bit_volts']
            data['data'] = f['recordings'][str(dataset)]['data']
            data['timestamps'] = ((np.arange(0,data['data'].shape[0])
                                   + data['info']['start_time'])
                                  / data['info']['sample_rate'])
        return(data)
    
    elif filename[-4:] == '.kwe':
        data = {}    
        data['Messages'] = f['event_types']['Messages']['events']
        data['TTLs'] = f['event_types']['TTL']['events']
        return(data)
    
    elif filename[-4:] == 'kwik':
        data = {}    
        data['Messages'] = f['event_types']['Messages']['events']
        data['TTLs'] = f['event_types']['TTL']['events']
        return(data)
    
    elif filename[-4:] == '.kwx':
        data = f['channel_groups']
        return(data)
    
    else:
        print('Supported files: .kwd, .kwe, .kwik, .kwx')
        
# --------------------------------------------------------------
#
# --------------------------------------------------------------

def variable_from_debugfile(lookup_str, pipeline_name) :
    '''
    Returns a variable stored under the lookup_str string in the debugfile.txt file
    '''
    file = open('./pipelines/%s/debugfile.txt'%pipeline_name, 'r')
    var = ''
    for row in file :
        if lookup_str in row :
            var = row.split(' = ')[1].split(' |')[0]
    return var

# --------------------------------------------------------------
#
# --------------------------------------------------------------
    
def concatenate2D_from_disk(arrays_paths, pipeline_name,
                            verbose = True):
    '''
    Loads a list of 2D int16 numpy array from disk and concatenate them row by row into a bin file
    This is used to merge multiple kwd files into a single file for spike sorting.
    Concatenation is done in the order in which the file are specified in arrays_paths
    
    The files are temporary loaded to get the shapes and the array is memmapped.
    The function then iterates over the channels and loads the corresponding channel in each memory mapped array
    
    Args :
        -arrays_paths LST : list of arrays paths to be loaded, ex ['A005_a17.npy', 'A007_a17.npy']
        -pipeline_name STR : name of the pipeline 
        -verbose BOOL : verbosity of the function
    '''
    print('# Concatenating electrode 2D arrays #')
          
    channel_length_list = []
    memmap_list = []
    for array_path in arrays_paths :
        temp_arr = np.load('./pipelines/%s/%s' % (pipeline_name, array_path))
        chan_length = temp_arr.shape[0]
        chan_nbr = temp_arr.shape[1]
        
        channel_length_list.append(chan_length)
        
        del temp_arr
        
        memmap_arr = np.memmap(filename = './pipelines/%s/%s' % (pipeline_name, array_path),
                               dtype = 'int16',
                               shape = (chan_length, chan_nbr),
                               mode = 'r+')
        memmap_list.append(memmap_arr)
        
        if verbose : print('Memory mapped file : %s of shape %s' % (array_path, (chan_length, chan_nbr)))
        
    if verbose : print('Concatenating arrays row by row ...')    
    for chan in range(chan_nbr):
        concat_list = []
        for memmap in memmap_list:
            concat_list.append(memmap[:,chan])
            
        concat = np.concatenate((concat_list))
        
        with open('./pipelines/%s/data.bin' % pipeline_name, 'a+') as file :
            concat.astype('int16').tofile(file)
            
    if verbose : print('Done ! Running sanity check ...') #due to file size we can't match shape directly
    merged_size = os.path.getsize('./pipelines/%s/data.bin' % pipeline_name)
    arrays_size = np.sum([os.path.getsize('./pipelines/%s/%s' % (pipeline_name,x)) for x in arrays_paths])
    
    if arrays_size-merged_size < 300 : #the size of the header is 256 bits (bytes?)
        if verbose : print('Sanity check passed.')
        
        for array_path in arrays_paths :
            os.remove('./pipelines/%s/%s' % (pipeline_name, array_path))
        if verbose : print('Temporary files deleted.')
        
        print('Concatenation of 2D arrays completed !\n')
    
    else :
        print('Merged arrays size is not equal to the sum of arrays + header size.')
        sys.exit()

# --------------------------------------------------------------
#
# --------------------------------------------------------------
    
def concatenate1D_from_disk(arrays_paths, pipeline_name,
                            output_name,
                            verbose = True):
    '''
    Loads a list of 1D numpy array from disk and concatenate them 
    This is used to merge timestamps (float64) or photodiodechannel(int16)
    Concatenation is done in the order in which the file are specified in arrays_paths
    
    The files aren't memory mapped
    
    Args :
        -arrays_paths LST : list of arrays paths to be loaded, ex ['A005_a17.npy', 'A007_a17.npy']
        -pipeline_name STR : name of the pipeline 
        -output_name STR : name of the concatenated output, whether 'phtdiode' or 'timestamps'
        -verbose BOOL : verbosity of the function
    '''
    
    print('# Concatenating %s 1D arrays #' %output_name)
          
    array_list = []
    for array_path in arrays_paths :
        arr = np.load('./pipelines/%s/%s' % (pipeline_name, array_path))
        array_list.append(arr)
        if verbose : print('Loaded file : %s of shape %s' % (array_path, arr.shape[0]))
        
    if verbose : print('Concatenating arrays ...')                
    concat = np.concatenate((array_list))
    
    with open('./pipelines/%s/%s.bin' % (pipeline_name, output_name), 'a+') as file :
        concat.tofile(file)
            
    if verbose : print('Done ! Running sanity check ...')
    test = np.fromfile('./pipelines/%s/%s.bin' % (pipeline_name, output_name), dtype = type(concat[0]))
    
    if np.array_equal(concat, test) :
        if verbose : print('Sanity check passed.')
        
        for array_path in arrays_paths :
            os.remove('./pipelines/%s/%s' % (pipeline_name, array_path))
        if verbose : print('Temporary files deleted.')
        
        print('Concatenation of 1D arrays completed !\n')
    
    else :
        print('Mismatch in merged and single arrays.')
        sys.exit()
        
# --------------------------------------------------------------
#
# --------------------------------------------------------------
        
