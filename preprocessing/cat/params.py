#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Data folders
# --------------------------------------------------------------
group_name = 'paper_data_2023' # name of the folder created under /processed/, which can contain multiple experiments

# name of the experimental folders to load, careful if they are diff stimulations as only one stimtype is used
folder_list = [
            'Mary_A006', 'Mary_B004', 'Mary_D005', 'Mary_E002',
            'Mary_C006a', 'Mary_C006b', 'Mary_C006c', 'Mary_C006d',

            'Tom_C001','Tom_D001','Tom_E003','Tom_F001','Tom_F002',
            'Tom_G001','Tom_G002','Tom_H001','Tom_H002','Tom_H004',
            'Tom_I001','Tom_I002','Tom_J001','Tom_J002','Tom_K001',
            'Tom_L001','Tom_M001','Tom_M002','Tom_N001','Tom_O001',
            'Tom_P001','Tom_R001','Tom_S001','Tom_T001',
            'Tom_V001','Tom_V002','Tom_V003','Tom_W001','Tom_X001',
            'Tom_Y001','Tom_Z001','Tom_AA01','Tom_BA01','Tom_CA01',

            'Steven_A001', 'Steven_B002', 'Steven_C001','Steven_D001','Steven_E001',
            'Steven_F001','Steven_F002','Steven_G001','Steven_G002','Steven_H001',
            'Steven_I001','Steven_I002','Steven_J001','Steven_K001','Steven_L001',
            'Steven_L002','Steven_M001','Steven_N001','Steven_N002',
            'Steven_O001','Steven_P001','Steven_Q001','Steven_R001','Steven_R001',
            'Steven_R002','Steven_S001',
            'Steven_T001','Steven_U001','Steven_V001',
            'Steven_Z001','Steven_AA001','Steven_AA002','Steven_AA003',
            'Steven_AB01','Steven_AC01','Steven_AD01','Steven_AF01',
            'Steven_AG01','Steven_AH01','Steven_AI01','Steven_AK001',
            'Steven_AL001','Steven_AM001'
]

# --------------------------------------------------------------
#  Parameters
# --------------------------------------------------------------
# Pipeline steps, all of them will overwrite the data by default, so rerun with caution
do_photodiode = True
do_mc = True
do_fit = True
do_idcard = False

# General parameters
plt.rcParams['agg.path.chunksize'] = 10000 # for photodiode plotting purposes
n_cpu = multiprocessing.cpu_count() # use max number of available CPUs


# Pipeline verbosity and plottingsity
verbose = True
debug_plot = False # plotting will sometimes hang the pipeline process until windows are closed


# Photodiode extraction parameters
fs = 30000. #Hz
beg_index = 30 # int(len(signal)/beg_index), index at which to end the signal beg visualisation
end_index = 22 # end = len(signal) - (end_index*fs) seconds at which to start the signal end visualisation
flash_height_percentile = 99.5 # np.percentile(signal, flash_height_percentile) height of photodiode blinks
baseline_height_percentile = 50 # height of the average signal, to distinguish start and end
width = 500 # units, if there is too much sequences compared to theoretical results, increase this param


# Baseline removal
substract_baseline = True #Substract the baseline activity in PSTH and TC, sampled from average grey-intertrial times
baseline_beg = -0.05 #s in PST
baseline_end = 0.01 #s in PST


# PSTH
beg_psth = -0.15 #s
end_psth = 0.45 #s
binsize = 10 #ms


# Tuning curves fitting
fit_type = 'Torch' # one of 'Torch' or 'lmfit', fit with either poisson or least square
learning_rate = 0.01 # only applies if fit_type == 'Torch'
num_epochs = 1024  # only applies if fit_type == 'Torch'


# Tuning curves
delta_ts = np.linspace(0, .3, 50) # sliding window beginnings for maximum variance
stack_TC = True # if True, averages both direction into one TC. Otherwise, takes the preferred direction


# Stimulus sequence generation
seq_type = 'drifting_mc_8_15' # drifting_mc_8_15 is the one used in the paper, where we have 8 Bthetas x 12 thetas, with 15 repetitions

if seq_type == 'static_mc' :
    # Static MotionClouds, where phases are really phases and not directions
    thetas = np.linspace(0, np.pi, 12, endpoint = False)
    B_thetas = np.linspace(np.pi/2, 0 , 8) / 2.5
    phases = np.linspace(0, .05, 6, endpoint = False)

    sf = np.array([.9]) / 2

    repetition = 10
    seed = 42

elif seq_type == 'drifting_mc_8_15' :
    # Drifting MotionClouds with 8 bthetas, 15 repetitions
    thetas = np.linspace(0, np.pi, 12, endpoint = False)
    B_thetas = np.linspace(np.pi/2, 0 , 8) / 2.5
    phases = np.linspace(0, .05, 2, endpoint = False)

    sf = np.array([.9]) / 2

    repetition = 15
    seed = 42


# Tuning curves plotting
colors = plt.cm.inferno(np.linspace(.8, .3, len(B_thetas))) #tc colormap, note that this might can be different than the post-processing


# Merge parameters 
do_export = True # whether to group all files into a new folder --> required for postprocessing 
do_exclude = True # whether to eliminate neurons based on exclusion parameters (see below)
do_rename = True # whether to rename the files to more convenient names


# Exclusion parameters
drop_value = 2 # spk/s, neuron needs to keep firing above this threshold during the whole recording
drop_duration = 30 # s, neuron needs to be above drop_value for this consecutive duration (sliding window)
min_R2 = .8 # u.a., minimum R2 goodness of fit on merge TC