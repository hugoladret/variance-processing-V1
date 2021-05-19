#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# General parameters 
plt.rcParams['agg.path.chunksize'] = 10000
n_cpu = multiprocessing.cpu_count() #use max number of available CPUs

# Pipeline verbosity and plottingsity
verbose = True
debug_plot = False # plot when doing waveform and photodiode

# Pipeline steps 
do_wav = False
do_photodiode = False
do_mc = False
do_fit = False
do_idcard = True


# --------------------------------------------------------------
# Pre-sorting parameters
# --------------------------------------------------------------
#pipeline_name = 'Tom_all' # name of the folder created under /results/, which can contain multiple pipelines

# name of the experimental folders to load, careful if they are diff stimulations as only one stimtype is used
folder_list = ['Mary_A006', 
               #'Mary_B004', 'Mary_D005', 'Mary_E002',
                #'Mary_C006a', 'Mary_C006b', 'Mary_C006c', 'Mary_C006d', 
                
                # 'Tom_C001','Tom_D001','Tom_E003','Tom_F001','Tom_F002',
                # 'Tom_G001','Tom_G002','Tom_H001','Tom_H002','Tom_H004',
                # 'Tom_I001','Tom_I002','Tom_J001','Tom_J002','Tom_K001',
                # 'Tom_L001','Tom_M001','Tom_M002','Tom_N001','Tom_O001',
                # 'Tom_P001','Tom_R001','Tom_S001','Tom_T001',
                # 'Tom_V001','Tom_V002','Tom_V003','Tom_W001','Tom_X001',
                # 'Tom_Y001','Tom_Z001','Tom_AA01','Tom_BA01','Tom_CA01',
                
                # 'Steven_A001', 'Steven_B002', 'Steven_C001','Steven_D001','Steven_E001',
                # 'Steven_F001','Steven_F002','Steven_G001','Steven_G002','Steven_H001',
                # 'Steven_I001','Steven_I002','Steven_J001','Steven_K001','Steven_L001',
                # 'Steven_L002','Steven_M001','Steven_N001','Steven_N002',
                # 'Steven_O001','Steven_P001','Steven_Q001','Steven_R001','Steven_R001',
                # 'Steven_R002','Steven_S001',
                # 'Steven_T001','Steven_U001','Steven_V001',
                # 'Steven_Z001','Steven_AA001','Steven_AA002','Steven_AA003',
                # 'Steven_AB01','Steven_AC01','Steven_AD01','Steven_AF01',
                #   'Steven_AG01','Steven_AH01','Steven_AI01','Steven_AK001',
                #   'Steven_AL001','Steven_AM001' 
]

# --------------------------------------------------------------
# Waveform analysis parameters
# --------------------------------------------------------------
n_chan = 32  # used to reshape raw array when extracting spike waveforms

lowcut = 300.0 # Hz
highcut = 3000.0 # Hz
order = 6
fs = 30000.0 # Hz

n_spikes = 20000 # number of spikes to extract to get the mean waveform NO LONGER USED, all spikes are taken now
window_size = 30 # points/2 around the spiketime
interp_points = 300 # number of points used in the spline interpolation of waveforms

n_clusters = 2 # number of clusters in K-means
k_init = 'k-means++' # K-means init method, K++ uses the PCs as barycenters

# --------------------------------------------------------------
# Photodiode extraction parameters 
# --------------------------------------------------------------
beg_index = 30 # int(len(signal)/beg_index), index at which to end the signal beg visualisation
end_index = 22 # end = len(signal) - (end_index*fs) seconds at which to start the signal end visualisation
flash_height_percentile = 99.5 # np.percentile(signal, flash_height_percentile) height of photodiode blinks
baseline_height_percentile = 50 # height of the average signal, to distinguish start and end
width = 500 #units, if there is too much sequences compared to theoretical results, increase this param

# --------------------------------------------------------------
# Baseline removal
# --------------------------------------------------------------
substract_baseline = True #Substract the baseline activity in PSTH and TC, sampled from average grey-intertrial times
baseline_beg = -.05 #s
baseline_end = 0.05 #s

# --------------------------------------------------------------
# PSTH
# --------------------------------------------------------------
beg_psth = -0.15 #s
end_psth = 0.45 #s
binsize = 10 #ms
psth_fit_type = 'DN' #one of 'heeger' or 'DN', see nb_test_psth_fits for more infos

# --------------------------------------------------------------
# Tuning curves fitting
# --------------------------------------------------------------
fit_type = 'Torch' # one of 'Torch' or 'lmfit', fit with either poisson or least square
learning_rate = 0.01 # only applies if fit_type == 'Torch'
num_epochs = 1024 * 1  # only applies if fit_type == 'Torch'

# --------------------------------------------------------------
# Tuning curves
# --------------------------------------------------------------
delta_ts = np.linspace(0, .3, 50)
beg_TC = .1 #s unused as of variance update
end_TC = .25 # unused as of variance update
stack_TC = True # if True, averages both direction into one TC. Otherwise, takes the preferred direction

# --------------------------------------------------------------
# Sequence generation
# --------------------------------------------------------------
seq_type = 'drifting_mc_8_15' 
export_verif = True # Export the stimulation array to double check that everything has been correctly regenerated

if seq_type == 'static_mc' : 
    # Static MotionClouds, where phases are really phases and not directions
    thetas = np.linspace(0, np.pi, 12, endpoint = False)    
    B_thetas = np.linspace(np.pi/2, 0 , 8) / 2.5
    phases = np.linspace(0, .05, 6, endpoint = False)

    sf = np.array([.9]) / 2 

    repetition = 10
    seed = 42
    
elif seq_type == 'drifting_mc_8/5_15' : 
    # Drifting MotionClouds, where Btheta range is divided by 5 and not 2.5
    thetas = np.linspace(0, np.pi, 12, endpoint = False)    
    B_thetas = np.linspace(np.pi/2, 0 , 8) / 5
    phases = np.linspace(0, .05, 2, endpoint = False)

    sf = np.array([.9]) / 2 

    repetition = 15
    seed = 42
    
elif seq_type == 'drifting_mc_12_15' :
    # Drifting MotionClouds with 12 bthetas
    thetas = np.linspace(0, np.pi, 12, endpoint = False)    
    B_thetas = np.linspace(np.pi/2, 0 , 12) / 2.5
    phases = np.linspace(0, .05, 2, endpoint = False)

    sf = np.array([.9]) / 2 

    repetition = 15
    seed = 42
    
elif seq_type == 'drifting_mc_8_20' :
    # Drifting MotionClouds with 8 bthetas, 20 repetitions
    thetas = np.linspace(0, np.pi, 12, endpoint = False)    
    B_thetas = np.linspace(np.pi/2, 0 , 8) / 2.5
    phases = np.linspace(0, .05, 2, endpoint = False)

    sf = np.array([.9]) / 2 

    repetition = 20
    seed = 42
    
elif seq_type == 'drifting_mc_8_15' :
    # Drifting MotionClouds with 8 bthetas, 15 repetitions
    thetas = np.linspace(0, np.pi, 12, endpoint = False)    
    B_thetas = np.linspace(np.pi/2, 0 , 8) / 2.5
    phases = np.linspace(0, .05, 2, endpoint = False)

    sf = np.array([.9]) / 2 

    repetition = 15
    seed = 42
    
elif seq_type == 'drifting_gratings' :
    # Drifting gratings, 20 repetitions
    thetas = np.linspace(0, np.pi, 12, endpoint = False)    
    B_thetas = np.linspace(np.pi/2, 0 , 1) / 2.5
    phases = np.linspace(0, .05, 2, endpoint = False)

    sf = .9 / 2 

    repetition = 20
    seed = 42
    
# These sequence types predate the 28 jan    
elif seq_type == 'mc_phase' :
    thetas = np.linspace(0, np.pi, 12, endpoint = False)    
    B_thetas = np.linspace(np.pi/2, 0 , 12) / 2.5
    phases = np.linspace(0, 2*np.pi, 8, endpoint = False)

    sf = np.array([.9])

    repetition = 25
    seed = 42

elif seq_type == 'mix_mc_grat' :
    N_thetas = 12 # np.linspace(min_theta, max_theta, N_thetas, endpoint = False)
    min_theta = 0
    max_theta = np.pi
    
    N_Bthetas = 10 # np.linspace(min_btheta, max_btheta, N_Bthetas) / rectification_btheta
    min_btheta = np.pi/2 
    max_btheta = np.pi/32
    rectification_btheta = 2.5
    
    min_sf = .3
    max_sf = .8 # spatial frequency for the gratings, in cpd
    theta_factor = np.pi/2 #correction to each gratings to align them with MC 
    
    stim_duration = .2 # duration of stim for debug purposes
    inter_duration = .15 # duration of intertrial for debug purposes
    
    repetition = 30 # nr of sequence repetition < ----------------------
    
    seed = 42 # random state during stim genration
    
elif seq_type == 'long_fix_mc' :
    N_thetas = 12 # np.linspace(min_theta, max_theta, N_thetas)
    min_theta = 0
    max_theta = np.pi
    
    N_Bthetas = 8 # np.linspace(min_btheta, max_btheta, N_Bthetas)
    min_btheta = np.pi/2 
    max_btheta = np.pi/32
    rectification_btheta = 2.5
    
    stim_duration = 2 # duration of stim for debug purposes
    repetition = 15 # nr of sequence repetition < ----------------------
    
    seed = 42 # random state during stim genration
    
elif seq_type == 'tc_fix_mc' :
    N_thetas = 12 # np.linspace(min_theta, max_theta, N_thetas)
    min_theta = 0
    max_theta = np.pi
    
    N_Bthetas = 8 # np.linspace(min_btheta, max_btheta, N_Bthetas)
    min_btheta = np.pi/2 
    max_btheta = np.pi/32
    rectification_btheta = 2.5
    
    stim_duration = 2 # duration of stim for debug purposes
    repetition = 5 # nr of sequence repetition
    
    seed = 42 # random state during stim genration


# --------------------------------------------------------------
# Tuning curves plotting
# --------------------------------------------------------------
colors = plt.cm.inferno(np.linspace(.8, .3, len(B_thetas))) #tc colormap
