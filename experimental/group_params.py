#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import pipeline_params as unit_params

# WARNING : WE NOW ASSUME THAT BOTH DIRECTION ARE AVERAGED

overwrite = False # whether to copy the files anew (in case new electrode-levels analysis have been run for example)

# General parameters
plt.rcParams['agg.path.chunksize'] = 10000 # makes matplotlib less whiny
n_cpu = multiprocessing.cpu_count() # use max number of available CPUs
fs = unit_params.fs# Hz
B_thetas = unit_params.B_thetas
thetas = unit_params.thetas
colors = unit_params.colors

# Verbosity and plottingsity
verbose = True
debug_plot = False # optional plots when available

# Group analysis steps
do_export = True
do_exclusion = True
do_analysis = True
do_plotting = True 

# --------------------------------------------------------------
# General parameters 
# --------------------------------------------------------------
group_name = 'countforpaper'

folders_list = ['Mary_A006', 'Mary_B004', 'Mary_D005', 'Mary_E002',
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
'''folders_list = ['Mary_A006', 'Mary_B004', 'Mary_D005', 'Mary_E002',
                 
                
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
                'Steven_R002','Steven_S001','Steven_T001','Steven_U001','Steven_V001'
                ,'Steven_Z001','Steven_AA001','Steven_AA002','Steven_AA003',
                'Steven_AB01','Steven_AC01','Steven_AD01','Steven_AF01',
                'Steven_AG01','Steven_AH01','Steven_AI01','Steven_AK001',
                'Steven_AL001','Steven_AM001']'''

exp_list = ['Mary', 'Tom', 'Steven'] #prefix to split experiences for jazayeri


# --------------------------------------------------------------
# Exclusion parameters
# --------------------------------------------------------------
# Waveform based
waveform_value = 80

# Recording / fr based
drop_value = 2 # spk/s, neuron needs to keep firing above this threshold during the whole recording
drop_duration = 30 # s, neuron needs to be above drop_value for this consecutive duration (sliding window)

# Firing rate based, might introduce biais towards inh neuron, see BrunoO Pulvinar paper
#min_max_FR = 2  # spk/s, At Bt = 0, needs to be found in at least one of the TC mean
#tc_power = 120 # %, minimum peak-to-baseline ratio of TC, needs to be found in at least one Bt

# Fit based
min_R2 = .8 # u.a., minimum R2 goodness of fit on merge TC
#max_c50 = 1.25 #u.a., maximum c50 to have in either phi or cirvar NKR. Easily removes nonfitted neurons


# --------------------------------------------------------------
# Waveform parameters
# --------------------------------------------------------------
window_size = unit_params.window_size # points/2 around the spiketime
interp_points = 300 # number of points used in the spline interpolation of waveforms

# --------------------------------------------------------------
# Orientation decoder parameters
# --------------------------------------------------------------
deco_norm = True # Normalization of the decoder 

# --------------------------------------------------------------
# Plot : recap parameters
# --------------------------------------------------------------
min_cv_nbins = 10 #  Minimum Circular Variance histogram, number of bins
min_phi_nbins = 10 # Minimum Phi histogram, number of bins
cv_phi_scatter_xrange = 10, 55 # scatter cv/phi, Xlim range, points past that are reassigned to max or min
nkr_bins = 10 # Naka Rushton parameters histogram, number of bins

# --------------------------------------------------------------
# Plot : decoding parameters
# --------------------------------------------------------------
dec_ori_idxs = [0,4,-2] #Bthetas idxs (flipped) so 0 = gratings-like
dec_ori_thetas = np.linspace(1, 11, 6, dtype = np.int16, endpoint = True) # thetas idxs, nonflipped