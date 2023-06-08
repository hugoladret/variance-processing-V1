#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
Parameters file, which contains all the parameters used in the analysis.
NOT the same as params.py from the preprocessing folder.
"""
# Use font params for Adobe Illustrator
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# CHANGE THESE PARAMETERS  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# --------------------------------------------------------------
postprocess_name = 'paper_2023_postprocess' # Folder in which everything will be saved
grouping_path = '../preprocessing/cat/processed/paper_data_2023/clusters/' # Folder from which to fetch pre-processed data
main_neurons = ['Mary_A006_cl18' , 'Tom_I002_cl185' , 'Tom_V001_cl103'] # Neurons to be displayed in main figures
ex_neurons = ['Mary_A006_cl18', 'Tom_I002_cl185', 'Tom_V001_cl103',
            'Mary_C006c_cl56', 'Steven_I002_cl128', 'Steven_L001_cl103',
            'Tom_L001_cl131', 'Tom_V001_cl39', 'Tom_V003_cl116',
            'Tom_I001_cl172', 'Tom_H001_cl14', 'Tom_V003_cl112'] # Neurons to be displayed in single neurons analysis (use recaps.pdf to see which ones are good)
# --------------------------------------------------------------
# CHANGE THESE PARAMETERS  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# --------------------------------------------------------------

# if postprocessedname does not exist, create it
if not os.path.exists('./data/%s' % postprocess_name):
    os.makedirs('./data/%s' % postprocess_name)

# --------------------------------------------------------------
# Misc parameters
# --------------------------------------------------------------
name_untuned = 'Vulnerable neurons' # name of the untuned neurons
name_tuned = 'Resilient neurons' # name of the tuned neurons
col_untuned = '#F8766D' # color of the untuned neurons
col_tuned = '#00BFC4' # color of the tuned neurons

# --------------------------------------------------------------
# Stimulation parameters
# --------------------------------------------------------------
thetas = np.linspace(0, np.pi, 12, endpoint = False)  # stimuli orientation
B_thetas = np.linspace(np.pi/2, 0 , 8) / 2.5 # stimuli bandwidth
N_thetas = len(thetas)
N_B_thetas = len(B_thetas)
phases = [0.000, 0.025] # directions of motion

# --------------------------------------------------------------
# PSTH and dynamical parameters
# --------------------------------------------------------------
beg_psth = -0.10 #s
end_psth = 0.4 #s
binsize = 10 #ms
timesteps = np.arange(-.2, .4, .01) # time bins for decoding
t_start = 0.0 # ??
t_stop = 0.3 # ??
timebins = len(timesteps) # ??
fs = 30000. # sampling frequency Hz
latencies = np.linspace(0.05, .4, 20) # steps to compute dynamical neuron tuning curves
learning_rate = 0.01 # re-used for dynamical tuning curves fitting
num_epochs = 1024 # re-used for dynamical tuning curves fitting

# --------------------------------------------------------------
# Data fetching parameters
# --------------------------------------------------------------
exp_list = ['Mary', 'Tom', 'Steven'] # List of experiments (animals), used for splitting data when verifying merging
clustering_available = False
try :
    cluster_list = np.load('./data/%s/cluster_list.npy'% postprocess_name, allow_pickle = True)  # shuffled list of clusters
except:
    cluster_list = [name for name in os.listdir(grouping_path) if os.path.isdir(grouping_path+name)]
    np.random.seed(42)
    np.random.shuffle(cluster_list)
    np.save('./data/%s/cluster_list.npy' % postprocess_name, cluster_list)

try :
    tuned_lst = np.load('./data/%s/kmeans_tuned_lst.npy' % postprocess_name) # neurons that have Kmeans cluster 1 (or 0, can't remember :))
    untuned_lst = np.asarray([x for x in cluster_list if x not in tuned_lst]) # all the others
    clustering_available = True
except:
    print('No Kmeans list found, this is probably the first time you run the postprocessed pipeline\n\n')

# --------------------------------------------------------------
# Logistic regression  parameters
# --------------------------------------------------------------
max_iter = 5000 # maximum iteration of Logistic Regression
test_size = .15 # train/test split size
C = 1. # regularization parameter
multi_class = 'multinomial' # multi-class classification, usually don't change it
win_size = .1 #s, to fetch the data
opts_LR = dict(max_iter=max_iter, multi_class=multi_class, penalty='l2', n_jobs=-1, tol = 1e-5) # options for Logistic Regression
seed = 19967568 # seed for Logistic Regression
n_splits = 5 # for cross-validation within a dataset

cm_timesteps = [0., .1, .2, .3] # confusion matrices timesteps
cm_bt = [B_thetas[-1], B_thetas[4], B_thetas[0]] # confusion matrices bandwidths

pscan_test_sizes = np.linspace(.15, .5, 6) # used to test parameters
pscan_Cs = np.geomspace(0.001, 10, 10) # used to test parameters
pscan_win_sizes = np.linspace(.05, .2, 6) # used to test parameters

n_subgroups = 100 # number of neurons for each subgroup when doing the LR on res vs vul
n_bootstrap = 5 # number of bootstrap iterations for the LR on res vs vul
n_continuous = 20 # size of increments between pools of neurons for decoding on the score-based approach
total_loops = (len(cluster_list) - n_subgroups)//n_continuous # number of loops for decoding on the score-based approach
# i.e. 0:100 --> 20:120 --> 40:140 --> 60:160 --> 80:180 --> 100:200 for a n_loop_continuous = 20

# --------------------------------------------------------------
# Plotting  parameters
# --------------------------------------------------------------
colors = plt.cm.inferno(np.linspace(.8, .2, len(B_thetas))) #tc colormap
col_bt = plt.cm.viridis([0.2]) # color for plotting btheta decoding (dark blue)
rbins, abins = np.linspace(0.1, B_thetas[0], N_B_thetas+1)[::-1], np.linspace(0, np.pi, N_thetas+1) # for weight matrices
EL, AZ = np.meshgrid(rbins, abins) # for weight matrices
block_plot = False # if False, the plots will be saved but not displayed
