#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import utils_single_neuron as sn_utils 
import numpy as np 

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --------------------------------------------------------------
# Plotting PSTH of neurons
# --------------------------------------------------------------
def make_psth() :
    print('Plotting PSTH for example neurons')
    for cluster_path in prm.ex_neurons:
        load_folder = '%s/%s' % (prm.grouping_path, cluster_path)
        nm_PSTH_list = np.load(load_folder + '/PSTH.npy', allow_pickle=True)
        nm_PSTH_list = np.flip(nm_PSTH_list, axis=0)
        nmmeans = np.load(load_folder + '/TC.npy').mean(axis=-1)

        sn_utils.make_reduced_raster(figsize=(7, 6), filename=cluster_path,
                            nm_PSTH_list=nm_PSTH_list, nmmeans=nmmeans,
                            bt_idxs=[0, 7, 0, 7])
