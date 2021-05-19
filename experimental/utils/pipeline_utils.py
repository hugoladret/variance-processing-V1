#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo

Contains the functions to manipulate folders for the pipeline such as :
    Exporting the data from the pipeline to the result folder
"""

import os
import numpy as np
import csv
from tqdm import tqdm
import shutil
import pipeline_params as prm
import group_params as gprm

# --------------------------------------------------------------
#
# -------------------------------------------------------------- 
def export_to_results():
    '''
    Exports the data to results folder/pipeline_name subfolder/clusters folders
    '''

    for folder in prm.folder_list :
        pipeline_name = folder
        if not os.path.exists('./results/%s/' % pipeline_name):
            
            print('# Exporting results to /results/%s #\n' % pipeline_name)
            path = './pipelines/'+pipeline_name
            resultpath = './results/'+pipeline_name
            
            # Spiketimes
            spiketimes = np.load(path+'/spike_times.npy')
            # Cluster ID per spiketimes
            spiketimes_clusters_id = np.load(path+'/spike_clusters.npy')
            
            # Spiketimes/clusters tuple table
            spike_cluster_table = []
            for i, spike in enumerate(spiketimes):
                spike_cluster_table.append((spike, spiketimes_clusters_id[i]))
                
            # Good clusters as labelled by phy
            good_clusters = []
            with open(path+'/cluster_info.tsv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    row_values = row[0].split('\t')
                    cluster_id, channel, group = row_values[0], row_values[5], row_values[8]
                    depth, n_spikes = row_values[6], row_values[9]
                    if group == 'good' :
                        good_clusters.append([int(cluster_id), int(channel), float(depth), int(n_spikes)])
                        
            # Spiketimes for each good cluster
            good_spikes = []
            for good_cluster in good_clusters :
                tmp_lst = []
                for spike_cluster in spike_cluster_table :
                    if spike_cluster[-1] == good_cluster[0] :
                        tmp_lst.append(spike_cluster[0])
                good_spikes.append(tmp_lst)
                
            # and remerging structure
            merged_clusters = []
            for i, g_cluster in enumerate(good_clusters) :
                cluster_id, cluster_channel = g_cluster[0], g_cluster[1]
                cluster_depth, cluster_nspikes = g_cluster[2], g_cluster[3]
                cluster_spiketimes = good_spikes[i]
                
                merged_cluster = [cluster_id, cluster_channel, cluster_depth, cluster_nspikes, cluster_spiketimes]
                merged_clusters.append(merged_cluster)
                
            # export everything
            os.makedirs(resultpath)
            for merged_cluster in tqdm(merged_clusters, 'Moving into subfolders') :
                cluster_path = resultpath+ '/%s_cl' %pipeline_name + str(merged_cluster[0])
                os.makedirs(cluster_path)
                
                # save spiketimes
                np.save(cluster_path + '/spiketimes.npy', merged_cluster[-1])
                
                #save infos
                with open(cluster_path + '/cluster_info.py', 'w+') as file :
                    file.write('channel_id = %d\n' % merged_cluster[1])
                    file.write('channel_depth = %.2f\n' % merged_cluster[2])
                    file.write('n_spikes = %d\n' % merged_cluster[3])
                    file.write('raw_path = %s\n' % ('r"'+path+'"'))
                
        else :
            print('# No export to /results/%s, the folder already exists #\n' % pipeline_name)
   
     
# --------------------------------------------------------------
#
# --------------------------------------------------------------        
def regroup_clusters(group_name, folders_list) :
    '''
    Exports the data from results to groups, to have every insertions under one group folder
    '''
    
    print('# Exporting clusters to /results/%s #\n' % group_name)
    
    if not os.path.exists('./results/%s/clusters' % group_name) :
        os.makedirs('./results/%s/clusters' % group_name)
        
    for folder_name in tqdm(folders_list) :
        parent_folder = './results/%s/' % folder_name
        clusters_folders = [os.path.join(parent_folder, clust) for clust in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, clust))]
        
        for clust_fold in clusters_folders :
            try :
                shutil.copytree(clust_fold, './results/%s/clusters/%s' % (group_name, clust_fold.split('/')[-1]))
            except FileExistsError : #overwrite behaviour
                if gprm.overwrite :
                    shutil.rmtree('./results/%s/clusters/%s' % (group_name, clust_fold.split('/')[-1]))
                    shutil.copytree(clust_fold, './results/%s/clusters/%s' % (group_name, clust_fold.split('/')[-1]))
        