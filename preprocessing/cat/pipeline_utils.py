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
import params as prm
import glob

# --------------------------------------------------------------
#
# --------------------------------------------------------------
def export_to_results():
    '''
    Exports the data to results folder/pipeline_name subfolder/clusters folders
    '''

    for folder in prm.folder_list :
        pipeline_name = folder
        if not os.path.exists('./processed/%s/' % pipeline_name):

            
            path = './raw_data/'+pipeline_name
            resultpath = './processed/'+pipeline_name
            print('# Exporting data from %s to %s #\n' % (path, resultpath))

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
def do_cleanup() :
    '''
    Exports the data from results to groups, to have every insertions under one group folder
    '''

    if prm.do_export :
        print('# Exporting clusters to /results/%s #\n' % prm.group_name)

        if not os.path.exists('./processed/%s/clusters' % prm.group_name) :
            os.makedirs('./processed/%s/clusters' % prm.group_name)

        for folder_name in tqdm(prm.folder_list) :
            parent_folder = './processed/%s/' % folder_name
            clusters_folders = [os.path.join(parent_folder, clust) for clust in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, clust))]

            for clust_fold in clusters_folders :
                try :
                    shutil.copytree(clust_fold, './processed/%s/clusters/%s' % (prm.group_name, clust_fold.split('/')[-1]))
                except FileExistsError : #overwrite behaviour
                    shutil.rmtree('./processed/%s/clusters/%s' % (prm.group_name, clust_fold.split('/')[-1]))
                    shutil.copytree(clust_fold, './processed/%s/clusters/%s' % (prm.group_name, clust_fold.split('/')[-1]))
                    
                    
    #Excludes clusters based on criterions
    if prm.do_exclude :
        print('# Excluding clusters... #')
        
        clusters_path = './processed/%s/clusters/' % prm.group_name
        clusters_list = os.listdir(clusters_path)
        
        total = len(clusters_list) # total clusters before cleaning
        excluded = np.zeros(2) # we have 5 criterions of exclusion
        
        for cluster in tqdm(clusters_list) :
            if not os.path.isdir(clusters_path + cluster) :
                continue 
            spiketimes = np.load(clusters_path + cluster + '/spiketimes.npy')
            r_squared = np.load(clusters_path + cluster + '/0.000_plot_neurometric_merged_fit_reports.npy')
            
            # Firing rate dropping exclusion --------------------------------------
            # Receipe at https://stackoverflow.com/questions/57712650/
            hist = np.histogram(spiketimes, bins = 1800)[0]
            below_mask = np.r_[False, hist < prm.drop_value, False]
            idx = np.flatnonzero(below_mask[:-1]!=below_mask[1:])
            lens = idx[1::2]-idx[::2]
            drops = lens >= prm.drop_duration
            if np.any(drops) :
                excluded[0] +=1 
                shutil.rmtree(clusters_path + cluster)
                continue
            
            # Goodness of fit exclusion, based on the merged TC --------------------------------------
            if r_squared[0] < prm.min_R2 :
                excluded[1] +=1 
                shutil.rmtree(clusters_path + cluster)
                continue
        
        print('Out of %s total neurons, excluded %s via firing rate, %s via goodness of fit.\nTotal excluded = %s/%s, remaining : %s' 
            % (total, int(excluded[0]), int(excluded[1]), int(np.sum(excluded)), total, (total-int(np.sum(excluded)))) )
    
    
    
    '''
    Deletes some files/aliases some others, for easier use in post-processing
    This is mostly because I do not want to rewrite the entirety of a 3 years old pipeline, include variable names :)))
    '''
    
    if prm.do_rename :
        clusters_path = './processed/%s/clusters/' % prm.group_name
        clusters_list = os.listdir(clusters_path)
        
        for cluster in tqdm(clusters_list) :
            fullpath = clusters_path + cluster + '/'
            
            for file_path in os.listdir(fullpath) : 
                if file_path.startswith('0.025_') : # Deleting all files of the form 0.025_*
                    os.remove(fullpath+file_path)
                if file_path.startswith('0.000_') : # Renaming all those of the other direction 
                    os.rename(fullpath+file_path, fullpath+file_path.replace('0.000_', ''))
                    
                
            # Removing useless files 
            os.remove(fullpath + 'r2_fit.npy')
            os.remove(fullpath + 'plot_neurometric_fit_reports.npy')
            os.remove(fullpath + 'plot_neurometric_merged_fit_reports.npy')
            
            os.remove(fullpath + 'plot_MC_PSTH_merged.npy')
            
            os.remove(fullpath + 'plot_MC_TC_merged_all.npy')
            os.remove(fullpath + 'plot_MC_TC_merged_means.npy')
            os.remove(fullpath + 'plot_MC_TC_merged_stds.npy')
            os.remove(fullpath + 'plot_MC_TC_nonmerged_means.npy')
            os.remove(fullpath + 'plot_MC_TC_nonmerged_stds.npy')
            os.remove(fullpath + 'plot_spiketime_density.npy')
            os.remove(fullpath + 'plot_neurometric_fit_params.npy')
            
            # Renaming the PSTH and TC files 
            os.rename(fullpath + 'plot_MC_PSTH_nonmerged.npy',
                    fullpath + 'PSTH.npy')
            os.rename(fullpath + 'plot_MC_TC_nonmerged_all.npy',
                    fullpath + 'TC.npy')
            os.rename(fullpath + 'plot_neurometric_fitted_TC.npy',
                    fullpath + 'fitted_TC.npy')
            os.rename(fullpath + 'plot_neurometric_merged_fitted_TC.npy',
                    fullpath + 'fitted_TC_merged.npy')
            os.rename(fullpath + 'plot_neurometric_merged_Btheta_fits.npy',
                    fullpath + 'hwhh_merged.npy')
            
            # Renaming the neurometric variables
            cv_fit = np.load(fullpath + 'cirvar_fit.npy', allow_pickle = True)
            cv_fit = cv_fit.item()
            cv_r2 = np.load(fullpath + 'cirvar_fitr2.npy', allow_pickle = True )
            cv_fit['r2'] = cv_r2
            np.save(fullpath + 'cirvar_fit.npy', cv_fit)
            os.remove(fullpath + 'cirvar_fitr2.npy')

            hwhh_fit = np.load(fullpath + 'phi_fit.npy', allow_pickle = True)
            hwhh_fit = hwhh_fit.item()
            hwhh_r2 = np.load(fullpath + 'phi_fitr2.npy', allow_pickle = True )
            hwhh_fit['r2'] = hwhh_r2
            np.save(fullpath + 'hwhh_fit.npy', hwhh_fit)
            os.remove(fullpath + 'phi_fitr2.npy')
            os.remove(fullpath + 'phi_fit.npy')
            os.rename(fullpath + 'plot_neurometric_Btheta_fits.npy',
                    fullpath + 'hwhh.npy')
            
            # rename this first 
            fitted_TC = np.load(fullpath + 'fitted_TC.npy', allow_pickle = True)
            rmaxs = np.max(fitted_TC, axis = 1)[::-1]
                
            rmax_fit = np.load(fullpath + 'rmax_fit.npy', allow_pickle = True)
            rmax_fit = rmax_fit.item()
            rmax_r2 = np.load(fullpath + 'rmax_fitr2.npy', allow_pickle = True )
            rmax_fit['r2'] = rmax_r2
            np.save(fullpath + 'rmax_fit.npy', rmax_fit)
            np.save(fullpath + 'rmax.npy', rmaxs)
            os.remove(fullpath + 'rmax_fitr2.npy')
