#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:10:30 2020

@author: hugo
"""

import numpy as np

import group_params as prm

import fileinput
import os
import sys
import imp

from tqdm import tqdm

from sklearn import cluster
from scipy.stats import spearmanr
from scipy.special import i0 as I0
import torch


# --------------------------------------------------------------
# Spearman correlation of NKR params
# --------------------------------------------------------------
def spearman_nkr(var = 'Phi') :
    print('# Analyzing correlation of NKR params #\n')
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    nkr_list = []
    for clust in clusters_list :
        if var == 'CirVar' :
            nkr_params = np.load(clusters_path + clust + '/0.000_cirvar_fit.npy', allow_pickle = True)
            nkr_params = nkr_params.item()
            nkr_list.append([nkr_params['b'],  nkr_params['c50']])
        if var == 'Phi' :
            nkr_params = np.load(clusters_path + clust + '/0.000_phi_fit.npy', allow_pickle = True)
            nkr_params = nkr_params.item()
            nkr_list.append([nkr_params['b'], nkr_params['c50']])
    
    baselines = [x[0] for x in nkr_list]
    b50s = np.asarray([x[1] for x in nkr_list]) * (prm.B_thetas.max() * 180 / np.pi)

    r, pval = spearmanr(baselines, b50s)    
    
    np.save('./results/%s/spearmanr_%s.npy' % (prm.group_name, var), [r, pval])

# --------------------------------------------------------------
# Waveform analysis
# --------------------------------------------------------------
def waveform_analysis() :
    print('# Analyzing waveform #\n')
    
    x1, y1, x2, y2 = kmean_waveforms()
    np.save('./results/%s/waveforms_clusters.npy' % prm.group_name, [x1, y1, x2, y2])
    
def kmean_waveforms():
    '''
    Perform k-mean clustering from the available waveform caracterisation points in /results
    '''
    unscaled_to_ms = prm.window_size / prm.fs
    unscaled_to_ms *= 1000
    
    all_carac_points = []
    path_to_carac_points = [] #use to write the kmeans info
    
    folder_path = './results/%s/clusters/' % prm.group_name
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
         
    kmeans = cluster.KMeans(n_clusters = 2, init = 'k-means++',
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
            
    return xs1, ys1, xs2, ys2
        
     
# --------------------------------------------------------------
# Information theory
# --------------------------------------------------------------
def pop_fisher_info():
    '''
    Computes the average maximum fisher information encoded by the population
    for each bthetas
    '''
    print('# Computing the fisher information #\n')
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    params_list = []
    for clust in clusters_list :
        params = np.load(clusters_path + clust + '/0.000_plot_neurometric_fit_params.npy', allow_pickle = True)
        
        _ = []
        for btheta in range(len(prm.B_thetas)) :
            mu = params[0]
            R0 = params[1][btheta]
            Rmax = params[2][btheta]
            kappa = params[3][btheta]
            _.append([mu, R0, Rmax, kappa])
        params_list.append(_)
        
    xs = np.linspace(0, prm.thetas[-1], 1000, endpoint = False)
    neurons_infos = []
    for btheta in range(len(prm.B_thetas)) :
        _ = []
        for neuron in range(len(clusters_list)):
            pars = params_list[neuron][btheta]
            
            tc = tuning_function(xs, pars[0], pars[3], pars[2], pars[1])
            tc_prime = derivative_tc(xs, pars[0], pars[3], pars[2], pars[1])
    
            info = (tc_prime **2) / ( tc) 
            
            _.append(np.mean(info))
        
        neurons_infos.append(_)

    np.save('./results/%s/pop_fisher_info.npy' % prm.group_name, neurons_infos)

    
    
    
# --------------------------------------------------------------
# Decoders
# --------------------------------------------------------------
def jazayeri_orientation_decoding():
    '''
    Decodes the log-likelihood orientation of the population using a PID hypothesis,
    based on Jazayeri & Movshon https://www.nature.com/articles/nn1691
    '''
    print('# Decoding orientation from population #\n')
    
    clusters_path = './results/%s/clusters/' % prm.group_name
    clusters_list = os.listdir(clusters_path)
    
    for exp in prm.exp_list :
        all_errors = []
        all_likelihoods = []
        for i0, btheta in enumerate(prm.B_thetas) : 
            
            # Loading data
            tcs_list = []
            for clust in clusters_list :
                if exp in clust :
                    tc = np.load(clusters_path + clust + '/0.000_plot_MC_TC_nonmerged_means.npy')
                    tc[tc<0] = 1. # problem with baseline removal, decoding assumes 1spk/stim
                    tcs_list.append(tc)
            tcs_arr = np.asarray(tcs_list)
            
            # ordering
            pref_thetas = []
            for neuron in range(tcs_arr.shape[0]) :
                tc = tcs_arr[neuron, -1, :] #Preferred orientation is computed on btheta = 0
                pref_thetas.append(np.argmax(tc))
            pref_thetas = np.asarray(pref_thetas)
            
            # Counts for each preferred orientation the number of neurons        
            sorted_prefs = sorted(pref_thetas)
            unique, counts = np.unique(sorted_prefs, return_counts = True)
            #print(counts)
    
            
            # concatenating on orientations
            ordered_neuron_list = []
            for ori in range(0,len(prm.thetas)) :
                idxs = np.where(np.asarray(pref_thetas) == ori)[0] #check where all neurons with preferred ori are
                for idx in idxs :
                    ordered_neuron_list.append(tcs_arr[idx, i0, :]) # <---------
            ordered_arr = np.asarray(ordered_neuron_list)
            #print(ordered_arr.shape)

            #ordered_arr = ordered_arr / counts[np.newaxis, :]
            #print(len(counts))
            # calculating
            errors = []
            likelihoods = []
            for stim_theta in range(0,len(prm.thetas)) :
                #print(ordered_arr.shape)
                #print(len(counts))
                #spikes =  np.asarray([x[stim_theta] for x in ordered_arr]) #n 
                spikes = ordered_arr[:,stim_theta]
                #logs = np.log(ordered_arr[:,stim_theta])
                # if len(counts) == 11 :
                    
                #     spikes *= (np.sum(counts) / counts[stim_theta-1])
                # else :
                #     spikes *= (np.sum(counts) / counts[stim_theta])#normalization by number of neuron for each pref tc ?
                # print(np.sum(counts))
                # print(counts[stim_theta-1])
                # print('--')
                logs = np.asarray([np.log10(x) for x in ordered_arr]) #log(fi(theta))
                log_likelihood = np.sum(spikes[:, None] * logs, axis = 0)
                err = np.abs(stim_theta - np.argmax(log_likelihood))
                
                errors.append(err)
                likelihoods.append(log_likelihood)
        
            all_errors.append(errors)
            all_likelihoods.append(likelihoods)
            
        if prm.deco_norm :
            arr = np.asarray(all_likelihoods)
            snrs = np.zeros_like(arr)
            for i0, btheta in enumerate(prm.B_thetas) : 
                for i1, theta in enumerate(prm.thetas) :
                    llk = arr[i0, i1, :]
                    norm_llk = (llk - llk.min()) / (llk.max() - llk.min())
                    arr[i0, i1, :] = norm_llk
                    snrs[i0, i1, :] = np.max(norm_llk) / np.mean(norm_llk)
                         
            np.save('./results/%s/%s_decoder_ori_likelihoods.npy' % (prm.group_name,exp), arr)
            np.save('./results/%s/%s_decoder_ori_snrs.npy' % (prm.group_name, exp), snrs[:,:,0])
        else :
            np.save('./results/%s/%s_decoder_ori_likelihoods.npy' % (prm.group_name,exp), all_likelihoods)
            
        np.save('./results/%s/%s_decoder_ori_errors.npy' % (prm.group_name,exp), all_errors)
    
    
def jazayeri_orientation_fitting():
    '''
        Fits a Von Mises on the population code to decode orientation more precisely
    '''

    print('# Fitting orientation decoders #\n')
    
    for exp in prm.exp_list :
        log_likelihoods = np.load('./results/%s/%s_decoder_ori_likelihoods.npy' % (prm.group_name,exp))
        log_likelihoods = np.flip(log_likelihoods, axis = 0 ) #easier to specify the indices in parameters this way
          
        N_bthetas = log_likelihoods.shape[0] 
        N_thetas = log_likelihoods.shape[1] 
        
        params_fit = np.zeros(N_bthetas * N_thetas, dtype = object).reshape(N_bthetas, N_thetas)
        tcs = np.zeros(N_bthetas * N_thetas, dtype = object).reshape(N_bthetas, N_thetas)
        errs = np.zeros(N_bthetas * N_thetas).reshape(N_bthetas, N_thetas)
        
        for btheta in tqdm(range(log_likelihoods.shape[0]), desc = 'Fitting %s experiments' % exp) :
            for stim_theta in range(log_likelihoods.shape[1]) :
                decoded_arr = log_likelihoods[btheta, stim_theta, :] 
                
                # fit the array
                theta0, R0, Rmax, kappa = fit_torch_jazayeri(decoded_arr)
                xfit = np.linspace(prm.thetas[0], prm.thetas[-1], 256)
                fit_tc = tuning_function(x = xfit,
                                         mu = theta0, kappa = kappa, fmax = Rmax, bsl = R0)
                err = np.abs(prm.thetas[np.argmax(decoded_arr)] - xfit[np.argmax(fit_tc)]) 
                             
                # save the array
                errs[btheta, stim_theta] = err
                params_fit[btheta, stim_theta] = np.array([theta0, R0, Rmax, kappa])
                tcs[btheta, stim_theta] = fit_tc
                
        
        np.save('./results/%s/%s_decoder_ori_params.npy' % (prm.group_name,exp), params_fit)
        np.save('./results/%s/%s_decoder_ori_tcs.npy' % (prm.group_name,exp), tcs)
        np.save('./results/%s/%s_decoder_ori_errs.npy' % (prm.group_name,exp), errs)


# --------------------------------------------------------------
# Info utils
# --------------------------------------------------------------
def tuning_function(x, mu, kappa, fmax, bsl):
    # Von Mises, with kappa the concentration, mu the location, I0 Bessel order 0
    # fmax the firing rate at pref ori, bsl the min firing rate (not the baseline, which was substracted)    
    return  bsl + (np.exp((kappa)*np.cos((x-mu)))/(2*np.pi*I0(kappa))) * fmax

def derivative_tc(x, mu, kappa, fmax, bsl) :
    up = kappa * np.sin(x - mu) * np.exp(kappa * np.cos (x - mu))
    down = 2 * np.pi * I0(kappa)
    return -((up / down) * fmax) + bsl

# --------------------------------------------------------------
# Torch utils
# --------------------------------------------------------------   
def fit_torch_jazayeri(data) :
    '''
    Single orientation decoded fitting
    '''

    class NeuralVonMisesModel(torch.nn.Module):
        def __init__(self, 
                     theta0,
                     log_R0, 
                     log_Rmax, 
                     log_kappa
                    ):
            super(NeuralVonMisesModel, self).__init__()
    
            self.theta0 = torch.nn.Parameter(theta0 * torch.ones(1))
            self.log_R0 = torch.nn.Parameter(log_R0 * torch.ones(1))
            self.log_Rmax = torch.nn.Parameter(log_Rmax * torch.ones(1)) 
            self.log_kappa = torch.nn.Parameter(log_kappa * torch.ones(1)) 
    
        def forward(self, theta):
            #self.theta0 = np.mod(self.theta0, np.pi)
            R0 = torch.exp(self.log_R0)
            Rmax = torch.exp(self.log_Rmax)
            kappa = torch.exp(self.log_kappa)
            out = R0 + (Rmax - R0) * torch.exp(kappa * (torch.cos(2*(theta-self.theta0)) - 1))
            return out
        
    # https://pytorch.org/docs/stable/nn.html?highlight=poisson#torch.nn.PoissonNLLLoss
    criterion_PL = torch.nn.PoissonNLLLoss(log_input=False, full=True, reduction="sum")
    learning_rate = 0.02
    beta1, beta2 = 0.9, 0.999
    betas = (beta1, beta2)
    num_epochs = 1024*2
    
    def fit_data(
                    theta,
                    fr,
                    criterion=criterion_PL,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    betas=betas,
                    verbose=False,
                    theta0 = np.pi/2,
                    log_R0 = 0.5,
                    log_Rmax=1.5,
                    log_kappa=-3.,                   
                    **kwargs
                    ):

        Theta, labels = torch.Tensor(theta[:, None]), torch.Tensor(fr[:, None])
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Theta, labels = Theta.to(device), labels.to(device)
    
        nvm_model = NeuralVonMisesModel(theta0=theta0,
                                        log_R0=log_R0,
                                        log_Rmax=log_Rmax,
                                        log_kappa=log_kappa
                                       )
        nvm_model = nvm_model.to(device)
        
        nvm_model.train()
        optimizer = torch.optim.Adam(nvm_model.parameters(), 
                                     lr=learning_rate, betas=betas)
        for epoch in range(int(num_epochs)):
            nvm_model.train()
            losses = []
            outputs = nvm_model(Theta)
            loss = criterion(outputs, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
            if verbose and (epoch % (num_epochs // 32) == 0):
                print(f"Iteration: {epoch} - Loss: {np.sum(losses)/len(theta):.5f}")
    
        nvm_model.eval()
        Theta, labels = torch.Tensor(theta[:, None]), torch.Tensor(fr[:, None])
        outputs = nvm_model(Theta)
        loss = criterion(outputs, labels).item() / len(theta)
        return nvm_model, loss
    
    def get_params(nvm_model, verbose=False):
        '''
        Get back the data from a fitted model
        '''
        theta0_ = nvm_model.theta0.item()
        theta0_ = np.mod(theta0_, np.pi)
    
        R0_ = torch.exp(nvm_model.log_R0).item()
        Rmax_ = torch.exp(nvm_model.log_Rmax).item()
        kappa_ = torch.exp(nvm_model.log_kappa).item()
    
        if verbose:
            print(f"theta0 = {theta0_:.3f}")
            print(f"R0 = {R0_:.3f}")
            print(f"Rmax = {Rmax_:.3f}")
            print(f"kappa = {kappa_:.3f}")
        return theta0_, R0_, Rmax_, kappa_


    if torch.cuda.is_available():
        cuda = True
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        cuda = False
        torch.set_default_tensor_type("torch.DoubleTensor")
    
    nvm_model_pl, ploss = fit_data(prm.thetas, data, criterion=criterion_PL, verbose=False,
                                   theta0=prm.thetas[np.argmax(data)], 
                                   log_R0=np.log(data.min()), 
                                   log_Rmax=np.log(data.max()),
                                  )
    theta0, R0, Rmax, kappas = get_params(nvm_model_pl, verbose=False)
    
    return theta0, R0, Rmax, kappas
# --------------------------------------------------------------
# Misc utils
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
            
def get_var_from_file(filename):
    f = open(filename)
    cluster_info = imp.load_source('cluster_info', filename)
    f.close()
    
    return cluster_info



