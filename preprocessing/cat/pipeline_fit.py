#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import os


import numpy as np
from lmfit import Model, Parameters

from scipy.stats import linregress
from scipy.special import i0 as I0

from tqdm import tqdm

import params as prm

import torch

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 
# --------------------------------------------------------------

def fit_curves() :
    '''
    Fits curves from previously exported data in pipeline_mc.py 
        - Fits tuning curve using either lmfit or torch 
        - Fits Naka Ruhston curves to HWHH and CirVar 
        - Fits a delayed normalization model to PSTHs
    '''
    
    print('# Running fitting procedures #')
    
    for folder in prm.folder_list :
        fit_tuning_curves(folder)
        fit_psychometrics(folder)
        
        print('# Curve fitting done ! #\n')

       
            
# --------------------------------------------------------------
# 
# --------------------------------------------------------------    
def fit_tuning_curves(folder) :
    '''
    Loads the TC data from the cluster folder
    and fit a VonMises function to it, using either lmfit or PyTorch (poisson)
    '''
    
    print('\nFitting tuning curve...')
          
    folder_path = './processed//%s/' % folder
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    for cluster_folder in tqdm(clusters_folders) :
        for phase in prm.phases :
            TC_data = np.load(folder_path + cluster_folder + '/%.3f_plot_MC_TC_nonmerged_all.npy'% phase)
            
            B_theta_fit_list, tuning_fits_list, fit_reports_list, params_fit_list = [], [], [], []

                 
            # Btheta Non merged TC
            TC_data = np.load(folder_path + cluster_folder + '/%.3f_plot_MC_TC_nonmerged_means.npy'% phase)
            
            if prm.fit_type == 'lmfit' :
                for it_0, btheta in enumerate(TC_data) :
                    mean_fr = TC_data[it_0]
                    xs = np.linspace(0, prm.thetas[-1], 1000, endpoint = False)
                    
                    best_vals, fit_report= lmfit_tc(mean_fr)
                    
                    tc = tuning_function(x=xs,
                                        mu=best_vals['mu'], fmax=best_vals['fmax'],
                                        kappa=best_vals['kappa'], bsl = best_vals['bsl'])
                    tuning_fits_list.append(tc)
                    fit_reports_list.append(fit_report)
                    #B_theta_fit_list.append(get_angle_opening(xs, tc))
                    kappa_val = np.arccos((np.log(.5) + best_vals['kappa']) / best_vals['kappa']) * 180 / np.pi
                    if np.isnan(kappa_val) :
                        kappa_val = 90
                    B_theta_fit_list.append(kappa_val)
                    params_fit_list.append(best_vals)
                    
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_fitted_TC.npy' % phase, tuning_fits_list)
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_fit_reports.npy' % phase, fit_reports_list)
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_Btheta_fits.npy' % phase, B_theta_fit_list)
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_fit_dict.npy' % phase, params_fit_list)


            elif prm.fit_type == 'Torch' :
                theta0, R0, Rmax, kappas, fitted_fr, ploss = torch_fit(TC_data)
                hwhh = .5*np.arccos(1+ np.log((1+np.exp(-2*kappas))/2)/kappas)
                hwhh = hwhh * 180 / np.pi
                fit_params = np.array([theta0, R0, Rmax, kappas])
                
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_fitted_TC.npy' % phase, fitted_fr)
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_fit_reports.npy' % phase, np.repeat(ploss, len(prm.B_thetas)))
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_Btheta_fits.npy' % phase, hwhh)
                np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_fit_params.npy' % phase, fit_params)
                
            
            else :
                print('Fitting algorithm not implemented !')
                raise
                
            
    
            # Btheta merged TC fit, 
            # no need for torch here, it's not even used in the results
            merged_TC_data = np.load(folder_path + cluster_folder + '/%.3f_plot_MC_TC_merged_all.npy' % phase)
            B_theta_fit_list, tuning_fits_list, fit_reports_list = [], [], []
            
            mean_fr = np.mean(merged_TC_data, axis = 1)
            best_vals, fit_report = lmfit_tc(mean_fr)
            fit_reports_list.append(fit_report)
            
            xs = np.linspace(0, prm.thetas[-1], 1000, endpoint = False)
            tc = tuning_function(x=xs,
                mu=best_vals['mu'], fmax=best_vals['fmax'],
                kappa=best_vals['kappa'], bsl = best_vals['bsl'])
            tuning_fits_list.append(tc)
            kappa_val = np.arccos((np.log(.5) + best_vals['kappa']) / best_vals['kappa']) * 180 / np.pi
            if np.isnan(kappa_val) :
                kappa_val = 90
            B_theta_fit_list.append(kappa_val)
            
            np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_merged_Btheta_fits.npy'% phase, B_theta_fit_list)
            np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_merged_fitted_TC.npy'% phase, tuning_fits_list)
            np.save(folder_path + cluster_folder + '/%.3f_plot_neurometric_merged_fit_reports.npy'% phase, fit_reports_list)



def fit_psychometrics(folder) :
    '''
    Fit Naka Rushton functions to Circular Variance and HWHHs
    Fit linear regression to poisson loss / r2
    '''
    print('\nFitting Naka-Rushtons...')
    
    folder_path = './processed//%s/' % folder
    clusters_folders = [file for file in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, file))]
    
    for cluster_folder in tqdm(clusters_folders) :
        for phase in prm.phases :
            
            # Fitting NKR to CirVar
            cvs = np.load(folder_path + cluster_folder + '/%.3f_cirvar.npy' % phase)
            arr_fit, r2 = fit_nkr(cvs)
            np.save(folder_path + cluster_folder + '/%.3f_cirvar_fit.npy' % phase, arr_fit)
            np.save(folder_path + cluster_folder + '/%.3f_cirvar_fitr2.npy' % phase, r2)

            # Fitting linear regression to poisson loss / r2, depending on the lmfit method
            fit_goodness = np.load(folder_path + cluster_folder + '/%.3f_plot_neurometric_fit_reports.npy' % phase)
            arr_fit = linregress(prm.B_thetas *  180 / np.pi, fit_goodness)
            np.save(folder_path + cluster_folder + '/%.3f_r2_fit.npy' % phase, arr_fit)
        
            # Fitting NKR to HWHH
            hwhh = np.load(folder_path + cluster_folder + '/%.3f_plot_neurometric_Btheta_fits.npy' % phase)
            arr_fit, r2 = fit_nkr(hwhh)
            np.save(folder_path + cluster_folder + '/%.3f_phi_fit.npy' % phase, arr_fit)
            np.save(folder_path + cluster_folder + '/%.3f_phi_fitr2.npy' % phase, r2)
            
            # Fitting NKR to Rmax
            fitted_TC = np.load(folder_path + cluster_folder + '/%.3f_plot_neurometric_fitted_TC.npy' % phase)
            Rmaxs = np.max(fitted_TC, axis = 1)[::-1]
            arr_fit, r2 = fit_nkr(Rmaxs)
            np.save(folder_path + cluster_folder + '/%.3f_rmax_fit.npy' % phase, arr_fit)
            np.save(folder_path + cluster_folder + '/%.3f_rmax_fitr2.npy' % phase, r2)
            
                
                
# --------------------------------------------------------------
# PyTorch functions
# --------------------------------------------------------------
def torch_fit(data, disable_cuda = False):
    # Fitting tuning curve with Torch
    
    # Torch init
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() :
        cuda = True
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    if disable_cuda or not torch.cuda.is_available() :
        cuda = False
        print('Disabling CUDA')
        torch.set_default_tensor_type("torch.DoubleTensor")
    criterion = torch.nn.PoissonNLLLoss(log_input=False, full=True, reduction="sum")
    
    log_R0 = np.log(np.min(data[0]))
    if np.isnan(log_R0): log_R0 = .5
    log_Rmax = np.log(np.max(data[-1]))
    if np.isnan(log_Rmax): log_Rmax = 1.
    # Params 
    theta0_init = prm.thetas[np.argmax(data[-1])] * torch.ones(1)
    log_R0_init = log_R0 * torch.ones(data.shape[0])
    log_Rmax_init = log_Rmax * torch.ones(data.shape[0])
    log_kappa_init = -1 * torch.ones(data.shape[0])
    
    # And fitting for the rest
    nvm_model, ploss = torch_fit_data(theta = prm.thetas,
                                    fr = data, 
                                    criterion=criterion,
                                    learning_rate = prm.learning_rate,
                                    num_epochs = prm.num_epochs,
                                    theta0_init = theta0_init,
                                    log_R0_init = log_R0_init,
                                    log_Rmax_init = log_Rmax_init,
                                    log_kappa_init = log_kappa_init, 
                                    N_B_theta = len(prm.B_thetas),
                                    verbose=False)
    # Getting the parameters back
    theta0, R0, Rmax, kappas = get_params(nvm_model, verbose=False, cuda = cuda)
    
    theta_ = np.linspace(0, prm.thetas[-1], 256, endpoint=True)
    Theta_ = torch.Tensor(theta_).unsqueeze(0) * torch.ones((len(prm.B_thetas), 1)) 
    fr_pl = nvm_model(torch.Tensor(theta_)).detach()
    fitted_fr = nvm_model(Theta_).detach()
    if cuda : 
        fitted_fr = fr_pl.cpu()
    fitted_fr = fitted_fr.numpy()
    
    return theta0, R0, Rmax, kappas, fitted_fr, ploss
    
def torch_fit_data(
                theta,
                fr,
                criterion,
                learning_rate,
                num_epochs,
                theta0_init,
                log_R0_init,
                log_Rmax_init,
                log_kappa_init,
                N_B_theta,
                verbose=False,
                **kwargs
                ):

    Theta = torch.Tensor(theta).unsqueeze(0) * torch.ones((N_B_theta, 1))
    FR = torch.Tensor(fr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Theta, FR = Theta.to(device), FR.to(device)

    nvm_model = NeuralVonMisesModel(theta0_init=theta0_init,
                                    log_R0_init=log_R0_init,
                                    log_Rmax_init=log_Rmax_init,
                                    log_kappa_init=log_kappa_init)
    nvm_model = nvm_model.to(device)
    
    nvm_model.train()
    optimizer = torch.optim.Adam(nvm_model.parameters(), lr=learning_rate)
    for epoch in range(int(num_epochs)):
        nvm_model.train()
        losses = []
        outputs = nvm_model(Theta)
        loss = criterion(outputs, FR)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (epoch % (num_epochs // 32) == 0):
            print(f"Iteration: {epoch} - Loss: {np.sum(losses)/len(theta):.5f}")

    nvm_model.eval()

    outputs = nvm_model(Theta)
    loss = criterion(outputs, FR).item() / len(theta)
    return nvm_model, loss

class NeuralVonMisesModel(torch.nn.Module):
    def __init__(self, 
                theta0_init,
                log_R0_init, 
                log_Rmax_init, 
                log_kappa_init,
            ):
        super(NeuralVonMisesModel, self).__init__()
        #N_B_theta = log_kappa_init.shape.numel()
        self.log_R0 = torch.nn.Parameter(log_R0_init.unsqueeze(1))
        self.log_Rmax = torch.nn.Parameter(log_Rmax_init.unsqueeze(1)) 
        self.log_kappa = torch.nn.Parameter(log_kappa_init.unsqueeze(1)) 
        if theta0_init.shape.numel() > 1:
            self.theta0 = torch.nn.Parameter(theta0_init.unsqueeze(1))
        else:
            self.theta0 = torch.nn.Parameter(theta0_init)
        
    def forward(self, theta):
        theta0 = self.theta0
        R0 = self.log_R0.exp()
        Rmax = self.log_Rmax.exp()
        kappa = self.log_kappa.exp()
        out = R0 + (Rmax - R0) * (kappa * (torch.cos(2*(theta-theta0)) - 1)).exp()
        return out
    
def get_params(nvm_model, verbose=False, cuda = False):
    '''
    Get back the data from a fitted model
    '''
    theta0_ = nvm_model.theta0.item()
    theta0_ = np.mod(theta0_, np.pi)
    
    if cuda : 
        R0_ = torch.exp(nvm_model.log_R0).squeeze().cpu().detach().numpy()
        Rmax_ = torch.exp(nvm_model.log_Rmax).squeeze().cpu().detach().numpy()
        kappa_ = torch.exp(nvm_model.log_kappa).squeeze().cpu().detach().numpy()
    else :
        R0_ = torch.exp(nvm_model.log_R0).squeeze().detach().numpy()
        Rmax_ = torch.exp(nvm_model.log_Rmax).squeeze().detach().numpy()
        kappa_ = torch.exp(nvm_model.log_kappa).squeeze().detach().numpy()

    return theta0_, R0_, Rmax_, kappa_
    
# --------------------------------------------------------------
# lmfit functions for tuning curves and NKR
# --------------------------------------------------------------
def tuning_function(x, mu, kappa, fmax, bsl):
    # Von Mises, with kappa the concentration, mu the location, I0 Bessel order 0
    # fmax the firing rate at pref ori, bsl the min firing rate (not the baseline, which was substracted)    
    return  bsl + (np.exp((kappa)*np.cos((x-mu)))/(2*np.pi*I0(kappa))) * fmax

def nakarushton(x, rmax, c50, b, n):
    nkr = b + (rmax-b) * ((x**n) / (x**n + c50**n))
    return nkr

    
def lmfit_tc(array):
    x = np.linspace(0, np.pi, len(array), endpoint = False)
    y = array
    
    mod = Model(tuning_function)
    pars = Parameters()
    pars.add_many(('mu', np.argmax(y), True,  0.0, np.argmax(y) * 3 + .001),
                ('kappa', 1., True,  0.01, 20),
                ('fmax', np.max(y), True,  0.0, 80.),
                  ('bsl', np.min(y), True,  0.0, 1e-5 + (np.min(y) * 1.5)))

    out = mod.fit(y, pars, x=x, nan_policy='omit', fit_kws = {'maxfev' : 2000})

    return out.best_values, np.abs(1-out.residual.var() / np.var(y))

# --------------------------------------------------------------
# lmfit functions for naka rushton 
# --------------------------------------------------------------
def fit_nkr(array) :
    x = np.linspace(0, 1, len(array))
    y = array[::-1]
    
    mod = Model(nakarushton)
    pars = Parameters()
    
    pars.add_many(('rmax',  np.max(y), True,  0.0,  100.),
            ('c50', .5, True,  0.001, 10.),
            ('b', y[0], True, y[0] * .5 + .001, y[0] * 3 + .002 ),
            ('n', 4, True,  1., 100.))
    
    out = mod.fit(y, pars, x=x, nan_policy='omit', fit_kws = {'maxfev' : 2000})
    return out.best_values, np.abs(1-out.residual.var() / np.var(y))