#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import utils 
import numpy as np 
from skimage.feature import hog
import MotionClouds as mc

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings 
warnings.catch_warnings()
warnings.simplefilter("ignore")

def compute_hog(image, N_thetas=33,feature_vector=False) :
    image = utils.norm_data(image)
    fd = hog(image, orientations=N_thetas, pixels_per_cell=(32, 32),
                        cells_per_block=(1, 1), visualize=False, multichannel=True,
                       feature_vector = feature_vector, block_norm='L2', transform_sqrt = False)
    if feature_vector:
        return fd
    else:
        means = []
        for x in range(0, fd.shape[0]) :
            for y in range(0, fd.shape[1]) :
                means.append(fd[x,y,:,:,:])

        means = np.asarray(means)
        means = np.mean(means, axis = 0).ravel()
        return means
    
def plot_hog(hog_vals, ax,
            roll, color, do_label, alpha=1, edgecolor=None) :
    hog_vals = np.roll(hog_vals, roll)
    #hog_vals = utils.norm_data(hog_vals)
        
    xs = np.linspace(0, np.pi, len(hog_vals))
    ax.bar(xs, hog_vals, align = 'center', width = xs[1] - xs[0],
        facecolor=color, edgecolor=edgecolor, alpha=alpha, lw=2)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.grid(True, which = 'major', axis = 'y', linestyle = '--')
    ax.set_axisbelow(True)
    
    if do_label :
        ax.set_xticks([0, np.pi/4,  np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['-90°', '-45°', r'$\theta_{0}$', '+45°', '+90°'])
        ax.set_xticklabels(['0', '45', '90', '135', '180'])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel('Orientation (°)', fontsize = 14)
        ax.set_ylabel('Frequency', fontsize = 14)
        
    else : 
        ax.set_xticks([0, np.pi/4,  np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels([])
        
    ax.set_ylim(np.min(hog_vals), np.max(hog_vals)+.01)
    return ax

def generate_cloud(theta, b_theta, phase,
                N_X, N_Y, seed, contrast=1., sf_0=0.06,
                B_sf = 0.06):
    
    
    fx, fy, ft = mc.get_grids(N_X, N_Y, 1)
    disk = mc.frequency_radius(fx, fy, ft) < .5

    if b_theta == 0 : 
        mc_i = mc.envelope_gabor(fx, fy, ft,
                                V_X=0., V_Y=0., B_V=0.,
                                sf_0=sf_0, B_sf=B_sf,
                                theta=0, B_theta=b_theta)
        mc_i = np.rot90(mc_i)
    else :
        mc_i = mc.envelope_gabor(fx, fy, ft,
                                V_X=0., V_Y=0., B_V=0.,
                                sf_0=sf_0, B_sf=B_sf,
                                theta=theta, B_theta=b_theta)
        
    im_ = np.zeros((N_X, N_Y, 1))
    im_ += mc.rectif(mc.random_cloud(mc_i, seed=seed),
                    contrast=2)
    im_ += -.5
    return im_[:,:,0]
