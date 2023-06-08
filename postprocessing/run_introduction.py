#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
Creates the introductory figure, describing natural images and making MotionClouds
"""

import params as prm
import utils 
import utils_introduction as utils_intro

import imageio
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import numpy as np 


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings 
warnings.catch_warnings()
warnings.simplefilter("ignore")

# --------------------------------------------------------------
# HOG on calanques
# --------------------------------------------------------------
def make_intro_img() :
    # Showing the image
    img = imageio.imread('./figs/callanques.jpg')
    x0, y0 = 650, 1200
    sizex, sizey = 3200, 2000
    img = img[y0:y0+sizey, x0:x0+sizex, :]

    fig, ax = plt.subplots(figsize = (8,8))
    ax.imshow(img)

    coos = [(550, 50), # border 
            (450, 2000),
            (1300, 2600), # boat
            (1750, 1050), # tree 
            ]
    width = 200

    cols = plt.cm.inferno([.5, .85, .2, .65, ])
    for i in range(len(coos)) :
        rect = Rectangle( (coos[i][1],coos[i][0]), width, width,
                        linewidth=2, edgecolor=cols[i], facecolor='none',
                        linestyle = '--')
        ax.add_patch(rect)
        
    plt.tight_layout()
    ax.axis('off')
    fig.savefig('./figs/intro_image.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)

    # Making the HOGs
    for i, coo in enumerate(coos) :
        fig, ax = plt.subplots(figsize = (4,4))
        
        hog_vals = utils_intro.compute_hog(img[coo[0]:coo[0]+width,
                                coo[1]:coo[1]+width],
                            N_thetas = 27)
        
        hog_plt = utils_intro.plot_hog(hog_vals, ax, roll = 0, color = cols[i], do_label = False)
        hog_plt.set_ylim(0, .6)
        
        ax.set_yticks([0, .3, .6])
        plt.tight_layout()
        if i != len(coos)-1 :
            ax.set_yticklabels([])
        else :
            ax.set_xticklabels(['0', '45', '90', '135', '180'])
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlabel('Orientation (°)', fontsize = 14)
            ax.set_ylabel('Frequency', fontsize = 14)
        fig.savefig('./figs/intro_image_hog_%s.pdf' %i, bbox_inches='tight', dpi=200, transparent=True)
        plt.show(block = prm.block_plot)
    
    
    
# --------------------------------------------------------------
# Generating MotionClouds and making HOGs
# --------------------------------------------------------------
def make_mc() :
    thetas_hog = np.linspace(0, np.pi, 27)
    mcs = []
    for ibt, bt in enumerate(prm.B_thetas) :
        fig, ax = plt.subplots(figsize = (4,4))
        
        img = utils_intro.generate_cloud(theta = np.pi/2, b_theta = bt, phase = 0,
                    N_X = 512, N_Y = 512, seed = 42, contrast=2.)
        
        hog_vals = utils_intro.compute_hog(img[:,:, None], 27)
        hog_plt = utils_intro.plot_hog(hog_vals, ax, roll = 17, color = prm.colors[ibt], do_label = True)
        hog_plt.set_ylim(0, .6)
        
        arr = hog_vals
        R = np.sum( (arr * np.exp(2j*thetas_hog)) / np.sum(arr) )
        cirvar = 1 - np.abs(np.real(R))
        
        ax.set_yticks([0, .3, .6])
        if ibt < len(prm.B_thetas)-1 :
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        fig.tight_layout()
        fig.savefig('./figs/intro_MC_hog_%s_distrib.pdf' % ibt, bbox_inches='tight', dpi=200, transparent=True)
        plt.show(block = prm.block_plot)
        mcs.append(img)
        
    B_thetas_labs = np.linspace(np.pi/2, 0.0, 8)/ 2.5
    fig, axs = plt.subplots(figsize = (16,2), nrows = 1, ncols = 8,
                        gridspec_kw = {'wspace':0.5, 'hspace':0.05})
    for ibt, i0 in enumerate([0, 1, 2, 3, 4, 5, 6, 7]) :
        ax = axs[ibt]
        img = mcs[ibt]
        im = ax.imshow(img, cmap = 'gray', interpolation = 'bilinear')
        ax.text(25, -25, r'$B_\theta=%.1f°$'% (B_thetas_labs[i0] * 180/np.pi),
                 color = prm.colors[i0], fontsize = 16, rotation = 0)
        im.set_clim(-1,1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig('./figs/intro_MC.pdf', bbox_inches='tight', dpi=200, transparent=True)
    
    
# --------------------------------------------------------------
# Generating a Cirvar/Btheta plot
# --------------------------------------------------------------
def make_cv() :
    thetas_hog = np.linspace(0, np.pi, 27)
    cv_list = []
    for ibt, bt in enumerate(np.linspace(0.001, prm.B_thetas.max(), 50)) :
        
        img = utils_intro.generate_cloud(theta = np.pi/2, b_theta = bt, phase = 0,
                    N_X = 512, N_Y = 512, seed = 42, contrast=2.)
        hog_vals = utils_intro.compute_hog(img[:,:, None], 27)

        arr = hog_vals
        R = np.sum( (arr * np.exp(2j*thetas_hog)) / np.sum(arr) )
        cirvar = 1 - np.abs(np.real(R))

        cv_list.append(cirvar)
        
    fig, ax = plt.subplots(figsize = (5,3))
    ax.plot(np.linspace(prm.B_thetas.min(), prm.B_thetas.max(), 50)*180/np.pi, cv_list)
    ax.set_xlabel(r'$B_\theta$ (°)')
    ax.set_ylabel('Circular variance')
    fig.tight_layout()
    fig.savefig('./figs/intro_CV_bt_curve.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.show(block = prm.block_plot)