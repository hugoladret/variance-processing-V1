#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import pipeline_params as prm
from analysis import pipeline_waveform as waveform
from analysis import pipeline_photodiode as photodiode
from analysis import pipeline_mc as mc
from analysis import pipeline_fit as fit
from analysis import pipeline_plotter as plotter
from utils import pipeline_utils


print('#################################')
print('#  Starting analysis pipeline   #')
print('#################################\n')       

# Creating /result folders and subfolders
pipeline_utils.export_to_results()

print('###############')
print('#  Analyzing  #')
print('###############\n')   
      
# Classifies the waveforms
if prm.do_wav :
    waveform.waveform_analysis()    
        
# Extract the photodiode data 
if prm.do_photodiode :
    photodiode.export_sequences_times()
    
# Does the full package analysis for MotionClouds    
if prm.do_mc : 
    mc.mc_analysis()
    
# And fit the curves from the previous analysis
if prm.do_fit :
    fit.fit_curves()
    
print('##############')
print('#  Plotting  #')
print('##############\n\n')   
      
if prm.do_idcard : 
    plotter.create_ID_card()