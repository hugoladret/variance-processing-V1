#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:44:11 2019

@author: hugo
"""

import params as prm
import pipeline_photodiode as photodiode
import pipeline_mc as mc
import pipeline_fit as fit
import pipeline_plotter as plotter
import pipeline_utils as putils


print('#################################')
print('#  Starting analysis pipeline   #')
print('#################################\n')       

# Creating /result folders and subfolders
putils.export_to_results()

print('###############')
print('#  Analyzing  #')
print('###############\n')

# Extract the photodiode data
if prm.do_photodiode :
    photodiode.export_sequences_times()

# Do the orientation selectivity analysis
if prm.do_mc :
    mc.mc_analysis()

# And fit the curves from the previous step
if prm.do_fit :
    fit.fit_curves()

print('################')
print('#  Regrouping  #')
print('################\n\n')
putils.do_cleanup()

print('##############')
print('#  Plotting  #')
print('##############\n\n')

if prm.do_idcard :
    plotter.create_ID_card()
    

