#!/usr/bin/env python
# coding: utf-8

# Absolute magnitude is defined to be the apparent magnitude an object would have if it were located at a distance of 10 parsecs.
# In astronomy, absolute magnitude (M) is a measure of the luminosity of a celestial object on an inverse logarithmic astronomical magnitude scale.

# In[1]:


import time
import numpy as np
import h5py
import hdf5plugin
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from unyt import yr, Myr, kpc, arcsec, nJy, Mpc, Msun, erg, s, Hz
from astropy.cosmology import Planck18 as cosmo
from scipy import signal
import os 
import csv
import resource
import pickle
import shutil
import json
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.particle.stars import sample_sfhz
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import Filter, FilterCollection
from synthesizer.sed import combine_list_of_seds

from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.kernel_functions import Kernel

from synthesizer.conversions import lnu_to_absolute_mag

current_directory = os.getcwd()
print("Current directory:", current_directory)


# # Start the timer
# start_time = time.time()
# 

# In[2]:


# Alternative method for LF:
# try this method again, but using AB mag instead of mass, and suply your own bins (up to -17, say)
def calc_lf(ab_mag, volume, massBinLimits):
# OG:        hist, dummy = np.histogram(np.log10(mstar), bins = massBinLimits)
        hist, dummy = np.histogram(ab_mag, bins = massBinLimits)
        hist = np.float64(hist)
        phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])
        phi_sigma = (np.sqrt(hist) / volume) /\
                    (massBinLimits[1] - massBinLimits[0]) # Poisson errors
        return phi, phi_sigma, hist
    


# 
# phi_arr =[] #phi
# phi_sigma_arr =[] # phi_sigma
# hist_arr = [] # hist
# z_arr = [] #redshift_074, 
# abs_mag_arr = [] #absolute mag (th filter)
# Vcom_arr = [] # comoving vol
# massBinLimits_arr = []
# 

# len(phi_arr)

# In[3]:


grid_name = "bc03-2016-Miles_chabrier-0.1,100.hdf5"
grid_dir = "/home/jovyan/"

# Create a new grid
#grid = Grid(grid_name, grid_dir=grid_dir, read_lines=False)

# instead of using a filter, which requires us to load in large SEDs first, pass the grid wavelength
#filt1 = Filter("top_hat/filter.1", lam_min=1400, lam_max=1600, new_lam=grid.lam)
#filt_lst = [filt1]

# Define a new set of wavelengths
lims_lams=(1400, 1600)

grid = Grid(grid_name, grid_dir=grid_dir, lam_lims=lims_lams, read_lines=False)
print(grid.shape)


# In[4]:


# Define the directory where the text files will be saved
output_dir = "/home/jovyan/camels/play/synth-play/LH/output/"


# In[5]:


# get gals
print('loading in galaxies')
LH_X = 'LH_50'
dir_ = '/home/jovyan/Data/Sims/IllustrisTNG/LH/' + LH_X
gals_074 = load_CAMELS_IllustrisTNG(
    dir_,
    snap_name='snapshot_074.hdf5', 
    fof_name='groups_074.hdf5',
    verbose=False,
)


# 
# for gal in gals_074:
#     print(f"Galaxy has {gal.stars.nstars} stellar particles")
#     print(f"Galaxy gas {gal.gas.nparticles} gas particles")
# 

# gals_074

# In[6]:


# Filter galaxies to only include those with 100 or more star particles
print('filtering for 100 stars/gas particles')
gals_074 = [gal for gal in gals_074 if gal.stars.nstars >= 100]
print(len(gals_074))
gals_074 = [gal for gal in gals_074 if gal.gas.nparticles >= 100]
#print(len(gals_074))


# 
# for gal in gals_074:
#     print(f"Galaxy has {gal.stars.nstars} stellar particles")
#     print(f"Galaxy gas {gal.gas.nparticles} gas particles")
# 

# In[7]:


cat_074 = dir_+'/groups_074.hdf5'
# open file
f_h5py = h5py.File(cat_074, 'r')

# read different attributes of the header
boxSize_074 = f_h5py['Header'].attrs[u'BoxSize']/1e3 #Mpc/h
redshift_074 = f_h5py['Header'].attrs[u'Redshift']


# # Get the number of star particles for each galaxy
# nstars_list = [gal.stars.nstars for gal in gals_074]
# # Get the number of gas particles for each galaxy
# ngas_list = [gal.gas.nparticles for gal in gals_074]
# 
# # Calculate the minimum and maximum number of star particles
# min_nstars = min(nstars_list)
# max_nstars = max(nstars_list)
# 
# # Calculate the minimum and maximum number of gas particles
# min_ngas = min(ngas_list)
# max_ngas = max(ngas_list)
# 
# print(f"Minimum number of star particles: {min_nstars}")
# print(f"Maximum number of star particles: {max_nstars}")
# print(f"Minimum number of gas particles: {min_ngas}")
# print(f"Maximum number of gas particles: {max_ngas}")
# 

# In[ ]:
# other method for getting spectra (still fails)
# try different method to get specs:
#specs = np.vstack([g.stars.get_spectra_incident(grid).lnu for g in gals_074])
#print(specs)


print('get incident spectra')
# FAILS HERE
spec_list = []
# Lets work with z=0 so gals_025
for index, gal in enumerate(gals_074):
    print("Gal: [", index , ']')
    # get_spectra_incident An Sed object containing the stellar spectra
    spec = gal.stars.get_spectra_incident(grid)
    print(spec)
    print('got spectra, getting fnu')
    spec.get_fnu0()
    print('appending spec to list')
    spec_list.append(spec)


# In[ ]:
# add filter anyway
filt1 = Filter("top_hat/filter.1", lam_min=1400, lam_max=1600, new_lam=grid.lam)
filt_lst = [filt1]

# combine
print('combining seds to list')
seds = combine_list_of_seds(spec_list)
seds.lnu # rest frame lumd

# filter to top hat
seds.get_photo_luminosities(filt_lst)
seds.photo_luminosities.photo_luminosities
abs_mag = lnu_to_absolute_mag(seds.photo_luminosities.photo_luminosities)
abs_mag_th = abs_mag[0]


# In[ ]:


# next steps, get luminosity function for these magnitudes
# co-moving volume: BoxSize_025 and redshift:
little_h =  0.6711
Vphys = (boxSize_074/little_h )**3
Vcom = Vphys * ((1+redshift_074)**3)


# In[ ]:


massBinLimits = np.arange(-22, -16, 0.5)
phi, phi_sigma, hist = calc_lf(abs_mag_th, Vcom, massBinLimits)
# NOTE: 074 is the same redshift as CV_0/025
massBinLimits = massBinLimits[:-1]


# In[ ]:


output_dir = "/home/jovyan/camels/play/synth-play/LH/output/"

# Define output file path
output_file = f"{output_dir}{LH_X}.txt"

# Write the data to the text file line by line
with open(output_file, 'w') as txtfile:
    # Write phi values
    txtfile.write("phi\n")
    for value in phi:
        txtfile.write(f"{value}\n")

    # Write phi_sigma values
    txtfile.write("phi_sigma\n")
    for value in phi_sigma:
        txtfile.write(f"{value}\n")

    # Write hist values
    txtfile.write("hist\n")
    for value in hist:
        txtfile.write(f"{value}\n")

    # Write massBinLimits values
    txtfile.write("massBinLimits\n")
    for value in massBinLimits:
        txtfile.write(f"{value}\n")


print('Written out: /home/jovyan/camels/play/synth-play/LH/output/',LH_X)


# # Stop the timer
# end_time = time.time()
# 
# # Calculate the elapsed time
# elapsed_time = end_time - start_time
# 
# print("Elapsed time:", elapsed_time, "seconds")

# In[ ]:


# Get memory usage in bytes
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("Memory usage end:", memory_usage/1000000, "GB")


# # will do latest LH_ set in loop as a test
# label_z = 'z = 0.46'
# 
# 
# # Plot the luminosity function
# plt.errorbar(massBinLimits, phi, yerr=phi_sigma, fmt='o', color='blue',label=label_z)
# plt.xlabel('Absolute Magnitude (AB)')
# plt.ylabel('Number Density (Mpc^-3 mag^-1)')
# plt.yscale('log')
# 
# plt.title('Luminosity Function XMM-OM filter')
# plt.grid(True)
# plt.show()

# In[ ]:




