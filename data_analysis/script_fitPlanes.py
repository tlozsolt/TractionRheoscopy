"""
script to process sed and gel fit planes and generate id based particle linking
across experiments.

==== Input =====
+ ovito output of particle ids of particles at top of gel and bottom of sediment output as xyz
+ dataframe of particle position for sed and gel in common (um, imageStack) coordinate system

---- Data (pickled?, hdf?) ----
    + ovito based index of particles at interface, output to csv file
    + dataframe of fitted planes for sediment
    + dataframe of fitted planes for gel
    + particle positions, rotated to para/perp, and in right handed rheo coordinates, with boolean flag for clean
"""
#%%
import pandas as pd
import sys
sys.path.append('/Users/zsolt/Colloid_git/TractionRheoscopy')
from data_analysis import static as da
import yaml
import os
import numpy as np

#%%
path = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018e'
# what are the files in the directory?
# also, make ovito output directory
#os.mkdir('{PATH}/ovito'.format(PATH=path))
os.listdir(path)
#%%
# load the particles and save xyz file of t=0 and complete trajectories
tMax = 63
fName ={'sed': '/sed_stitched_t0_t{:03}_maxDisp_1.3.h5'.format(tMax),
        'gel': '/gel_stitched_t0_t{:03}_maxDisp_1.5.h5'.format(tMax)}
#%%
#load position
sed_pos = da.loadData2Mem(path + fName['sed'])
gel_pos = da.loadData2Mem(path + fName['gel'])
#%%
# get complete indices
sed_complete_idx = da.getCompleteIdx(sed_pos.xs(tMax,level='frame'), sed_pos.xs(0,level='frame'))
gel_complete_idx = da.getCompleteIdx(gel_pos.xs(tMax,level='frame'), gel_pos.xs(0,level='frame'))

# write xyz files of particles with complete traj, at the start of the deformation
#da.df2xyz(sed_pos.xs(0,level='frame').loc[sed_complete_idx], '{PATH}/ovito'.format(PATH=path),'/step_{}_sed_complete_t00.xyz'.format(path[-1]))
#da.df2xyz(gel_pos.xs(0,level='frame').loc[gel_complete_idx], '{PATH}/ovito'.format(PATH=path),'/step_{}_gel_complete_t00.xyz'.format(path[-1]))
#%%
# optional, output the full particle trajectory to xyz file.
for t in range(tMax):
    print(t)
    #da.df2xyz(sed_pos.xs(t,level='frame'),'{PATH}/ovito'.format(PATH=path),'/step_{}_sed_all_t{:02}.xyz'.format(path[-1], t))
    da.df2xyz(sed_pos.xs(t,level='frame').loc[sed_complete_idx],'{PATH}/ovito'.format(PATH=path),'/step_{}_sed_completeTraj_t{:02}.xyz'.format(path[-1], t))

#%%
# fit the plane from the ovito output
sed_bottom_idx = da.readOvitoIdx(path+'/ovito/sed_complete_bottom_surface.xyz')
sed_fits = da.fitSurface(sed_pos, sed_bottom_idx,coordStr='(um, imageStack)')

#%%
# clean it up

