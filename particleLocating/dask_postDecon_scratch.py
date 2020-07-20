from particleLocating.postDeconCombined import PostDecon_dask as postDecon
from particleLocating import pyFiji
import trackpy as tp
from particleLocating import locating
from functools import partial
import yaml

#%%
# load some standard data
yamlPath = '/Users/zsolt/Colloid_git/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'
tmpPath = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/pyFiji'
computer = 'IMAC'
hv = 0

#%%
# init instance
inst = postDecon(yamlPath,hv,computer=computer)

#%%
# smart crop
da_decon = inst.init_da
da_smartCrop, log_smartCrop = inst.smartCrop_da(da_decon)

#%%
# threshold
da_threshold = inst.threshold_da(da_smartCrop, **inst.dpl.metaData['postDecon']['threshold'])

#%%
# post threshold filter
metaData =inst.dpl.metaData['postDecon']
da_postThresholdFilter = inst.postThresholdFilter_da(da_threshold,**metaData)
np_postThresholdFilter = da_postThresholdFilter.compute()

#%%
# locate
df_locations = tp.locate(np_postThresholdFilter,
          diameter=(19,27,27),
          minmass=6000,
          separation=(17,25,25),
          percentile=0.90,
          preprocess=False,
          engine='numba')

#%%
df_refineHat = tp.refine_leastsq(df_locations,
                                 np_postThresholdFilter,
                                 diameter=(21,29,29),
                                 fit_function='disc',
                                 param_val={'disc_size':0.6},
                                 compute_error=True)
df_refineGauss = tp.refine_leastsq(df_locations,
                                   np_postThresholdFilter,
                                   diameter=(19,27,27),
                                   fit_function='gauss',
                                   compute_error=False)
# Commenta
# diameter is the search range, not the feature size or sigma as these are variable during the fit


#%%
# function to join centroid and lsq dataframes without unnecessary duplication
# Removes all the columns are known duplicates and keep everything else with join and
# suffix on left and right.

# find all columns in df_locations that are unchanged by refine_lsq
col_duplicate = [key for key in df_locations if df_locations[key].equals(df_refineHat[key])]
# drop those columns from both hat and gauss
# and join with rsuffix and lsuffix
df_refinelsq = df_refineHat.drop(columns=col_duplicate).join(
    df_refineGauss.drop(columns=col_duplicate),
    lsuffix='_refineHat', rsuffix='_refineGuass')
# join again back to loc
df_locations = df_locations.join(df_refinelsq, lsuffix='_centroid')


#%%
# resize, locate iteratively
postDeconMeta = inst.dpl.metaData['postDecon']
da_postDeconResize = da_postThresholdFilter.map_blocks(
    partial(inst.resize_da,**postDeconMeta),
    dtype='float32')
dict_locParam = inst.dpl.metaData['locating']
np_postDeconResize = da_postDeconResize.compute()

#%%
# iterative locating with cmobined gauss and hat lsq refinement
df_loc = locating.iterate(np_postThresholdFilter,inst.dpl.metaData['locating'],inst.mat)

#%%
# visualize
#pyFiji.send2Fiji([da_decon.compute(),da_smartCrop.compute(),da_threshold.compute,da_postThresholdFilter.compute()],\
#                 wdir=tmpPath)
pyFiji.send2Fiji(da_threshold.compute(),wdir=tmpPath)
pyFiji.send2Fiji(np_postThresholdFilter,wdir=tmpPath)
pyFiji.send2Fiji(np_postThresholdFilter.astype('uint16'),wdir=tmpPath)
pyFiji.send2Fiji(np_postDeconResize,wdir=tmpPath)
