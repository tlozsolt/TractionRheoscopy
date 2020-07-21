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
inst.dpl.metaData['postDecon']['threshold']

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
                                 diameter=(17,23,23),
                                 fit_function='disc',
                                 param_val={'disc_size':0.6},
                                 compute_error=True)
#%%
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
# pickle the current variables
import pickle
with open(tmpPath+'/postDeconCombined.pkl','wb') as f:
    pickle.dump([np_postThresholdFilter,df_locations, df_refineHat],f)

#%%
# unpickle to restart
import pickle
with open(tmpPath+'/postDeconCombined.pkl','rb') as f:
    np_postThresholdFilter, df_locations, df_refineHat = pickle.load(f)

#%% refine with dask dataframe
# set up a node with 4 cores and 16Gb of ram with dask client or whatever
# DOES NOT WORK
#from dask.distributed import Client, LocalCluster
#from dask import dataframe as ddf
#node = LocalCluster(n_workers=4,threads_per_worker=24)
#client = Client(node)
"""
How to start a dask distributed client on IMAC

At the command line, start the scheduler:
   > dask-scheduler

This will return some information, with an ip address of the scheduler.
On IMAC this looks like 
   distributed.scheduler - INFO -   Scheduler at:      tcp://10.0.0.33:8786

Now you can open, in a web browswer to track the processes by going to:
   http://localhost:8787/status

Add workers at the command line:
   > dask-worker tcp://10.0.0.33:8786 --nprocs 16 --nthreads 24

When this is set up the workers should show up on the web browswer along with this type of output at the terminal
   >> distributed.core - INFO - Starting established connection
      distributed.worker - INFO -         Registered to:       tcp://10.0.0.33:8786

Now attach the client to python script using:
   from dask.disctributed import Client
   client = Client('10.0.0.33:8786')

Now **any** compute() that you call will be sent to the pool of workers..threaded and multicore.
Woah...

But...there is still somethings missing 

ToDo:
  [ ] Find out how to upload particleLocating module to all workers.
  [ ] Set memory limits on each worker, or alternately a global mem limit of 16Gb 
"""

#%%
# convert the dataFrame to daskDataFrame and partition rows into sizes that will max out at 4Gb of ram
# the size of the partition must be controlled, not the number of partitions but this should be easy to do
#   -> Yes, you can either specific npartitions (int) or chunksize (number of rows)
from dask import dataframe as ddf
ddf_loc = ddf.from_pandas(df_locations,npartitions=16)

# wrap tp.refine_leastsq() to:
#    -> read parameters as I currently do including a full copy of image array
#    -> write to csv without any other saving or merging, maybe indexed by partition
#    -> print a statement to let me know when a partition is complete

def ddf_refine(ddf_chunk, np_imageArray, **tpRefineKwargs):
    df_chunk = tp.refine_leastsq(ddf_chunk,np_imageArray,**tpRefineKwargs)
    print('Completed a chunk!')
    print(df_chunk.head(10))
    with open(tmpPath+'/tmp.csv','a') as f:
        df_chunk.to_csv(f, header=f.tell()==0, sep=' ')
    return df_chunk

#%%
from functools import partial
import numpy as np
dict_refine = {'diameter':(17,23,23),
               'fit_function':'disc',
               'param_val':{'disc_size':0.6},
               'compute_error':True}

refine_dtypes = df_refineHat.dtypes.to_dict()

refineDisc_ddf2 = ddf_loc.map_partitions(partial(ddf_refine,np_imageArray =np.copy(np_postThresholdFilter),**dict_refine)
                       ,meta=refine_dtypes).compute()

#
# run gaussian refinement across the partitions using map_partitions and sending and deep copy of the image array

# to each node

#%%
# visualize
#pyFiji.send2Fiji([da_decon.compute(),da_smartCrop.compute(),da_threshold.compute,da_postThresholdFilter.compute()],\
#                 wdir=tmpPath)
pyFiji.send2Fiji(da_threshold.compute(),wdir=tmpPath)
pyFiji.send2Fiji(np_postThresholdFilter,wdir=tmpPath)
pyFiji.send2Fiji(np_postThresholdFilter.astype('uint16'),wdir=tmpPath)
pyFiji.send2Fiji(np_postDeconResize,wdir=tmpPath)
pyFiji.send2Fiji(da_smartCrop.compute(),wdir=tmpPath)
