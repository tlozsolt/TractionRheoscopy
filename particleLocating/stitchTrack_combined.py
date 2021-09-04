from particleLocating import postLocating as pl
from particleLocating import locationStitch as ls
from data_analysis import static as da
from particleLocating import dplHash_v2 as dpl
import trackpy as tp
import pandas as pd
import os
from data_analysis.static import loadStitched, stitched_h5

"""
This class combines all steps to go from directory of location and log files to a single
stitched and tracked h5 file. 
All parameters are read from stitchTrack keyword in metaDataYaml file specific
for the experiment, and typically stored in the PROJECT or LOCAL directory (not git directory)

The scripting on multiple runs is done by either calling a bash script with single python executable
and passing a list of directories corresponding to location and log files for distinct shear runs.
All parameters (number of time steps, parameters for tracking, parallel resources, etc)
need to read from the metaDataYaml file

Zsolt
Aug 10, 2021
"""

# sketch of the code for a single instance with hard coded values


#these either need to be read form metaData file
# or passed to the function

# what files to loop over and the name of the directory structure

#pathStem_frmt = '/Volumes/PROJECT/tfrGel10212018x/{}'
pathStem_frmt = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/{}'
#steps_fName = ['tfrGel10212018A_shearRun10292018{}'.format(x) for x in ['a','b','c','d','e','f']]
steps_fName = ['tfrGel10212018A_shearRun10292018{}'.format(x) for x in ['e','f']]

# what is the max displacement in microns
max_disp_dict = {'gel': 1.5, 'sed': 1.3}

pipeline = {'qc':True, 'stitch':False, 'track': True, 'locHDF':True, 'xyz': False}

for step in steps_fName:

    # set up some files and paths
    # CAUTION, the filepath in local metaData yaml have to be updated.
    path = pathStem_frmt.format(step) # this is a local path, however locations and log are both sym links to kaposzta
    log, meta = path + '/log', path +'/{}_metaData.yaml'.format(step)
    hash_df = dpl.dplHash(meta).hash_df

    # how many hashValues for a single time point
    hv_mod_t = {}
    hv_mod_t['gel'] = ((hash_df['t'] == 0) & (hash_df['material'] == 'gel')).value_counts().loc[True]
    hv_mod_t['sed'] = ((hash_df['t'] == 0) & (hash_df['material'] == 'sed')).value_counts().loc[True]

    # how many time points to stitch?
    tMax = hash_df['t'].max()

    # 0) quality control, not sure if I am going to save this or not
    if pipeline['qc']:
        outliers, binHash, particleCount = pl.test(log, meta)
        qcDict = {'outliers': outliers, 'binHash': binHash, 'particleCount': particleCount}
    #mat = 'sed'
    #for hv in range(hv_mod_t[mat]):
    #    pl.plotParticle(qcDict, hv)

    # 1) stitch the location hashes
    # writes an h5 file for each time step and sed and gel separately (but all in one call)
    # path is in metaData yaml file in /PROJECT/locations (I think)
    if pipeline['stitch']:
        inst = ls.ParticleStitch(meta)
        inst.parStitchAll(0,tMax, n_jobs=12)

    for mat in ['sed', 'gel']:
        max_disp = max_disp_dict[mat]
        stitched = path+'/{}_stitched_t0_t{:03}_maxDisp_{}.h5'.format(mat, tMax, max_disp)

        # 2) track from the output of each stitched time steps, simulate the output you would get
        # from funning trackpy on the all the time steps (ie one big h5 file)
        if pipeline['track']:
            if os.path.exists(stitched):
                print('removing previous stitched h5 file')
                os.remove(stitched)

            with tp.PandasHDFStoreBig(stitched) as s:
                print("Preprocessing tracking: making large h5 database of all stitched times")
                print(stitched)
                stitched_fName_frmt = step + '_stitched_{}'.format(mat)+'_t{:03}.h5'
                for data in loadStitched(range(tMax+1),mat,path=path+'/locations', fName_frmt=stitched_fName_frmt): s.put(data)

            # track the particles keeping all the location fields, and maybe list these location field in the metaData file?
            # run tracking using trackpy link on stitched hdf5 file as input
            with tp.PandasHDFStoreBig(stitched) as s:
                print("Starting tracking on {}".format(step))
                for linked in tp.link_df_iter(s,max_disp, neighbor_strategy='KDTree',
                                              pos_columns=['{} (um, imageStack)'.format(x) for x in ['x', 'y', 'z']],
                                              t_column='frame',
                                              adaptive_stop=0.1,
                                              adaptive_step=0.95):
                    s.put(linked)

        # 4) create smaller h5 file with just the particle locations for loading into memory
        if pipeline['locHDF']:
            pass

        # 5) write xyz files to visualize the particle trajectories, and maybe include an xyz file
        if pipeline['xyz']:
            pass


### WARNING, not sure this works. There may be problems with the boolean selection for
# complete trajectories.

# write xyz files for particles and all particles with complete trajectories for both sed and gel
# dont worry aobut any of the other columns for the moment

#complete_id = {'sed': 959859, 'gel': 3551 }
#sed_flags = ['sed-all', 'sed-complete','sed-incomplete']
#gel_flags = ['gel-all', 'gel-complete', 'gel-incomplete']
#fName_frmt ='tfrGel10212018A_shearRun10292018f_{flag}_t{frame:03}.xyz'
#path = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/scratch'
#col_keys = ['{} (um, imageStack)'.format(x) for x in ['x','y','z']] + ['particle']
#
#for mat in ['sed','gel']:
#    cut = complete_id[mat]
#    with tp.PandasHDFStoreBig('/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/{mat}_stitched.h5'.format(mat=mat)) as s:
#        for frame in range(90):
#            loc = s.get(frame)
#            for flag in sed_flags+gel_flags:
#                fName = fName_frmt.format(flag=flag, frame=frame, mat=mat)
#                if flag == '{mat}-all'.format(mat=mat): da.df2xyz(loc,path,fName)
#                elif flag == '{mat}-complete'.format(mat=mat): da.df2xyz(loc[loc['particle'] < cut],path,fName)
#                elif flag == '{mat}-incomplete'.format(mat=mat): da.df2xyz(loc[loc['particle'] >= cut],path, fName)
#            print( "xyz file for {frame} written to {path}".format(frame=frame, path=path + '/{}'.format(fName_frmt)))