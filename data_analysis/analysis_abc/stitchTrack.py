import sys

import pandas as pd

sys.path.append('/')
from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis import static as da
import yaml
import os
import trackpy as tp
from particleLocating import postLocating as pl
from particleLocating import locationStitch as ls
from particleLocating import dplHash_v2 as dpl
import pickle as pkl

# read step param file
# run stitching, flagging particle to remove
#   -> this produces a single h5 file for each frame
#   -> No data from csv files are deleted, but columns for stitch keepBool are added

# tpCombine: combine framewise h5 files to a single h5 file mimicing the output of trackpy batch processing
#   -> this produces a single h5 file for all frames (no deletions)
#   -> create a deep copy with generic file name (sed_stitched.h5) for example and throw away all stitch doubles

# tracking sed_stitched.h5 using parameters from step param file

# all downstream analysis starts from here.

# I think this can be implemented as a child of datanalysis abstract base class.

class StitchTrack(Analysis):

    def __init__(self, globalParamFile, stepParamFile):
        # load yaml metaData for both experiment and step
        super().__init__(globalParamFile=globalParamFile, stepParamFile=stepParamFile)

        # add additional attributes that I will need here
        self.dpl_metaData = self.stepParam['paths']['dplMetaData']
        self.max_disp = self.stepParam['stitchTrack']['linking']['max_disp']
        self.locationCSV_frmt = self.stepParam['paths']['locationCSV']

        # add attributes inherited from dpl class without inheriting the methods etc
        self.dpl = dpl.dplHash(self.dpl_metaData)
        self.hash_df = self.dpl.hash_df

    def sed(self, frame): super().sed(frame)
    def gel(self, frame): super().gel(frame)
    def setPlotStyle(self): pass
    def log(self): super().log()

    def qc(self):
        """
        quality control for decon particle locating including computation of outliers
        using Tukey windows, and the nubmer of particleCounts per hashValue. Useful for diagnosing
        problems during particle locating at the coarsest level...if particle counts dropped dramitically
        there was likely an error with thresholding the images before particle locating
        """
        dplMetaPath = self.paths['stem'] + self.paths['dplMetaData']
        log = self.paths['stem'] + '/log'

        outliers, binHash, particleCount = pl.test(log, dplMetaPath)
        qcDict = {'outliers': outliers, 'binHash': binHash, 'particleCount': particleCount}
        with open(self.paths['stem'] + '/dpl_quality_control.pkl', 'wb') as f:
            pkl.dumps(qcDict, f)
        #tmp = pd.DataFrame(qcDict)
        #tmp.to_hdf(self.paths['stem'] + '/dpl_quality_control.h5', 'qcDict')


        # plot particle counts for fixed spatial location?
        # not sure what exactly particleStitch.plotParticle does...write to a file?
        # maybe this should be broken down into a different fuctnion.
        #for mat in ['gel','sed']:
        #    hv_mod_t = {}
        #    hv_mod_t[mat] = ((self.hash_df['t'] == 0) & (self.hash_df['material'] == mat)).value_counts().loc[True]
        #    #hv_mod_t['gel'] = ((self.hash_df['t'] == 0) & (self.hash_df['material'] == 'gel')).value_counts().loc[True]
        #    #hv_mod_t['sed'] = ((self.hash_df['t'] == 0) & (self.hash_df['material'] == 'sed')).value_counts().loc[True]
        #    for hv in range(hv_mod_t[mat]): pl.plotParticle(qcDict, hv)
        return True

    def stitch(self):
        metaPath = self.paths['stem'] + self.paths['dplMetaData']
        # this is parallel over frames, with no parallelization within a frame.
        inst = ls.ParticleStitch(metaPath)
        # number of jobs could probably be a 18 or so on IMAC
        inst.parStitchAll(0, self.frames -1 , n_jobs=12)

    def track(self, mat: str,
              forceRecompute: bool = True,
              verbose: bool = True):

        stitched_frmt = self.paths['stem'] +'/{}_stitched.h5'

        stitched = stitched_frmt.format(mat)

        # delete previous if force recompute is true
        if os.path.exists(stitched) and forceRecompute:
            if verbose: print('removing previous stitched h5 file')
            os.remove(stitched)

        # move onto making one large h5 file to mimic the output of trackpy batch.
        if verbose:
            print("Preprocessing tracking: making large h5 database of all stitched times")
            print(stitched)
        with tp.PandasHDFStoreBig(stitched) as s:
            # for one time point, this should be identical to stitched.h5
            # this legacy from decon particle locating...so in principle I could read from
            # dpl metaData file?
            stitched_fName_frmt = self.nameLong + '_stitched_{}'.format(mat) + '_t{:03}.h5'
            for frame, data in enumerate(da.loadStitched(range(self.frames),
                                        path=self.paths['stem'] + '/locations/',
                                        fName_frmt=stitched_fName_frmt,
                                        posKeys=self.posKeys_dict[mat],
                                        colKeys=['frame'])):
                print(frame)
                s.put(data)

        if verbose: print('Starting tracking')
        param = self.stepParam['stitchTrack']['linking']['param']
        param['search_range'] = self.stepParam['stitchTrack']['linking']['max_disp'][mat]
        with tp.PandasHDFStoreBig(stitched) as s:
            for linked in tp.link_df_iter(s, **param) : s.put(linked)
        return True


    def __call__(self, verbose: bool = True):

        pipeline = self.stepParam['stitchTrack']['pipeline']
        p = self.stepParam['stitchTrack']

        # 0) quality control
        if verbose: print('Carrying out quality control')
        if pipeline['qc']: self.qc(**p['qc'])

        # 1) stitch the dpl hashes
        # writes an h5 file for each time step and sed and gel separately (but all in one call)
        # path is in dplmetaData yaml file in /PROJECT/locations (I think)
        if verbose: print('starting stitching')
        if pipeline['stitch']: self.stitch(**p['stitch'])

        # 2,3)
        # create one large h5 file and track with all the time points.
        if verbose: print('starting tracking')
        if pipeline['track']:
            for mat in ['gel','sed']:
                if verbose: print('starting on material {}'.format(mat))
                self.track(mat, **p['track'])

        return True