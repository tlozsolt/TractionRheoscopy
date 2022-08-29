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

    def __init__(self, globalParamFile, stepParamFile, test: bool=False):
        # load yaml metaData for both experiment and step
        super().__init__(globalParamFile=globalParamFile, stepParamFile=stepParamFile)
        if test: self.frames=3

        # add additional attributes that I will need here
        self.dpl_metaDataPath = self.stepParam['paths']['dplMetaData']
        self.dpl_log = self.stepParam['paths']['dpl_log']
        self.max_disp = self.stepParam['stitchTrack']['linking']['max_disp']
        self.locationCSV_frmt = self.stepParam['paths']['locationCSV']

        self.stitchCentroid = self.stepParam['stitchTrack']['stitchCentroid']

        # add attributes inherited from dpl class without inheriting the methods etc
        # this was added to Analysis paraent class
        #self.dpl = dpl.dplHash(self.dpl_metaDataPath)
        #self.hash_df = self.dpl.hash_df

    def sed(self, frame): return super().sed(frame)
    #def gel(self, frame): return super().gel(frame)
    def gel(self, frame, step=None, gelGlobal: bool = True):
        return super().gel(frame=frame, step=step, gelGlobal=gelGlobal)

    def gelGlobal2Local(self, gelGlobalGen): return super().gelGlobal2Local(gelGlobalGen)

    def setPlotStyle(self): pass

    def log(self): super().log()

    def posDict(self,
                posKey_frmt: str = '{coord} ({units}, {coordSys})',
                coordTuple: tuple = ('z', 'y', 'x'),
                units: str = 'um',
                coordSys: str = 'rheo_sedHeight'):
        return super().posDict(posKey_frmt=posKey_frmt,
                        coordTuple=coordTuple,
                        units=units,
                        coordSys=coordSys)

    def posList(self,
                posKey_frmt: str = '{coord} ({units}, {coordSys})',
                coordTuple: tuple = ('z', 'y', 'x'),
                units: str = 'um',
                coordSys: str = 'rheo_sedHeight'):
        return super().posList(posKey_frmt=posKey_frmt,
                        coordTuple=coordTuple,
                        units=units,
                        coordSys=coordSys)

    def qc(self):
        """
        quality control for decon particle locating including computation of outliers
        using Tukey windows, and the nubmer of particleCounts per hashValue. Useful for diagnosing
        problems during particle locating at the coarsest level...if particle counts dropped dramitically
        there was likely an error with thresholding the images before particle locating

        open quality control with:
        >> with open('./dpl_quality_control.pkl','rb') as f: qc = pkl.load(f)
        """
        #dplMetaPath = self.paths['stem'] + self.paths['dplMetaData']
        dplMetaPath = self.dpl_metaDataPath
        #log = self.paths['stem'] + '/log'
        log = self.paths['dpl_log']

        outliers, binHash, particleCount = pl.test(log, dplMetaPath)
        qcDict = {'outliers': outliers, 'binHash': binHash, 'particleCount': particleCount}
        with open(self.paths['stem'] + '/dpl_quality_control.pkl', 'wb') as f:
            pkl.dump(qcDict, f)
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

        if self.stitchCentroid:
            print('Stitching centroid Locations as well')
            inst = ls.ParticleStitch(metaPath,lsqCentroid='centroid')
            inst.parStitchAll(0,self.frames -1, n_jobs=12)



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
                #print(frame)
                # get a dictionary of dtypes for keys common to data and self.dtypes
                dtype_dict = da.insersectDict(self.dtypes, dict(data.dtypes))
                data = data.astype(dtype_dict)
                if mat == 'gel':
                    data = data[(data['gel_Tracer'] + data['fluorescent_chunk']) > data['gel_Background']]
                s.put(data)

        if verbose: print('Starting tracking')
        param = self.stepParam['stitchTrack']['linking']['param']
        param['search_range'] = self.stepParam['stitchTrack']['linking']['max_disp'][mat]
        with tp.PandasHDFStoreBig(stitched) as s:
            for linked in tp.link_df_iter(s, **param) : s.put(linked)
        return True

    def trackGelGlobal(self, verbose=False):
        """
        Stitch and track gel tracers across experiments, including reference stacks

        Refactored from da.static.stitchGelGlobal
        -Zsolt Nov 23, 2021

        ToDo: Modify to have the possibility of trackGelGlobal through mulitple reference stacks: ie
                 ref, a, ref2, b, ref3, ref4, c, ... g ref9,ref10,...ref 23.
                 Its possible this is alreafy compatible with that, but its worth checking
              -Zsolt Nov 27 2021
        """
        from particleLocating import dplHash_v2 as dpl

        #global_stitch = self.gelGlobal['path']
        global_stitch = os.path.join(self.exptPath, self.gelGlobal['path'])
        max_disp = self.gelGlobal['max_disp']

        # linked list of step kwrds (ie ref, a,b,c..)  and relative paths [./path/to/a/, path/to/b/, ... ]
        stepList = [list(step.keys())[0] for step in self.exptStepList]
        pathList = [list(step.values())[0] for step in self.exptStepList]

        mIdx_tuples = []
        tMax_list = []
        # create the single large stitched h5 file to mimic locating all the gel regions all at once, across experiments
        # one file to rule them all, also not the force overwrite by calling with 'w' flag
        with tp.PandasHDFStoreBig(global_stitch, 'w') as s:
            for n in range(len(stepList)):
                step = stepList[n]
                path = pathList[n]
                if verbose: print('Starting step {}'.format(step))

                # open correspodning yaml file to find out time steps. Note this cant be done in analysis abc
                # as it is not just for this step, but rather all the steps
                #_ = dpl.dplHash(self.dpl_metaDataPath)
                #_ = dpl.dplHash(self.gelGlobal['metaPath_frmt'].format(step=self.exptStepList[n][step]))
                _ = dpl.dplHash(pathList[n] + self.gelGlobal['metaPath_frmt'].format(step=stepList[n]))
                #_ = dpl.dplHash(path+'tfrGel10212018A_shearRun10292018{}_metaData.yaml'.format(step))
                metaData, hash_df = _.metaData, _.hash_df  # not sure if I need all the metaData or just the hash_df
                del _

                # open corresponding stitched file
                prefix = self.gelGlobal['fName_frmtStep_prefix']
                suffix = self.gelGlobal['fName_frmtTime_suffix']
                stitchedPathDict = dict(path=path+'/locations/', fName_frmt=prefix.format(step) + suffix)

                tMax = hash_df['t'].max() + 1
                offset = sum(tMax_list)  # note the edge case sum([]) = 0 works by default
                # TODO: add custom columns to loadStitched call trough dictionary expansion

                refShift = self.gelGlobal['ref2ImageZ']['ref'] # 143
                imageShift = self.gelGlobal['ref2ImageZ']['image'] # 29
                for frame, data in enumerate(da.loadStitched(range(tMax),
                                                             posKeys=self.posKeys_dict['gel'],
                                                             colKeys=['frame'], **stitchedPathDict)):
                    #for frame, data in enumerate(loadStitched(range(tMax), **stitchedPathDict)):
                    dtype_dict = da.insersectDict(self.dtypes, dict(data.dtypes))
                    data = data.astype(dtype_dict)
                    data['frame_local'] = data['frame']

                    # increment to prevent tp from overwriting the same frame number at different steps
                    data['frame'] = data['frame'] + offset
                    data['step'] = step
                    if step == 'ref':
                        data['z (um, refStack)'] = data['z (um, imageStack)']
                        data['z (um, imageStack)'] = data['z (um, refStack)'] - (refShift - imageShift)
                    else:
                        data['z (um, refStack)'] = data['z (um, imageStack)'] + (refShift - imageShift)
                    data['step'] = step
                    s.put(data)
                    mIdx_tuples.append((step, frame))

                # increment now, after looping. Also, I explicitly checked off-by-one errors here.
                tMax_list.append(tMax)

        stitch_param = self.gelGlobal['stitch_param']
        with tp.PandasHDFStoreBig(global_stitch) as s:  # now track
            for linked in tp.link_df_iter(s, max_disp, **stitch_param): s.put(linked)

        # assign mIdx to attribute
        tmp = pd.DataFrame(pd.MultiIndex.from_tuples(mIdx_tuples), columns=['mIdx'])
        tmp['step'] = tmp['mIdx'].apply(lambda x: x[0])
        tmp['frame local'] = tmp['mIdx'].apply(lambda x: x[1])

        self.gelGlobal_mIdx = tmp
        # save to dataFrame
        self.gelGlobal_mIdx.to_hdf(os.path.join(self.exptPath,self.gelGlobal['mIdx']), 'mIdx')
        return mIdx_tuples, global_stitch

    #def _xyzSed(self, frame: int, coordList: list, path: str, fName: str):
    #    """
    #    Export xyz file to path
    #    """
    #    posDF = self.sed(frame)[coordList]
    #    da.df2xyz(posDF,fPath = path,fName=fName)
    #    return True

    def xyzFirstLast(self):
        """
        Write xyz files for complete trajecotires on first and last frames to ovito subfolder
        """
        first = dict(n=0, posDF= self.sed(0))
        last = dict(n=self.frames -1, posDF= self.sed(self.frames -1 ))

        idx = first['posDF'].index.intersection(last['posDF'].index)

        coordList = ['{} (um, imageStack)'.format(coord) for coord in self.xyz]
        ovitoPath = './ovito'

        if not os.path.exists(ovitoPath): os.mkdir(ovitoPath)

        for frameDict in [first,last]:
            da.df2xyz(frameDict['posDF'].loc[idx][coordList],
                      fPath=ovitoPath, fName='/sed_t{:03}_complete.xyz'.format(frameDict['n']))
            da.df2xyz(frameDict['posDF'][coordList],
                      fPath=ovitoPath, fName='/sed_t{:03}_all.xyz'.format(frameDict['n']))

        return True

    def writexyz(self, frames: list= None):
        """
        Write xyz files for complete trajecotires on first and last frames to ovito subfolder
        """
        first = dict(n=0, posDF=self.sed(0))
        last = dict(n=self.frames - 1, posDF=self.sed(self.frames - 1))

        idx = first['posDF'].index.intersection(last['posDF'].index)

        coordList = ['{} (um, imageStack)'.format(coord) for coord in self.xyz]
        ovitoPath = './ovito'

        if not os.path.exists(ovitoPath): os.mkdir(ovitoPath)

        if frames is None: frames = range(self.frames)

        for t in frames:
            print('Exporting frame {} to xyz'.format(t))
            frameDict = dict(n=t, posDF = self.sed(t), gelDF = self.gel(t,gelGlobal=False))

            # export sed
            da.df2xyz(frameDict['posDF'].loc[idx][coordList], fPath=ovitoPath, fName='/sed_t{:03}_complete.xyz'.format(frameDict['n']))
            da.df2xyz(frameDict['posDF'][coordList], fPath=ovitoPath, fName='/sed_t{:03}_all.xyz'.format(frameDict['n']))

            # export gel
            da.df2xyz(frameDict['gelDF'][coordList], fPath=ovitoPath, fName='/gel_t{:03}_all.xyz'.format(frameDict['n']))

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

        try:
            if pipeline['gelGlobal']:  self.trackGelGlobal()
        except KeyError: pass

        # output the first and last frames
        self.xyzFirstLast()

        return True

if __name__ == '__main__':

    #refList =["a_preShearFullStack", "b_preShearFullStack",
    #             "c_preShearFullStack", "d_postFullShearStack",
    #             "d_preShearFullStack", "e_postShearFullStack",
    #             "e_preShearFullStack", "f_postShearFullStack",
    #             "f_preShearFullStack", "g_preShearFullStack"]

    globalParamFile = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/tfrGel23042022_strainRamp_globalParam.yml'
    with open(globalParamFile, 'r') as f: globalParam = yaml.load(f, Loader=yaml.SafeLoader)

    stepList = globalParam['experiment']['exptSteps']
    exptDir = globalParam['experiment']['path']
    for step in stepList:
        #if step == 'a_imageStack': pass
        #else: continue
        os.chdir(exptDir)
        os.chdir(step)
        print(os.getcwd())
        print('Start step {}'.format(step))
        #testPath = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/{}_imageStack'.format(step)
        #testPath = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/{}'.format(step)
        param = dict(globalParamFile = '../tfrGel23042022_strainRamp_globalParam.yml',
                 stepParamFile = '{}/step_param.yml'.format(step), test=False)
        test = StitchTrack(**param)
        #test()
        p = test.stepParam['stitchTrack']
        test.track('gel', **p['track'])