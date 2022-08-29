from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis import static as da
import pandas as pd
import os
import numpy as np
import yaml
from datetime import datetime
import trackpy as tp
import os
from warnings import warn
import re
import statsmodels.api as sm

from particleLocating import dplHash_v2 as dpl

class GelGlobal(Analysis):
    """
    This class carries out  global gel tracking through an order list of steps in the
    experiment. The basic idea is to find the gel/sediment interface in each step. Fit it
    and then generate an ordered list of z-offsets across the experiment. After that,
    the z-offset is subtracted from all (um, imageStack) coordinates and the gel tracers
    are tracked across the deformation.
    It is unclear that the xy movement, especially between adjacent refernce stacks is
    large enough that xy rigid body displacement should also be subtracted prior to tracking.
    This would be best done by tracking particles that are well separated at the bottom
    of the stack.

    This class should be run after stitching but before dataCleaning.

    steps:
    -a_preShearFullStack:
      -path: '../a_preShearFullStack'
      -frames: 1
    -a_imageStack:
      -path: '../a_imageStack'
      -frames: 22
    ...
    -f_postShearFullStack:
      -path: '../postShearFullStack'
      -frames: 1
    -e_preShearFullStack:
      -path: '../preShearFullStack'
      -frames: 1
    -e_imageStack:
      -'../e_imageStack'
      -frames: 62

    pieces that are already written and may be reused or refactored:

    analysis_abc/gelGlobal2Local: index generator to acess global hdfStore
    and access as if it was local..ie access (step f, time 3) will give the right
    absolute index in the gelGlobal hdfStore.

    dataCleaning/fitReference: fit sed/Gel interface on a reference stack
    dataCleaning/fitSurface: fit sed/gel interfrace on imageStack time series for all times

    stitchTrack/trackGelGlobal: carry out the tracking a list of steps assuming a single
    fixed offset between reference and image. Writen for a step list of ref,a,b,c,..f
    that is just a single reference stack at the beginning and no intermediate or back
    to back references.

    globalParamYAML/gelGlobal: key in global param file specifying some tracking parameters
    and output locations. Additionally has gelGlobal/ref2ImageZ dict that has manually entered
    refernce imageStack conversion in z only.

    Steps in this class
    -read in list of steps and paths
    -for each step, fit the surface and save the output in an dataFrame for the whole expt
     with continous ordered index and columns for step, frame number, ref/image type, fit param
    - Track gel global for each step and frame by subtracting the fit param z to link up steps

    """

    def __init__(self,globalParamFile, stepParamFile, test=False):
        super().__init__(globalParamFile=globalParamFile,stepParamFile=stepParamFile)
        if test:
            if self.frames > 3: self.frames=3
            print('Creating GelGlobal class in test mode with {} frames'.format(self.frames))

        self.stepParamFile = './step_param.yml'
        #self.interfaceIdx = da.readOvitoIdx(self.paths['sed_interface_idx'])

        self.sedGelFit_df = pd.DataFrame([])

        # list of strain tensors between reference stacks. initial is identity
        def F_simpleShear(exz, eyz, coordSys:str = '(um, rheo_sedHeight)'):
            if coordSys == '(um, imageStack)':
                R = np.array([[1,0,0],[0,-1,0],[0,0,1]])
            else: R = np.array([[1,0,0],[0,1,0],[0,0,1]])
            return (R@np.array( [[1,0, exz],[0,1,eyz], [0,0,1] ])@R)

        # This was obtained on strainRamp steps a-f, including postShearFullStacks, by running
        # gelGlobal.trackGelGlobal(stepListFlag='ref') with linking up to 2um displacement
        # Then, the resulting particle displacements were computed and pass to
        # gelGlobal.simpleShearFit()
        self.refEpsilonValues = np.array([ [ 2.17680230e-04, +7.50444288e-04],
                                           [ 6.40066268e-04, +1.51512487e-04],
                                           [ 6.22562743e-04, -5.94828806e-05],
                                           [ 1.05778342e-03, +6.15225415e-04],
                                           [ 3.44382188e-04, -6.03385258e-04],
                                           [-1.44213845e-03, +3.04616293e-03],
                                           [ 3.85316240e-04, -6.60361489e-04],
                                           [ 3.50499052e-03, +5.08353804e-04]])
        self.refEpsilon =  [ F_simpleShear(0.000218, -0.000750),
                             F_simpleShear(0.000640, -0.000152),
                             F_simpleShear(0.000623, 0.000059),
                             F_simpleShear(0.001058, -0.000615),
                             F_simpleShear(0.000344, 0.000603),
                             F_simpleShear(0.003 , 0.002),
                             F_simpleShear(0.000385, 0.000660),
                             F_simpleShear(0.003505, -0.000508)
                            ]
        # list of rigid body translation vectors between reference stacks. initial is identity
        self.refRigidBody = np.array([ [-0.15592773, -0.27232756, 0.0],
                                       [-0.22669358, +0.07953752, 0.0],
                                       [-0.06876816, -0.15013789, 0.0],
                                       [ 0.17037964, -0.22574348, 0.0],
                                       [-0.80909318, -0.12097859, 0.0],
                                       [ 0.272, -0.204,  0 ],
                                       [ -0.97139705, -0.23814871, 0],
                                       [+1.25943623, -0.78482061, 0]])

        #make global so that predictor can be a static method and be vectorized.
        # there is probably a better way?
        # zsolt Aug 10 2022
        global refEpsilon
        refEpsilon = self.refEpsilon

        global refRigidBody
        refRigidBody = self.refRigidBody



    def __call__(self): pass

    def sed(self,frame):
        sedPos = super().sed(frame)
        return sedPos

    def gel(self, frame,step=None, gelGlobal: bool = False):
        gelPos = super().gel(frame,step=step, gelGlobal=False)
        return gelPos

    def gelGlobal2Local(self, gelGlobalGen): pass

    def log(self): pass

    def setPlotStyle(self): pass

    def posDict(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'rheo_sedHeight'): pass

    def posList(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'imageStack'):
        out = super().posList(posKey_frmt=posKey_frmt, coordTuple=coordTuple,
                              units=units, coordSys=coordSys)
        return out


    def loadStepParam(self,step):
        """
        This changes the attributed of the GelGlobal instance to reflect the step parameter
        passed to the function. More or less a partial reinitialization of just step specific
        attributes.
        """
        print('Changing step parameters to step {} without reinitializing GelGlobal instance'.format(step))

        # go to stem, then down to step folder and load step param
        os.chdir(self.exptDir)
        os.chdir(step)
        with open(self.stepParamFile, 'r') as f: self.stepParam = yaml.load(f, Loader=yaml.SafeLoader)

        self.step = self.stepParam['step']
        self.paths = self.stepParam['paths']
        self.name = self.stepParam['name']
        self.nameLong = self.stepParam['nameLong']
        #self.frames = self.stepParam['frames']
        self.frames = self.globalParam['experiment']['frames'][self.step]

        self.dpl = dpl.dplHash(self.paths['dplMetaData'])
        self.dplMeta = self.dpl.metaData
        self.hash_df = self.dpl.hash_df

        # set parameter posDF and interfaceidx
        self.interfaceIdx = da.readOvitoIdx(self.paths['sed_interface_idx'])


    def parseStep(self, step=None, verbose=False):
        """
        Decides, given the step name 'a_imageStack' or 'e_postShearFullStack' whether
        this is a reference stack or an imageStack. Simple string parsing with defined
        return value as either 'imageStack' or 'refStack'
        """
        if step is None: parsed = self.stepParam.split(sep='_')
        else: parsed = step.split(sep='_')
        if verbose: print('Parsing step to either imageStack or refStack based on step_param/step. \
                           Assuming not imageStack is refStack')
        if parsed[1] == 'imageStack': return 'imageStack'
        else: return 'refStack'

    def fitSurface(self,
                   mat: str = 'sed',
                   coordStr: str = '(um, imageStack)',
                   save2HDF: bool = True,
                   **kwargs) -> dict:
        """
        This function computes the surface fit for each frame in the step
        If the step is a refernce step, then the loop over frames is just a single
        round: range(self.frames) -> range(1) -> 0

        """
        # decide on what to do
        if coordStr != '(um, imageStack)' or mat != 'sed':
            msg = 'calling fitSurface to mat {} and coordStr {}'.format(mat, coordStr)
            raise KeyError(msg)

        # do it
        out = {}
        for frame in range(self.frames):
            posDF = self.sed(frame)
            out[frame] = da.fitSurface_singleTime(posDF, self.interfaceIdx.intersection(posDF.index), coordStr)
        outDF = pd.DataFrame(out).T

        # decide on how to send output back
        if save2HDF: outDF.to_hdf(self.paths['interfaceFits'], 'interfaceFits')
        #self.interfaceFits[]
        return outDF

    def fitExptSurface(self):
        """
        Fit all sed/gel interfaces in the expt step list.
        Return a dictionary with keys matching self.exptStepList
        Pandas dataframe formatting is dealt with in a different function
        """
        out = {}
        for step in self.exptStepList:
            self.loadStepParam(step)
            out[step] = self.fitSurface(save2HDF=False)
        return out

    def formatSurfaceFits(self, surfaceFit: dict, loadBool: bool = True):
        """
        Takes as input surfaceFit output of self.fitExptSurface
        and formats pandas dataFrame with extra keys and one continuous index
        """
        if loadBool:
            print('Loading surface fits from file')
            self.surfaceFits = pd.read_hdf(self.exptPath + '/interfaceFitsGlobal.h5')
            return self.surfaceFits
        else:
            outDF = pd.DataFrame([])
            for step in self.exptStepList:
                tmp = surfaceFit[step]

                # add extra keys
                tmp['ref_img'] = self.parseStep(step=step)
                tmp['local frame'] = tmp.index
                tmp['step'] = step
                outDF = outDF.append(tmp)
            outDF.reset_index(inplace=True)
            self.surfaceFits = outDF
            outDF.to_hdf(self.exptPath + '/interfaceFitsGlobal.h5', 'interfaceFits')
            return outDF

    global simpleShearRefList
    @staticmethod
    @tp.predict.predictor
    def simpleShearRefList(t1, particle):
        #epsilonList = [np.array([[1, 0], [0.6 / 3, 1]]), np.array([[1, 0], [0.6 / 3, 1]])]
        F = refEpsilon[int(t1-1)]
        #rigidBody = np.array([-1*refRigidBody[int(t1-1)][0],refRigidBody[int(t1-1)][1],0])
        rigidBody = refRigidBody[int(t1-1)]
        #return particle.pos @ epsilon + refRigidBody[int(t1)]
        return F@particle.pos + rigidBody

    def trackGelGlobal(self, stepListFlag='all', verbose=False):
        """
        Stitch and track gel tracers across experiments, including reference stacks

        Refactored from da.static.stitchGelGlobal
        -Zsolt Nov 23, 2021

        ToDo: Modify to have the possibility of trackGelGlobal through mulitple reference stacks: ie
                 ref, a, ref2, b, ref3, ref4, c, ... g ref9,ref10,...ref 23.
                 Its possible this is alreafy compatible with that, but its worth checking
              -Zsolt Nov 27 2021

        Refactored from analysis_abc/stitchTrack/trackGelGlobal
        -Zsolt Aug 23 2022

        """

        # global_stitch = self.gelGlobal['path']
        if stepListFlag == 'all':
            global_stitch = os.path.join(self.exptPath, self.gelGlobal['path'])
            global_mIdx_path = os.path.join(os.path.join(self.exptPath, self.gelGlobal['mIdx']))
            predictor=None
        elif stepListFlag == 'ref':
            global_stitch = os.path.join(self.exptPath,'./referenceStack_gelGlobal.h5')
            global_mIdx_path = os.path.join(self.exptPath, './referenceStack_gelGlobal_mIdx.h5')
            predictor = simpleShearRefList.__func__
        else:
            raise Warning('stepList was not all or ref, assuming it is a list of strings specifying steps')
        max_disp = self.gelGlobal['max_disp']
        print('max disp is {}'.format(max_disp))

        # linked list of step kwrds (ie ref, a,b,c..)  and relative paths [./path/to/a/, path/to/b/, ... ]
        #stepList = [list(step.keys())[0] for step in self.exptStepList]
        #pathList = [list(step.values())[0] for step in self.exptStepList]
        if stepListFlag == 'all': stepList = self.exptStepList
        elif stepListFlag == 'ref': stepList = [ref for ref in self.exptStepList if self.parseStep(ref) =='refStack']
        pathList = ['/{}'.format(step) for step in stepList]

        #make stepList dataFrame
        stepList_DF = self.surfaceFits[self.surfaceFits['local frame'] == 0][['c', 'step']].reset_index().reset_index().rename(
            {'level_0': 'step_index'}, axis=1).set_index('step')
        """ Example 
                              step_index  index           c
            step                                               
            a_preShearFullStack            0      0    143.3876
            a_imageStack                   1      1   35.228547
            b_preShearFullStack            2     23  142.518483
            ...                    
        And you can query by prevStep = stepList_DF.loc[step]['step_index'] - 1
        """
        if stepListFlag == 'ref':
            stepList_DF = stepList_DF.loc[stepList]
            stepList_DF['expt_index'] = stepList_DF['step_index']
            stepList_DF['step_index'] = range(stepList_DF.shape[0])

        mIdx_tuples = []
        tMax_list = []
        # create the single large stitched h5 file to mimic locating all the gel regions all at once, across experiments
        # one file to rule them all, also not the force overwrite by calling with 'w' flag
        with tp.PandasHDFStoreBig(global_stitch, 'w') as s:
            for step in stepList:
                #for step in self.exptStepList:

                #load step specific parameters and change directories
                self.loadStepParam(step)
                if verbose: print('Starting step {}'.format(step))

                metaData, hash_df = self.dpl.metaData, self.hash_df

                # open corresponding stitched file
                prefix = self.gelGlobal['fName_frmtStep_prefix']
                suffix = self.gelGlobal['fName_frmtTime_suffix']
                stitchedPathDict = dict(path='./' + 'locations/', fName_frmt=prefix.format(step) + suffix)

                tMax = hash_df['t'].max() + 1
                offset = sum(tMax_list)  # note the edge case sum([]) = 0 works by default
                # TODO: add custom columns to loadStitched call trough dictionary expansion

                # determine ref and imageShift from fitted c values
                ref_idx = stepList_DF.loc[step]['step_index'] -1
                if ref_idx <0: ref_idx =0

                refShift = stepList_DF.iloc[ref_idx]['c']
                imageShift = stepList_DF.loc[step]['c']
                print(step,ref_idx, refShift, imageShift)

                #refShift = self.gelGlobal['ref2ImageZ']['ref']  # 143
                #imageShift = self.gelGlobal['ref2ImageZ']['image']  # 29
                for frame, data in enumerate(da.loadStitched(range(tMax),
                                                             posKeys=self.posKeys_dict['gel'],
                                                             colKeys=['frame'], **stitchedPathDict)):
                    # for frame, data in enumerate(loadStitched(range(tMax), **stitchedPathDict)):
                    dtype_dict = da.insersectDict(self.dtypes, dict(data.dtypes))
                    data = data.astype(dtype_dict)
                    data['frame_local'] = data['frame']

                    # increment to prevent tp from overwriting the same frame number at different steps
                    data['frame'] = data['frame'] + offset
                    data['step'] = step
                    #if step == 'ref':
                    # ToDo:
                    #   -there is something wrong with the values that are being subtracted/added to reference
                    #    stacks
                    #   - this leads to a loss of trajectories every time a new reference stack is pass through, but
                    #     doesnt seem to affect tracking within the step.
                    if self.parseStep(step) == 'refStack':
                        print('refStack: {}'.format(self.step))
                        data['z (um, refStack)'] = data['z (um, imageStack)']

                        # I think this has the wrong sign
                        #data['z (um, imageStack)'] = data['z (um, refStack)'] - (refShift - imageShift)
                        data['z (um, imageStack)'] = data['z (um, refStack)'] + (refShift - imageShift)
                    else:
                        data['z (um, refStack)'] = data['z (um, imageStack)'] + (refShift - imageShift)
                    data['step'] = step

                    # add rheo_sedHeight columns for gel
                    data['x (um, rheo_sedHeight)'] = data['x (um, imageStack)']
                    data['y (um, rheo_sedHeight)'] = -1*data['y (um, imageStack)']
                    data['z (um, rheo_sedHeight)'] = data['z (um, imageStack)'] # include the z-shift so that sed/gel serves as fidicial marker.

                    # add the data to the hdfStore
                    s.put(data)
                    mIdx_tuples.append((step, frame))

                # increment now, after looping. Also, I explicitly checked off-by-one errors here.
                tMax_list.append(tMax)

        stitch_param = self.gelGlobal['stitch_param']
        with tp.PandasHDFStoreBig(global_stitch) as s:  # now track
            for linked in tp.link_df_iter(s, max_disp, predictor=predictor, **stitch_param): s.put(linked)

        # assign mIdx to attribute
        tmp = pd.DataFrame(pd.MultiIndex.from_tuples(mIdx_tuples), columns=['mIdx'])
        tmp['step'] = tmp['mIdx'].apply(lambda x: x[0])
        tmp['frame local'] = tmp['mIdx'].apply(lambda x: x[1])

        self.gelGlobal_mIdx = tmp
        # save to dataFrame
        #self.gelGlobal_mIdx.to_hdf(os.path.join(self.exptPath, self.gelGlobal['mIdx']), 'mIdx')
        self.gelGlobal_mIdx.to_hdf(global_mIdx_path, 'mIdx')
        return mIdx_tuples, global_stitch



    def simpleShearFit(self, dataFrame: pd.DataFrame,
                       coordSys: str = '(um, imageStack)', cutBins: int = 25) -> dict:
        """
        Given a dataFrame of gelPositions and displacements
        -cut/groupby on z and apply median
        -fit xz and yz median strains
        -output dictionary with labels
        """
        tmp = dataFrame.groupby(pd.cut(dataFrame['z (um, imageStack)'], cutBins)).median()
        xzParam = {'xKey': 'z {}'.format(coordSys), 'yKey': 'dx {}'.format(coordSys)}
        yzParam = {'xKey': 'z {}'.format(coordSys), 'yKey': 'dy {}'.format(coordSys)}
        xz = da.linearFit(tmp, **xzParam)
        yz = da.linearFit(tmp, **yzParam)
        out = dict(exz=xz.params[1], eyz=yz.params[1], dx=xz.params[0], dy=yz.params[0])
        return out


if __name__ == '__main__':
    stem = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp'
    inst = GelGlobal(globalParamFile=stem+'/tfrGel23042022_strainRamp_globalParam.yml',
                     stepParamFile='./a_preShearFullStack/step_param.yml')
    #surfaceFit = inst.fitExptSurface()
    surfaceFit_DF = inst.formatSurfaceFits(None, loadBool=True)
    gelGlobal = inst.trackGelGlobal(stepListFlag='ref',verbose=False)
    #gelGlobal = inst.trackGelGlobal(stepListFlag='all',verbose=True)

