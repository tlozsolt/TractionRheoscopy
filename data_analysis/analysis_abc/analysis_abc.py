from abc import ABC, abstractmethod

import pandas as pd
import yaml
from particleLocating import dplHash_v2 as dpl

import data_analysis.static
import trackpy as tp
from datetime import datetime

"""
-Should also add a call method, maybe to the class instances, but not to abc Analysis?
Completely untested as of Oct 26 2021
-Zsolt
"""

class Analysis(ABC):

    def __init__(self, globalParamFile, stepParamFile):

        self.abcParam = dict(globalParamFile=globalParamFile, stepParamFile=stepParamFile)

        with open(globalParamFile, 'r') as f: self.globalParam = yaml.load(f,Loader=yaml.SafeLoader)
        with open(stepParamFile, 'r') as f: self.stepParam = yaml.load(f,Loader=yaml.SafeLoader)

        # some useful things from global parameters
        self.keySets = self.globalParam['keySets']
        self.exptStepList = self.globalParam['experiment']['steps']
        self.dim = self.globalParam['experiment']['dimensions']
        self.rheo = {'gelModulus' : self.globalParam['experiment']['gelModulus'],
                     'gelThickness':  self.globalParam['experiment']['gelThickness']}
        self.dtypes = self.globalParam['locatingOutput']['dtypes']
        self.gelGlobal = self.globalParam['experiment']['gelGlobal']

        #legacy
        self.zyx = ['z','y','x']
        self.xyz = ['x','y','z']

        #global gel deformation

        # some useful attributes specific to a step
        # paths are **always** given under steps until I write a function to concatenate or form a union of paths
        # specified in global and step

        self.step = self.stepParam['step']
        self.paths = self.stepParam['paths']
        self.name = self.stepParam['name']
        self.nameLong = self.stepParam['nameLong']
        #self.frames = self.stepParam['frames']
        self.frames = self.globalParam['experiment']['frames'][self.step]

        self.dpl = dpl.dplHash(self.paths['dplMetaData'])
        self.dplMeta = self.dpl.metaData
        self.hash_df = self.dpl.hash_df

        self.posKeys_dict = {
            'sed': ['z (px, hash)', 'y (px, hash)', 'x (px, hash)', 'hashValue',
                    'x (um, imageStack)', 'y (um, imageStack)', 'z (um, imageStack)',
                    'x_std', 'y_std', 'z_std', 'cost', 'totalError', 'n_iteration',
                    'size_z', 'size_y', 'size_x',
                    'mass', 'raw_mass', 'signal', 'signal_std', 'background', 'background_std',
                    'sed_Colloid_core', 'sed_Colloid_shell',
                    'fluorescent_chunk_core', 'fluorescent_chunk_shell',
                    'gel_Tracer_core', 'gel_Tracer_shell',
                    'sed_Background_core', 'sed_Background_shell',
                    'nonfluorescent_chunk_core', 'nonfluorescent_chunk_shell',
                    'gel_Background_core', 'gel_Background_shell'],
            'gel': ['z (px, hash)', 'y (px, hash)', 'x (px, hash)', 'hashValue',
                    'mass', 'raw_mass', 'n_iteration',
                    'disc_size', 'size_x', 'size_y', 'size_z',
                    'size_z_std', 'size_y_std', 'size_x_std', 'disc_size_std',
                    'background', 'background_std',
                    'signal', 'signal_std',
                    'z_std', 'y_std', 'x_std', 'cost',
                    'x (um, imageStack)', 'y (um, imageStack)', 'z (um, imageStack)',
                    'totalError', 'keepBool', 'ep_z', 'ep_y', 'ep_x',
                    'gel_Background', 'gel_Tracer',
                    'sed_Colloid', 'sed_Background',
                    'fluorescent_chunk', 'nonfluorescent_chunk']}

        try: self.gelGlobal_mIdx = pd.read_hdf(self.gelGlobal['mIdx'])
        except FileNotFoundError: pass

    @abstractmethod
    def __call__(self): pass

    #def frameGen(self, path):
    #    """
    #    Given a a path to hdf file of tracked particle positions, return a generator over
    #    """
    #    return tp.PandasHDFStoreBig(path)

    #@abstractmethod
    #def _sed(self):
    #    return tp.PandasHDFStoreBig(self.paths['sed'])
    #    #return self.frameGen(self.paths['sed'])

    #@abstractmethod
    #def _gel(self):
    #    return tp.PandasHDFStoreBig(self.paths['gel'])
    #    #return self.frameGen(self.paths['gel'])

    @abstractmethod
    def sed(self, frame):
        with tp.PandasHDFStoreBig(self.paths['sed']) as s:
            out = s.get(frame)
        out.set_index('particle', inplace=True)
        out.sort_index(inplace=True)
        return out

    @abstractmethod
    def gel(self, frame, step=None, gelGlobal: bool = True):
        # write gel method to default to gel global stiched file
        # and by default return dataFrame in the same format as previous gelStep() method
        if gelGlobal:
            if step is None: step=self.step

            mIdx = pd.MultiIndex.from_tuples(self.gelGlobal_mIdx['mIdx'])
            frame = mIdx.get_loc((step, frame))
            #tmp = {}
            with tp.PandasHDFStoreBig(self.gelGlobal['path'],'r') as steps:
                out =steps.get(frame)
                #tmp[frame] = steps.get(frame).set_index(['particle'])
            #out.set_index('particle', inplace=True)
            #out.sort_index(inplace=True)
            #return out
            #return pd.DataFrame(tmp[frame]).sort_index()
        else:
            with tp.PandasHDFStoreBig(self.paths['gel']) as s: out = s.get(frame)
        out.set_index('particle', inplace=True)
        out.sort_index(inplace=True)
        return out

    @abstractmethod
    def gelGlobal2Local(self, gelGlobalGen):
        """
        This is will a trackpy hdfStore generator for global gel
        and return a generator that can be index as if it was local.

        Old code segment for step dependent gel locations and tracks
            for frame, posDF in enumerate(tp.PandasHDFStoreBig(self.paths['gel']):
                foo(frame, posDF)

        new code segments that should be equivalent to above:
            globalGen = tp.PandasHDFStoreBig(self.gelGlobal['paths']
            gen = self.gelGlobal2Local(gen)
            for frame, posDF in enumerate(gen):
                foo(frame, posDF)
            # then for garbage collection, manually close the global generator as it doesnt reach the end
            globalGen.close()
        """
        # dataFrame with columns 'mIdx', 'step', and 'frame local' with index of frame
        gelGlobal = self.gelGlobal_mIdx

        # select only the step you are on now
        gelLocal = gelGlobal[gelGlobal['step'] == self.step]

        # what the upper and lower bounds of the global index for this step?
        stepMin, stepMax = gelLocal.index.min(), gelLocal.index.max()

        for frameGlobal, posDF in enumerate(gelGlobalGen): # interate over global index
            if frameGlobal < stepMin or frameGlobal > stepMax:
                # ignore iterations outside bounds
                continue
            else:
                # if within bounds, yield the posDF
                yield posDF

    #def gelStep(self, frame:int):
    #    with tp.PandasHDFStoreBig(self.paths['gel']) as s: out = s.get(frame)
    #    out.set_index('particle', inplace=True)
    #    out.sort_index(inplace=True)
    #    return out

    #@abstractmethod
    #def writeXYZ(self, mat: str, outPath: str, keySet: str):
    #    with tp.PandasHDFStoreBig(self.paths[mat]) as s:
    #        for frame, posDF in enumerate(s):
    #            posDF.set_index('particle', inplace=True)
    #            data_analysis.static.df2xyz(posDF[self.keySets[keySet]],outPath,'t{:03}.xyz'.format(frame))
    #    return True


    @abstractmethod
    def setPlotStyle(self):
        """ Loads plot style from globalParamFile"""

        # there is probably a more elegant way to do this
        import seaborn as sns
        plotStyle = self.globalParam['plotStyle']

        context = plotStyle['context']
        figsize = plotStyle['figsize']
        font_scale = plotStyle['font_scale']

        sns.set(rc={'figure.figsize': figsize})
        sns.set_context(context,font_scale=font_scale)

    @abstractmethod
    def posDict(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'rheo_sedHeight'):
        """
        Create a dictionary of key value from standard position format 'coord (units, coordSys)'
        """
        return {'{}'.format(coord=coord):
                    posKey_frmt.format(coord=coord, units=units, coordSys=coordSys) for coord in coordTuple}

    @abstractmethod
    def posList(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'rheo_sedHeight'):
        """
        Create a dictionary of key value from standard position format 'coord (units, coordSys)'
        """
        return [posKey_frmt.format(coord=coord, units=units, coordSys=coordSys) for coord in coordTuple]

    @abstractmethod
    def log(self):
        # get all local variables including class attributes
        #args = self.__dict__

        ##pop off any attributes you dont want output to yaml
        #args.pop('keySets')

        ## add anything you do want to output to yaml
        #args['time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        #return args
        pass




