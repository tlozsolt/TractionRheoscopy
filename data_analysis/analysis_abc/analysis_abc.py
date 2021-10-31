from abc import ABC, abstractmethod
import yaml

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
        with open(globalParamFile, 'r') as f: self.globalParam = yaml.load(f,Loader=yaml.SafeLoader)
        with open(stepParamFile, 'r') as f: self.stepParam = yaml.load(f,Loader=yaml.SafeLoader)

        # some useful things from global parameters
        self.keySets = self.globalParam['keySets']
        self.exptStepList = self.globalParam['experiment']['steps']
        self.dim = self.globalParam['experiment']['dimensions']

        # some useful attributes specific to a step
        # paths are **always** given under steps until I write a function to concatenate or form a union of paths
        # specified in global and step

        self.paths = self.stepParam['paths']
        self.name = self.stepParam['name']
        self.nameLong = self.stepParam['nameLong']
        self.frames = self.stepParam['frames']

        self.posKeys_dict = {
            'sed': ['z (px, hash)', 'y (px, hash)', 'x (px, hash)', 'hashValue',
                    'x (um, imageStack)', 'y (um, imageStack)', 'z (um, imageStack)',
                    'x_std', 'y_std', 'z_std', 'cost', 'totalError', 'size', 'n_iteration',
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
    def gel(self, frame):
        with tp.PandasHDFStoreBig(self.paths['gel']) as s:
            out = s.get(frame)
        out.set_index('particle', inplace=True)
        out.sort_index(inplace=True)
        return out

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
    def log(self):
        # get all local variables including class attributes
        #args = self.__dict__

        ##pop off any attributes you dont want output to yaml
        #args.pop('keySets')

        ## add anything you do want to output to yaml
        #args['time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        #return args
        pass




