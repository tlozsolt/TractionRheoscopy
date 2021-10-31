import pandas as pd
from data_analysis import static as da
import trackpy as tp
import yaml
import os
import numpy as np
from datetime import datetime

"""
Maybe a useful abstraction is to create an instance of an experiment class that
   has as attributes: 
   - paths to data pyTables file
   - frame number etc,
   - h5.index table and access functions
It wont hold any of the data, should be quick to create a ''fresh'' instance of,
   and would mostly serve to encapsulate the metaData of analysis procedures that are
   are loaded in jupyter lab notebook. 
Instantiating the class would be the first step
   in any experiment specific jupyter notebook. 
"""

class ExperimentStep:
    """
    Class to hold data analysis methods and metaData for a specific step
    in an experiment

    Initalize by passing path to data analysis parameter file for that step

    I hope to create an Experiment wrapper class that has each ExperimentStep as an attribute
    as in, for example, shearRun1029201.f = ExperimentStep('./path/to/step/f/directory/paramFile.yml')

    Loop over frames lazy loading and column keys using:
        out = []
        for frame in inst.frames:
            posDF = inst.sed(frame)[['list', 'of', 'column', 'keys']]
            out.append(posDF).mean()

    Attributes:

        da_param: dictionary of data analysis parameters loaded from yaml
                  file when the class was instantiated

        paths: list of paths to h5 files read from param file. The 'paths' dictionary
               key from the da_param

        frames: range() spanning the entire number of frames in this experimental step

        dim: dictionary of dimensions

        interfaceIdx: parsed pandas index of sed_gel interface particle ids

        dim: dimensions of the sample in um

        interfaceFits: pandas DataFrame of fits to sed-gel interface, by default
                       computed over the sediment particles found by looking at
                       sample in ovito. If [path][interfaceFits] file exists, it
                       will be loaded rather than recomputed by calling
                       self.fitSurface()

        refFit: pandas DataFrame of fits to nearest prior reference stack


    """
    def __init__(self, paramFile):

        with open(paramFile, 'r') as f: self.da_param = yaml.load(f, Loader=yaml.SafeLoader)

        self.name = self.da_param['name']
        self.nameLong = self.da_param['nameLong']
        self.paths  = self.da_param['paths']
        self.frames = range(self.da_param['frames'])
        self.dim = self.da_param['dimensions']
        self.keySets = self.da_param['keySets']

        # attributes for logging data analysis
        self.logPath = os.path.abspath('data_analysis_log.yaml')


        # Optional attributes. Will default to None if there is an error
        # These attributes must be populated using child classes like
        # dataCleaning for example.
        self.interfaceIdx = da.readOvitoIdx(self.paths['sed_interface_idx'])
        self.interfaceFits = None
        self.refFit = None

    # method to create fresh generator for sed or gel pos
    def _gel(self):
        """
        with inst.gel() as s:
           for frame, posDF in enumerate(s):
               fucntion_of_posDF(posDF)
        """
        return tp.PandasHDFStoreBig(self.paths['gel_pos'])

    def _sed(self): return tp.PandasHDFStoreBig(self.paths['sed_pos'])

    def gel(self, frame: int) -> pd.DataFrame:

        with self._gel() as s: out = s.get(frame)

        out.set_index('particle', inplace=True)
        out.sort_index(inplace=True)
        return out

    def sed(self, frame: int) -> pd.DataFrame:

        with self._sed() as s: out = s.get(frame)

        out.set_index('particle', inplace=True)
        out.sort_index(inplace=True)
        return out

    def getFrame(self,mat: str, frame: int) -> pd.DataFrame:
        """
        Return a single frame
        """
        if mat == 'gel':
            with self._gel() as s:
                out = s.get(frame)
                out.set_index('particle', inplace=True)
                out.sort_index(inplace=True)
                return out

        elif mat == 'sed':
            with self._sed() as s:
                out = s.get(frame)
                out.set_index('particle', inplace=True)
                out.sort_index(inplace=True)
                return out

        else: return False

    def getFrames(self,mat: str, frames: list = []):
        """
        Create a lazy loading generator over a list of frames with
        default behavior on empty list to return a generator over all
        frames
        """
        if mat == 'gel':
            with self._gel() as s:
                if frames == []: frames = list(range(s.max_frame + 1))
                for frame in frames:
                    out = s.get(frame)
                    out.set_index('particle', inplace=True)
                    out.sort_index(inplace=True)
                    yield out

        elif mat == 'sed':
            with self._sed() as s:
                if frames == []: frames = list(range(s.max_frame + 1))
                for frame in frames:
                    out = s.get(frame)
                    out.set_index('particle', inplace=True)
                    out.sort_index(inplace=True)
                    yield out

        else: return False

    def log(self, key, newDict):
        """
        Update data analysis yml file to log progress.
        The basic idea is that at every data analysis step I would
        update the data_analysis file and write any attributes
        directly to the file. Then whenever **any** child class
        is loaded (for example dataCleaning) it would autopopulate
        with all past progress.
        Anytime a file is written, write an entry to the paths directory
        """
        self.da_param[key] = newDict
        self.da_param['time last updated'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(self.logPath, 'w') as f:
            yaml.dump(self.da_param,f)

    def addPath(self, key, path):
        """
        Just add a path data analysis yaml file
        """
        self.paths[key] = path
        self.log('paths', self.paths)



class dataCleaning(ExperimentStep):

    def __init__(self, paramFile):
        super().__init__(paramFile) # inherit everything from ExperimentStep parent calss upon instantiation

        # add attributes specific to dataCleaning
        self.interfaceFits = None
        self.refFit = None

    def fitSurface(self,
                   mat: str = 'sed',
                   coordStr: str = '(um, imageStack)',
                   add2Inst: bool = False,
                   save2HDF: bool = True,
                   forceRecompute: bool = False) -> dict:
        """
        This function computes the surface fit for each frame
        By default this will not recompute if h5 file exists
        and will not add to instance.
        """
        if forceRecompute == False and os.path.exists(self.paths['interfaceFits']):
            outDF = pd.read_hdf(self.paths['interfaceFits'])
            if add2Inst: self.interfaceFits = outDF
            return outDF

        else:
            if mat == 'sed':
                posDF_gen = self._sed()
                particleIdx = self.interfaceIdx
            elif mat == 'gel':
                print('Warning: calling fitSurface with gel coordinates')
                posDF_gen = self._gel()

            out = {}
            for frame, posDF in enumerate(posDF_gen):
                posDF.set_index('particle', inplace=True)
                idx = posDF.index.intersection(particleIdx)
                surf = da.fitSurface_singleTime(posDF, idx, coordStr=coordStr)
                out[frame] = surf

            # decide on the output
            outDF = pd.DataFrame(out).T
            if add2Inst:
                self.interfaceFits = outDF
            if save2HDF:
                outDF.to_hdf(self.paths['interfaceFits'], 'interfaceFits')
                return self.paths['interfaceFits']
            else:
                return outDF

    def fitReference(self,
                     coordStr: str = '(um, imageStack)',
                     add2Inst: bool = True) -> pd.DataFrame:

        # open the data store
        with tp.PandasHDFStoreBig(self.paths['reference_sed_pos']) as s:
            sedRef = s.get(0)

        # reset the index to particle id
        sedRef.set_index('particle', inplace=True)

        # read the index of ovito ids
        refSedFit = da.readOvitoIdx(self.paths['reference_sed_interface_idx'])

        # compute the fit
        refFit = da.fitSurface_singleTime(sedRef, refSedFit, '(um, imageStack)')

        # decide on output
        out = pd.DataFrame({'refFit': refFit}).T
        if add2Inst: self.refFit = out
        return out

    def cleanSedGel(self,
                    mat: str,
                    pos_keys: list = ['{} {}'.format(x, '(um, imageStack)') for x in ['z', 'y', 'x']],
                    dist_key: str = 'dist from sed_gel interface (um, imageStack)',
                    keepBool_str: str = 'cleanSedGel_keepBool') -> pd.DataFrame:
        """
        Given an existing pandas dataStore of gel or sed particles
          add a new column of booleans on whether to keep particles
        Stream directly to this file, dont overwrite, just update with new column

        Have the column key defined in external parameter file under the keyword dataCleaning

        In principle I could in the future decide to throw away particle for other reasons
        (incomplete traj, high prob of chunk etc) in which I want the column clean or keep to be the product
        of all the flags *_keepBool

        There should be a function in da.static that takes a dataFrame and compute keepBool as the product of all columns
        with *_keepBool found by pattern matching the column keywords.
        """

        # load a single frame, if out_bool is present, then just return...maybe print a statement
        if mat == 'gel':
            posDF = self.gel(0)
        elif mat == 'sed':
            posDF = self.sed(0)
        # saying that the h5 file of positions is already cleaned
        if keepBool_str in posDF.keys():
            print('Positions already have cleanSedGel() with column key {}'.format(keepBool_str))
            return True

        else:

            # this is an awkward construct...really is it necessary?
            if self.interfaceFits is None:
                interfaceFitDF = self.fitSurface(mat)
            else:
                interfaceFitDF = self.interfaceFits

            # compute the new columns: dist from interface
            if mat == 'sed':
                posDF_gen = self._sed()
            elif mat == 'gel':
                posDF_gen = self._gel()

            # loop over frames
            for frame, posDF in enumerate(posDF_gen):

                # set particle index
                posDF.set_index('particle', inplace=True)

                # compute distance from plane
                fit_tmp = interfaceFitDF.loc[frame][['a', 'b', 'c']]
                posDF[dist_key] = da.pt2Plane(posDF[pos_keys].values, fit_tmp.values)

                # flag sed particles that are below the interface, and gel particles above the interface
                z_cutoff = self.da_param['interface_fit']['clean_cutoff']['garbage']['sed_{}'.format(mat)]
                if mat == 'sed':
                    posDF[keepBool_str] = posDF[dist_key] > z_cutoff
                    # add rheo coordinates
                    posDF['x (um, rheo_sedHeight)'] = posDF['x (um, imageStack)'] - self.dim['x (um)'] / 2.0
                    posDF['y (um, rheo_sedHeight)'] = -1 * (posDF['y (um, imageStack)'] - self.dim['y (um)'] / 2.0)
                    posDF['z (um, rheo_sedHeight)'] = posDF[dist_key]

                elif mat == 'gel':
                    posDF[keepBool_str] = posDF[dist_key] < z_cutoff

                    # compute height above coverslip
                    if self.refFit is None: self.fitReference()
                    z_offset = self.refFit.loc['refFit']['c'] - self.interfaceFits.loc[frame]['c']

                    # add rheo coordinates
                    posDF['x (um, rheo_sedHeight)'] = posDF['x (um, imageStack)'] - self.dim['x (um)'] / 2.0
                    posDF['y (um, rheo_sedHeight)'] = -1 * (posDF['y (um, imageStack)'] - self.dim['y (um)'] / 2.0)
                    posDF['z (um, rheo_sedHeight)'] = posDF['z (um, imageStack)'] + z_offset

                # update the data store.. is it really this simple?
                posDF_gen.put(posDF.reset_index())
        return True

    def xyzClean(self,
                 fNameFlag: str='',
                 keySet: list = ['locations_rheoSedHeight', 'uncertainty'],
                 keepBool_str: str = 'cleanSedGel_keepBool',
                 forceWrite: bool = False) -> str:
        """
        Write a time series of xyz files to xyz directory
        for given keyset
        """

        # set the outPath, using relative path
        outPath = './xyz/{}/'.format(keepBool_str)

        for mat in ['sed','gel']:

            # deal with the two materials
            if mat == 'sed':
                fName_frmt = self.name + fNameFlag + '_sed_t{:03}.xyz'
                posDF_gen = self._sed()

            elif mat == 'gel':
                fName_frmt = self.name + fNameFlag + '_gel_t{:03}.xyz'
                posDF_gen = self._gel()

            if os.path.exists(outPath + fName_frmt.format(0)):
                if forceWrite is False:
                    print('xyz files already exist')
                    return True
                else: print('xyz files will be overwritten')

            # loop over the frames
            for frame, posDF in enumerate(posDF_gen):

                # set up the output filename, formatted to frame
                fName = fName_frmt.format(frame)

                # set particle index
                posDF.set_index('particle', inplace=True)

                # select only the clean positions
                out = posDF[posDF[keepBool_str]]

                # create list of output col keys
                out_cols = []
                for key in keySet:
                    for col in self.keySets[mat][key]: out_cols.append(col)

                # output
                da.df2xyz(out[out_cols], outPath, fName)
        print('xyz files output with keySet {} with selection on {}'.format(keySet, keepBool_str))
        return os.path.abspath(outPath)









