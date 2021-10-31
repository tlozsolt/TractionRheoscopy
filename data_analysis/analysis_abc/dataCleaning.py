from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis import static as da
import pandas as pd
import os
import numpy as np
import yaml
from datetime import datetime
import trackpy as tp
import os

class Cleaning(Analysis):

    def __init__(self, globalParamFile, stepParamFile):
        super().__init__(globalParamFile, stepParamFile)

        self.interfaceIdx = da.readOvitoIdx(self.paths['sed_interface_idx'])
        self.interfaceFits = None
        self.refFit = None


    #def _sed(self): super()._sed()

    #def _gel(self): super()._gel()

    def __call__(self):
        # all functions should by design take a material string or posDF_gen
        # and output a dictionary of summary results

        # this part should be part of the call function
        # but i have already coded to be part of individual functions
        #
        # create the generator for the str
        #if mat =='sed': posDF_gen = self._sed()
        #elif mat == 'gel': posDF_gen = self._gel()

        # call fit surface with default values
        print('Fitting interface with sed particles from ovito')
        p = self.stepParam['cleaning']
        self.fitSurface(**p['fitSurface'])

        # call fitRefernce
        print('fitting refenence stack to find aboslute thickness of the gel')
        self.fitReference(**p['fitReference'])

        # call cleanSedGel on both sed and gel
        print('Flagging particles that should be removed from further analysis')
        print(' due to gel particles above sed/gel interface or sed particles below')
        self.cleanSedGel(mat='sed', **p['cleanSedGel'])
        self.cleanSedGel(mat='gel', **p['cleanSedGel'])

        # output to xyz for both sed and gel
        print('writing xyz files')
        self.xyzClean(mat='sed', **p['xyzClean'])
        self.xyzClean(mat='gel', **p['xyzClean'])

        #output log file
        with open('./log.yml', 'w') as f: yaml.dump(self.log(),f)

    def setPlotStyle(self):
        super().setPlotStyle()
        #optionally set other plot style parameters here

    def sed(self, frame):
        sedPos =  super().sed(frame)
        #potentiall do more things like boolean selection on sedPos
        return sedPos

    def gel(self, frame):
        gelPos = super().gel(frame)
        # potentially do more thing like boolean section for Cleaning specific behavior
        return gelPos

    def log(self):
        #args = super().getArgs()
        ## pop off anything you dont want output to yaml dict
        #args.pop('globalParam')
        #args['Cleaning'] = self.__dict__
        #args.pop('refFit')
        #args.pop('interfaceIdx')
        #args.pop('interfaceFits')
        # I dont know how to make this work to log input vbariables for when a method was called
        # some combination or manipulationi of locals() and self.__dict__ to remove attributes like
        # dataFrames and fits, while keep attibutes like parameters
        #local = locals()
        #local.pop('self')
        return {'global': self.globalParam,
                'step': self.stepParam,
                'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    def fitSurface(self,
                   mat: str = 'sed',
                   coordStr: str = '(um, imageStack)',
                   add2Inst: bool = False,
                   save2HDF: bool = True,
                   forceRecompute: bool = False,
                   **kwargs) -> dict:
        """
        This function computes the surface fit for each frame
        By default this will not recompute if h5 file exists
        and will not add to instance.

        All of these parameters will be loaded into stepParam under kwarg of calling function
        and then wrapped into __call__ method for class
        and so when this is wrapped into __call__ it could be called like

        fitSurface(**self.stepParam['cleaning']['fitSurface'])
        """
        print(locals())

        if forceRecompute == False and os.path.exists(self.paths['interfaceFits']):
            outDF = pd.read_hdf(self.paths['interfaceFits'])
            if add2Inst: self.interfaceFits = outDF
            return outDF

        else:
            if mat == 'sed':
                posDF_gen = tp.PandasHDFStoreBig(self.paths['sed'])
                particleIdx = self.interfaceIdx
            elif mat == 'gel':
                print('Warning: calling fitSurface with gel coordinates')
                posDF_gen = tp.PandasHDFStoreBig(self.paths['gel'])

            out = {}
            for frame, posDF in enumerate(posDF_gen):
                posDF.set_index('particle', inplace=True)
                idx = posDF.index.intersection(particleIdx)
                surf = da.fitSurface_singleTime(posDF, idx, coordStr=coordStr)
                out[frame] = surf

            # close the generator, is this even necessary?
            posDF_gen.close()
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
                     add2Inst: bool = True,
                     **kwargs) -> pd.DataFrame:

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
                    garbage: dict = {'sed_sed': -1, 'sed_gel': -1.5},
                    dist_key: str = 'dist from sed_gel interface (um, imageStack)',
                    keepBool_str: str = 'cleanSedGel_keepBool',
                    forceRecompute: bool = False,
                    **kwargs) -> pd.DataFrame:
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


        I think this function should take a generator in place of a str that is used to find the right
        generator...or maybe still write a call with one input str as in Clean(stepa,'sed') which
        call step.cleanSedGel(sed, **stepa.stepParam['clean'])
        -Zsolt Oct 26 2021
        """

        # load a single frame, if out_bool is present, then just return...maybe print a statement
        if mat == 'gel':
            posDF = self.gel(0)
        elif mat == 'sed':
            posDF = self.sed(0)
        # saying that the h5 file of positions is already cleaned
        if keepBool_str in posDF.keys() and forceRecompute is False:
            print('Positions already have cleanSedGel() with column key {}'.format(keepBool_str))
            return True

        else:
            print('Recomputing column keys {}'.format(keepBool_str))
            if self.interfaceFits is None:
                interfaceFitDF = self.fitSurface(mat)
            else:
                interfaceFitDF = self.interfaceFits

            if mat == 'sed':
                posDF_gen = tp.PandasHDFStoreBig(self.paths['sed'])
            elif mat == 'gel':
                posDF_gen = tp.PandasHDFStoreBig(self.paths['gel'])
                #posDF_gen = self._gel()

            # loop over frames
            for frame, posDF in enumerate(posDF_gen):

                # set particle index
                posDF.set_index('particle', inplace=True)

                # compute distance from plane
                fit_tmp = interfaceFitDF.loc[frame][['a', 'b', 'c']]
                posDF[dist_key] = da.pt2Plane(posDF[pos_keys].values, fit_tmp.values)

                # flag sed particles that are below the interface, and gel particles above the interface
                #z_cutoff = self.da_param['interface_fit']['clean_cutoff']['garbage']['sed_{}'.format(mat)]
                if mat == 'sed':
                    z_cutoff = garbage['sed_sed']
                    posDF[keepBool_str] = posDF[dist_key] > z_cutoff
                    # add rheo coordinates
                    posDF['x (um, rheo_sedHeight)'] = posDF['x (um, imageStack)'] - self.dim['x (um)'] / 2.0
                    posDF['y (um, rheo_sedHeight)'] = -1 * (posDF['y (um, imageStack)'] - self.dim['y (um)'] / 2.0)
                    #posDF['z (um, rheo_sedHeight)'] = posDF[dist_key]
                    posDF['z (um, rheo_sedHeight)'] = posDF['z (um, imageStack)'] - self.interfaceFits.loc[frame]['c']

                elif mat == 'gel':
                    z_cutoff = garbage['sed_gel']
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

        # not necessary to close generator as, I think, it closes since we have iterator through it
        #posDF_gen.close()
        return True

    def xyzClean(self,
                 mat: str,
                 fNameFlag: str = '',
                 keySet: list = ['locations_rheoSedHeight', 'uncertainty'],
                 extraCols: list = [],
                 keepBool_str: str = 'cleanSedGel_keepBool',
                 forceWrite: bool = False,
                 **kwargs) -> str:
        """
        Write a time series of xyz files to xyz directory
        for given keyset
        """

        # set the outPath, using relative path
        outPath = './xyz/{}/'.format(keepBool_str)

        #create the directory if necesary
        try: os.mkdir(outPath)
        except FileExistsError: pass

        if mat == 'sed':
            fName_frmt = self.name + fNameFlag + '_sed_t{:03}.xyz'
            #posDF_gen = self._sed()
            posDF_gen = tp.PandasHDFStoreBig(self.paths['sed'])

        elif mat == 'gel':
            fName_frmt = self.name + fNameFlag + '_gel_t{:03}.xyz'
            #posDF_gen = self._gel()
            posDF_gen = tp.PandasHDFStoreBig(self.paths['gel'])

        if os.path.exists(outPath + fName_frmt.format(0)) and forceWrite is False:
            print('xyz files already exist')

        elif not(os.path.exists(outPath + fName_frmt.format(0)) and forceWrite is False):
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
                # add any extra column keys not in the keySet passed
                for col in extraCols: out_cols.append(col)
                # output
                da.df2xyz(out[out_cols], outPath, fName)

            # log changes
            # ideally set the current variables to output
            paramLog = self.log()
            comment ='xyz files output with keySet {} with selection on {}'.format(keySet, keepBool_str)
            paramLog['comment'] = comment
            with open(outPath +'/log.yml','w') as f: yaml.dump(paramLog,f)

        return os.path.abspath(outPath)
