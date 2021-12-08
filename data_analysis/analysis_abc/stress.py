import sys
import os
import pandas as pd
import numpy as np
import trackpy as tp

from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis.analysis_abc.dataCleaning import Cleaning
from data_analysis.analysis_abc.strain import Strain
from data_analysis import static as da

from scipy.stats.mstats import theilslopes as theilSen

class Stress(Analysis):
    """
    Compute the stress in the sample by analyzing the deformation of the gel.

    This class includes several options for computing the stress including:

        - stress relative to first frame of step
        - stress relative to reference configuration
        - stress based on linear deformation profile with rigid body displacement
        - stress computed with absolute height above coverslip (z um rheo_sedHeight)
        - stress computed with scaled height to include shape of gel (I think)
             >> avgHeight*(absolute height)/( (absolute height) - (distance to sed-gel interface) )

    The class does not plot anything, however it does produce data that can be pass to
    some functions in plots module including:
    - stress as a function of frame for both xz and yz
    - rigid body displacement
         -> ToDo: pass to tracking of sediment to improve tracks.
    - residual difference (which can be binned in xy to produce heatmaps of spatially correlated residuals)
    - data for combined fitting with density plot of points and fit
    - fit parameters for linear deformation based on:
        - tukey filtered displacement on bins, then oridinary least squares regression
        - thiel sen robust regression
        - confidence intervals based on fit uncertainties in ordinary least squares regression
        - confidence intervals base on 5-95% interval for thiel sen
    - outlier detection (return indices to visualize in ovito)
    """

    def __init__(self, globalParamFile, stepParamFile, test: bool =False):
        # load yaml metaData for both expt and step
        super().__init__(globalParamFile=globalParamFile, stepParamFile=stepParamFile)

        if test == True: self.frames = 3

        # add additional attributes I will need
        self.strainInst = Strain(**self.abcParam)

        #use the same keepBool and posCoordinateSystem as strain
        self.strain = self.stepParam['strain']
        self.posCoordinateSystem = self.strain['posCoordinateSystem']
        self.keepBool = self.strain['keepBool']


        # rheo calibration
        self.gelModulus = self.rheo['gelModulus']
        self.gelThickness = self.rheo['gelThickness']

        #cache some commonly used time points
        self.globalRefGel = self.gel(0,step='ref')
        self.stepRefGel = self.gel(0)

        # attributes that will be populated with functions in the class
        if os.path.exists('./theilSen.h5'): self.theilSen =  pd.read_hdf('./theilSen.h5',key='theilSen')
        else:
            self.theilSen = None
            #self.gelStrain_theilSen()

        # constant z bins across all frames (possibly all steps as well)
        self.zBin_dim = (118,145,3)
        self.zBins = np.arange(self.zBin_dim[0], self.zBin_dim[1] + self.zBin_dim[2], self.zBin_dim[2])

    def __call__(self,forceRecompute: bool=True):
        """
        - z scale
        - compute displacement
        - fit deformation
        - correct drift

        - update gel dataFrames

        """
        #with tp.PandasHDFStoreBig(self.paths['gel']) as s:
        #    for frame, posDF in enumerate(s):
        #        posDF = self.driftCorr(frame=frame)
        #        s.put(posDF.reset_index())

        # compute z-scaled height
        for frame in range(self.frames): self.zScaledFractionalHeight(frame=frame)

        #reset self.refGel to have new columns
        self.globalRefGel = self.gel(0, step='ref')

        # compute displacement
        for frame in range(self.frames): self.disp(frame)

        # fit the deformation
        self.gelStrain_theilSen(forceRecompute=forceRecompute)

        # correct the drift
        for frame in range(self.frames): self.driftCorrResid(frame=frame, forceRecompute=forceRecompute)

    def sed(self, frame: int):
        # inherit
        out = super().sed(frame)

        # select
        out = out[out[self.keepBool]]
        return out

    def gel(self, frame: int, step=None, gelGlobal: bool = True):
        # inherit
        out = super().gel(frame, step=step, gelGlobal=gelGlobal)
        # select
        out = out[out[self.keepBool]]
        return out

    def gelGlobal2Local(self, gelGlobalGen): return super().gelGlobal2Local(gelGlobalGen)

    def setPlotStyle(self): super().setPlotStyle()

    def posDict(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'rheo_sedHeight'):
        return super().posDict(posKey_frmt=posKey_frmt,
                               coordTuple=coordTuple,
                               units=units,
                               coordSys=coordSys)

    def posList(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'rheo_sedHeight'):
        return super().posList(posKey_frmt=posKey_frmt,
                               coordTuple=coordTuple,
                               units=units,
                               coordSys=coordSys)

    def log(self): pass

    def zScaledFractionalHeight(self,
                                frame: int,
                                key: str = 'z scaled fractional height (um, rheo_sedHeight)',
                                step: str = None,
                                save2hdf: bool = True):

        """
        Add a column of scaled fractional height to input dataFrame.
        Update the hdf and return the new column(s)
        """
        if step is None: step = self.step
        posDF = self.gel(frame,step=step)
        hAvg = self.gelThickness

        zScaled = pd.Series(
            hAvg * (posDF['z (um, rheo_sedHeight)']) /
            (posDF['z (um, rheo_sedHeight)'] - posDF['dist from sed_gel interface (um, imageStack)']),
            name=key)

        if save2hdf:
            posDF[key] = zScaled
            with tp.PandasHDFStoreBig(self.gelGlobal['path']) as s: s.put(posDF.reset_index())
        return zScaled

    def disp(self,
             cur: int,
             ref=None,
             save2hdf: bool = True,
             forceRecompute: bool = False,
             verbose: bool = False):
        """
        Compute the dispalcement of all gel tracers between current frame and reference frame.
        This should be modified have the option of displacement over global gel deformation.
        """
        if ref is None:
            ref = 'globalRef'
            if verbose: print('Using global reference 0 to compute displacement')

        # create the mapping dictionary
        newCols = ['disp {} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']]
        newCols += ['disp z sed_gel interface (um, imageStack)']
        newCols += ['disp z scaled fractional height (um, rheo_sedHeight)']

        cols = ['{} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']] \
               + ['dist from sed_gel interface (um, imageStack)',
                  'z scaled fractional height (um, rheo_sedHeight)']

        # load the gel positions at the current time for this step
        g = self.gel(cur)

        # set the reference, not ref = None has been mapped to ref = globalRef
        if ref == 'globalRef': refGel = self.globalRefGel
        else: refGel = self.gel(ref)

        # select particles and drop unused columns
        refGel = refGel[refGel[self.keepBool]][cols]
        g = g[g[self.keepBool]][cols]

        # compute the displacement
        disp_gel = (g - refGel).dropna()

        # remap the columns
        # ToDo: add remapping dictionary to input yaml file and rewrite using dict.items() and values() etc.
        old2New = dict(zip(cols, newCols))
        disp_gel.rename(columns = old2New,inplace=True)
        #disp_gel.rename(
        #    columns={'{} (um, rheo_sedHeight)'.format(coord):
        #                 'disp {} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']},
        #    inplace=True)
        #disp_gel.rename(columns={'dist from sed_gel interface (um, imageStack)':
        #                             'disp z sed_gel interface (um, imageStack)'}, inplace=True)
        #disp_gel.rename(columns={'z scaled fractional height (um, rheo_sedHeight)':
        #                             'disp z scaled fractional height (um, rheo_sedHeight)'}, inplace=True)
        if save2hdf:
            # access with local frame number
            posDF = self.gel(cur)

            # attempt to drop new columns in order to overwrite if necessary
            try: posDF.drop(newCols, axis=1, inplace=True)
            except KeyError: pass

            # join old with new columns
            out = posDF.join(disp_gel)

            # write it global with tp knowing what key to put it on with frame column
            with tp.PandasHDFStoreBig(self.gelGlobal['path']) as s: s.put(out.reset_index())

        return disp_gel

    def gelStrain_theilSen(self,
                           coord: str = '(um, rheo_sedHeight)',
                           fractionalHeight: bool = True,
                           add2Inst: bool = True,
                           save2hdf: bool = True,
                           forceRecompute: bool = False):
        """
        Return a dataFrame of fit parameters for all frames and both XZ and YZ using median of slopes of all
        lines of pairwise points: ThielSen robust linear regression

        returns a dictionary with keys xz and yz, and values of dataFrame of fit parameters for each frame

        Optionally saves output, and adds as class attribute and checks if previously computed
        """
        if forceRecompute or not os.path.exists('./theilSen.h5'):
            xz = {}
            yz = {}
            if fractionalHeight:
                keys = dict(dx='disp x ' + coord,
                            dy='disp y ' + coord,
                            z='z scaled fractional height ' + coord)
            else:
                keys = dict(dx='disp x ' + coord,
                            dy='disp y ' + coord,
                            z='z ' + coord)

            for cur in range(self.frames):
                # hmmm disp should update the posDF datastore or create its own new dataStore
                # this should read it hopefully with just the self.gel() method.
                g = self.gel(cur)

                inputDF = {}
                for var, col in keys.items(): inputDF[var] = g[col]
                inputDF = pd.DataFrame(inputDF).dropna()

                m, b, m05, m95 = theilSen(inputDF['dx'], inputDF['z'])
                xz[cur] = dict(m=m, b=b, m05=m05, m95=m95)

                m, b, m05, m95 = theilSen(inputDF['dy'], inputDF['z'])
                yz[cur] = dict(m=m, b=b, m05=m05, m95=m95)

            xz_df = pd.DataFrame(xz).T
            yz_df = pd.DataFrame(yz).T
            out = xz_df.join(yz_df, lsuffix='_xz', rsuffix='_yz')

            if add2Inst: self.theilSen = out
            if save2hdf: out.to_hdf('./theilSen.h5', 'theilSen')
            return out

        else:
            if self.theilSen is not None:
                return self.theilSen
            else:
                out = pd.read_hdf('./theilSen.h5')
                if add2Inst: self.theilSen = out
                return out

    def driftCorrResid(self,
                       frame: int,
                       key_frmt: str = 'disp {} (um, rheo_sedHeight)',
                       save2hdf: bool = True,
                       forceRecompute: bool = False,
                       verbose: bool = False):
        """
        Compute the displacement correct x,y column on displacement dataFrame

        This relies on the following being computed in this order
        1. z scaled fractional height
        2. displacement
        3. fitted deformation
        """
        newCols = ['driftCorr ' + key_frmt.format(coord) for coord in ['x', 'y']]
        newCols += ['residual ' + key_frmt.format(coord) for coord in ['x', 'y']]

        dispDF = self.disp(frame)
        z = self.gel(frame)['z scaled fractional height (um, rheo_sedHeight)']
        driftCorrDF = pd.DataFrame()

        for coord in ['x', 'y']:
            key = key_frmt.format(coord)
            b = self.theilSen.loc[frame]['b_{}z'.format(coord)]
            m = self.theilSen.loc[frame]['m_{}z'.format(coord)]
            driftCorrDF['driftCorr ' + key] = dispDF[key] - b
            driftCorrDF['residual ' + key] = dispDF[key] - (m * z + b)

        if save2hdf:
            posDF = self.gel(frame)
            try: posDF.drop(newCols, axis=1, inplace=True)
            except KeyError: pass
            out = posDF.join(driftCorrDF)
            with tp.PandasHDFStoreBig(self.gelGlobal['path']) as s: s.put(out.reset_index())

        return driftCorrDF




    def xyResiduals_frame(self, cur: int, ref: int=0):
        """
        Return a dtaFrame of just residual differences: x, y, and magnitude for each particle
        """
        #if ref != 0: raise NotImplementedError
        #else:
        # residDF = self.driftCorrResid(cur)[residCols]
        # residDF = posDF[residCols]

        # posDF = self.gel(cur)[posCol]

        ## join
        # out = posDF.join(residDF).dropna()

        # compute z binning

        posDF = self.gel(cur)

        residCols = ['residual disp {} (um, rheo_sedHeight)'.format(coord) for coord in ['x', 'y']]
        posCol = ['{} (um, rheo_sedHeight)'.format(coord) for coord in ['x','y','z']]
        posCol += ['z scaled fractional height (um, rheo_sedHeight)']

        out = posDF[posCol + residCols].dropna()
        out['z scaled bin'] = pd.cut(posDF['z scaled fractional height (um, rheo_sedHeight)'], bins=self.zBins)
        return out



    def xyResiduals_step(self):
        """
        Call xy residuals on each frame and collapse bins to get spatially averaged residuals
        """
        return True

    def stressStrain(self,
                     stressStr: str = 'theilSen_mxz',
                     avgStrainKey: str = 'ref0: mean fl 2e{}z (%)',
                     strainTupleList: list = None,
                     forceRecompute: bool = False):
        """
        Return a dataFrame of stress and avg strain computed in multiple ways index by frame
        with constant reference
        """
        if stressStr != 'theilSen_mxz': raise NotImplementedError
        else:
            if strainTupleList is None: strainTupleList = [(0,frame, 'falkLanger') for frame in range(1,self.frames)]
            out = []
            for coord in ['x','y']:
                stress = pd.Series(self.gelModulus * self.gelStrain_theilSen()['m_{}z'.format(coord)], name='{}z stress (mPa)'.format(coord))
                strain = pd.Series(
                    self.strainInst.avgStrain(idString='ref0',
                                              strainTupleList=strainTupleList,
                                              forceRecompute=forceRecompute)[avgStrainKey.format(coord)],
                    name = avgStrainKey.format(coord))
                out.append(stress)
                out.append(strain)
            return pd.concat(out,axis=1)

if __name__ == '__main__':
    testPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018e'
    param = dict(globalParamFile = '../tfrGel10212018A_globalParam.yml',
                 stepParamFile = './step_param.yml', test=False)
    os.chdir(testPath)
    test = Stress(**param)

    # test displacement
    #print(test.disp(12))


