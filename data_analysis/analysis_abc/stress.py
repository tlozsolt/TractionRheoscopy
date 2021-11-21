import sys
import os
import pandas as pd
import numpy as np

from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis.analysis_abc.dataCleaning import Cleaning
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

        #use the same keepBool and posCoordinateSystem as strain
        self.strain = self.stepParam['strain']
        self.posCoordinateSystem = self.strain['posCoordinateSystem']
        self.keepBool = self.strain['keepBool']


        # rheo calibration
        self.gelModulus = self.rheo['gelModulus']
        self.gelThickness = self.rheo['gelThickness']

        #cache some commonly used time points
        self.refGel = self.gel(0)
        # not clear how to get global gel but this should be added to Analysis abc
        # self.refGelGlobal = self.globalRefGel

        # attributes that will be populated with functions in the class
        self.theilSen = None

    def __call__(self): pass

    def sed(self, frame: int):
        # inherit
        out = super().sed(frame)

        # select
        out = out[out[self.keepBool]]
        return out

    def gel(self, frame: int):
        # inherit
        out = super().gel(frame)
        # select
        out = out[out[self.keepBool]]
        return out

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

    def scaledFractionalHeight(self,
                               posDF: pd.DataFrame,
                               key: str = 'z scaled fractional height (um, rheo_sedHeight)'):
        """
        Add a column of scaled fractional height to input dataFrame.
        Return the augmented dataFrame
        """
        hAvg = self.gelThickness
        posDF[key] = hAvg*(posDF['z (um, rheo_sedHeight)'])/(posDF['z (um, rheo_sedHeight)'] - posDF['dist from sed_gel interface (um, imageStack)'])
        return posDF

    def driftCorr(self,
                  frame: int,
                  key_frmt: str = 'disp {} (um, rheo_sedHeight)'):
        """
        Compute the displacement correct x,y column on displacement dataFrame

        This relies on the following being computed in this order
        1. z scaled fractional height
        2. displacement
        3. fitted deformation
        """
        dispDF = self.disp(frame)
        for coord in ['x','y']:
            key = key_frmt.format(coord)
            b = self.theilSen.loc[frame]['b_{}z'.format(coord)]
            dispDF['driftCorr ' + key] = dispDF[key] - b
        return dispDF

    def gelStrain_theilSen(self,
                           coord: str = '(um, rheo_sedHeight)',
                           fractionHeight: bool = True,
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
            if fractionHeight:
                keys = dict(dx ='disp x ' + coord,
                            dy = 'disp y ' + coord,
                            z = 'z scaled fractional height ' + coord)
            else:
                keys = dict(dx ='disp x ' + coord,
                            dy = 'disp y ' + coord,
                            z = 'z ' + coord)

            for cur in range(self.frames):
                g = self.disp(cur)

                inputDF = {}
                for var,col in keys.items(): inputDF[var] = g[col]

                m,b,m05, m95 = theilSen(inputDF['dx'],inputDF['z'])
                xz[cur] = dict(m=m, b=b, m05=m05, m95=m95)

                m,b,m05, m95 = theilSen(inputDF['dx'],inputDF['z'])
                yz[cur] = dict(m=m, b=b, m05=m05, m95=m95)

            xz_df = pd.DataFrame(xz).T
            yz_df = pd.DataFrame(yz).T
            out = xz_df.join(yz_df,lsuffix='_xz', rsuffix='_yz')

            if add2Inst: self.theilSen = out
            if save2hdf: out.to_hdf('./theilSen.h5', 'theilSen')
            return out

        else:
            if self.theilSen is not None: return self.theilSen
            else:
                out = pd.read_hdf('./theilSen.h5')
                if add2Inst: self.theilSen = out
                return out

    def disp(self,
             cur: int,
             ref: int=0):
        """
        Compute the dispalcement of all gel tracers between current frame and reference frame.
        This should be modified have the option of displacement over global gel deformation.
        """
        if ref == 0: refGel = self.refGel
        else: refGel = self.gel(ref)

        g = self.gel(cur)

        # select particles and drop unused columns
        cols = ['{} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']] \
               + ['dist from sed_gel interface (um, imageStack)']
        refGel = refGel[refGel[self.keepBool]][cols]
        g = g[g[self.keepBool]][cols]

        # compute z scaled fractional height current and reference frames.
        refGel = self.scaledFractionalHeight(refGel)
        g = self.scaledFractionalHeight(g)

        # compute the displacement
        disp_gel = (g - refGel).dropna()

        # remap the column keys in order to join later
        disp_gel.rename(
            columns={'{} (um, rheo_sedHeight)'.format(coord):
                         'disp {} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']},
            inplace=True)
        disp_gel.rename(columns={'dist from sed_gel interface (um, imageStack)':
                                     'disp z sed_gel interface (um, imageStack)'}, inplace=True)
        disp_gel.rename(columns={'z scaled fractional height (um, rheo_sedHeight)':
                                     'disp z scaled fractional height (um, rheo_sedHeight)'}, inplace=True)

        # join pos and disp,
        g = g.join(disp_gel)

        return g.dropna()

    def xyResiduals_frame(self, cur: int, ref: int=0):
        """
        Compute the residual difference in x disp and y disp relative to deoformation profile.
        Return a dtaFrame of just reisudal differences: x, y, and magnitude for each particle
        Compute bin column as well, but do not groupby.
        """
        return True

    def xyResiduals_step(self):
        """
        Call xy residuals on each frame and collapse bins to get sptailly averaged residuals
        """
        return True

    def stressStrain(self):
        """
        Return a dataFrame of stress and avg strain computed in multiple ways index by frame
        with constant reference
        """

if __name__ == '__main__':
    testPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018e'
    param = dict(globalParamFile = '../tfrGel10212018A_globalParam.yml',
                 stepParamFile = './step_param.yml', test=True)
    os.chdir(testPath)
    stress = Stress(**param)

    # test displacement
    print(stress.disp(12))


