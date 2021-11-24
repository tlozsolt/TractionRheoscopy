import sys
import os
import yaml
import pickle as pkl
sys.path.append('/')

from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis.analysis_abc.dataCleaning import Cleaning
from data_analysis import static as da

import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

from multiprocessing import Pool

#import trackpy as tp
#import ovito
#import freud

class Strain(Analysis):

    def __init__(self,globalParamFile, stepParamFile,test=False):
        # load yaml metaData for both expt and step
        super().__init__(globalParamFile=globalParamFile, stepParamFile=stepParamFile)

        if test: self.frames = 3

        # add additional attributes that I will need here
        self.figPath = self.paths['figPath']
        # I dont think I need this as there is no single strain hdf data store, but rather a directory of strain pandas dataframes.
        #self.strainHDF = self.paths['strainHDF']
        self.strainDataDir = self.paths['strainDataDir']

        self.strain = self.stepParam['strain']
        self.verbose = self.strain['verbose']
        self.key_frmt = self.strain['key_frmt']
        self.strainTypes = set(self.strain['strainTypes'])
        self.posCoordinateSystem = self.strain['posCoordinateSystem']
        self.keepBool = self.strain['keepBool']
        self.upperIdx = da.readOvitoIdx(self.paths['topIdx']['init']).intersection(
            da.readOvitoIdx(self.paths['topIdx']['final']))
        self.lowerIdx = da.readOvitoIdx(self.paths['sed_interface_idx'])

        # stress
        self.gelModulus = self.rheo['gelModulus']
        self.gelThickness = self.rheo['gelThickness']

    def __call__(self): pass

    def sed(self, frame: int):
        """ get sed positions that are used for strain computation"""
        # the frame using pytables
        out = super().sed(frame)

        # select on keep bool
        out = out[out[self.keepBool]]

        #return only the position columns
        posList = self.posList(**self.posCoordinateSystem)
        return out[posList]

    def gel(self, frame: int):
        """ get gel positions that are used for strain computation"""
        # dont know if this is the right syntax
        # to assign output of inherited method to local variables
        out = super().gel(frame)
        out = out[out[self.keepBool]]
        posList = self.posList(**self.posCoordinateSystem)
        return out[posList]

    def setPlotStyle(self):
        super().setPlotStyle()
        # I will likely need to add other figure modifications here

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

    def log(self):
        super(Strain, self).log()
        # i am suprised ptCharm autofilled this syntax.
        # I would have typed super().log() but maybe this works
        # and makes sense if I had inheritances from multiple abstract base classes

    def gap(self,frame: int):
        """ Compute a return the gap height as yet undetermined algorightm
            Currently just return a fixed value of 55 um

            upperHeight = 55.36
            lowerHeight = 0
            return upperHeight - lowerHeight

            return a dict of all three:
            - min distance: place a plane with z normal at highest point on lower plane and lowest point on upper plane
                            and compute the separation between the planes
            - max distance: same thing as above but with the conditions reversed
            - mean: fit the planes and report the distance between mid points
        """

        # load the positions and get particle index
        pos = self.sed(frame)
        idx = pos.index

        uIdx = self.upperIdx.intersection(idx)
        lIdx = self.lowerIdx.intersection(idx)

        # now get equations for each of these planes
        u = da.fitSurface_singleTime(pos,uIdx,'(um, rheo_sedHeight)')
        l = da.fitSurface_singleTime(pos,lIdx,'(um, rheo_sedHeight)')

        # eval the plane at 5 points: 4 corners and 0
        def plane(x,y,a,b,c, **kwargs): return a*x + b*y + c
        def uPlane(pt):
            x,y = pt[0], pt[1]
            return plane(x,y,**u)
        def lPlane(pt):
            x,y = pt[0], pt[1]
            return plane(x,y,**l)

        a = 235/2 # distance of side length in microns...spatial extent (-a,a) with center at 0 in (um, rheoSedHeight)
        ptList = [(a,a),(-a,a),(-a,-a),(a,-a),(0,0)]
        uPts = list(map(uPlane,ptList))
        lPts = list(map(lPlane,ptList))
        uDict = {'min': min(uPts), 'max': max(uPts), 'mean': uPts[-1]}
        lDict = {'min': min(lPts), 'max': max(lPts), 'mean': lPts[-1]}

        return {'min': uDict['min'] - lDict['max'],
                'max': uDict['max'] - lDict['min'],
                'mean': uDict['mean'] - lDict['mean']}

    def _key(self, ref: int, cur:int, strainType: str):
        if strainType in self.strainTypes and cur >= ref:
            return self.key_frmt.format(ref=ref,cur=cur,strainType=strainType)
        else: raise KeyError('Not supported strainType {} or cur ({}) < ref ({})?'.format(strainType,ref,cur))

    def _strainPath(self,ref:int, cur:int, strainType:str):
        """ Return the path to the (ref,cur,strainType) hdf, regardless of whether the actual file exists"""
        return self.strainDataDir + self._key(ref,cur,strainType) + '.h5'

    def writeStrain(self, strainDF: pd.DataFrame,
                    ref:int, cur:int, strainType: str,
                    replicate:int = 0):
        """
        This will add a key to strainHDF that is formatted in a way to reference both ref and cur
        and support queries

        key structure is 'ref{ref:03}/cur{cur:03}/{strainType}'.format(ref,cur, strainType)
        for example strain measured between frames 3 and 7 using ovito would be 'ref003/cur007/atomicStrainOvito'

        cur > ref and there should be exactly three possible strainTypes:
        atomicStrainOvito, falkLanger, and boundary

        I think dictionary of params should also be written to this hdfStore under param key

        >> https://glowingpython.blogspot.com/2014/08/quick-hdf5-with-pandas.html

        Multiprocessing write to single h5 file using separate worker for queue:
        >> https://stackoverflow.com/questions/15704010/write-data-to-hdf-file-using-multiprocessing

        UPDATE: one way to solve the problem of multiprocessing writing to one hdf file is have each job write
                to its own hdf file, and then, at the end of the mulitprocessing queueu add soft links to
                separate output files
                Note the key structure for hdf files is already formatted as a path: ref003/cur007/atomicStrainOvito
                so just prepend a './strain_data/' and append '.h5' to get the relative path of the file that needs
                to be added a soft link to './strain.h5'
                >>
        UPDATE 2: Nope, fuck this. Just put the files in the directory structure like './strainData/ref000/cur002/falkLanger.h5
                  using a pandas write hdf and then overload this access functions sed() or maybe strain() to load that
                  file into memory. Its not worth the overhead to deal with pyTables or anything for such easily atomized
                  data structure. As a bonus its obvious where to put paramFiles and how to test if the
                  file 1) exists 2) should be computed or 3)recomputed all based on whether the file exists and whether
                  the strain function parameters are identical to the param file saved at time of writing.
                  Q: I wonder if this can be stored on the kaposzta? A?: I think it can be stored on kaposzta, no problem
                  At the end of the day I dont think this is radically different than the motivation and structure of
                  hdf, it just uses the os filesystem and a homebrew accessing and retrieval function
                  param file should be stored at './strainData/param.yml'
                  It would also be useful to have strainData_index where index doesnt really mean anything other than
                  the paramFiles are different and the sybmolic link in $COLLOID/DATA/{expt}/{step}/strainData points to
                  what whatever 'strainData_index' on kaposzta I want. Very transparent as to what is going and
                  what parameters I am working with at the time
        """
        try: os.mkdir('{}/'.format(self.strainDataDir))
        except FileExistsError: pass

        try: os.mkdir('{}/ref{:03}'.format(self.strainDataDir, ref))
        except FileExistsError: pass

        try: os.mkdir('{}/ref{:03}/cur{:03}'.format(self.strainDataDir,ref, cur))
        except FileExistsError: pass

        hdfPath = self._strainPath(ref,cur,strainType)

        # i think if key exists already this will overwrite?
        # not sure how to handle the cases:
        #   (i) file doesnt exist and needs to be created
        #   (ii) file exists but I dont know what keys are there...how to determine if key already exists
        #   (iii) file exists but I need to overwrite the key? overwrite the file?
        # Do I really need to handle all these cases? Probably not.
        # I am just going to pass for now but keep in mind that replicate 0 is reserved for the spot estimate.
        strainDF.to_hdf(hdfPath, str(replicate), mode='a')


    def getStrain(self, strainType: str, ref: int=-1, cur: int=-1, replicate: int = 0):
        """
        This will print the number of atlantic city trials and return the strain the strain obtained w/o atlantic city
        and return None if the strain wasnt computed yet
        """
        if strainType =='avgStrain':
            strainData = self.strainDataDir +'/frameAverage.h5'
            return pd.read_hdf(strainData)

        else:
            strainData = self.strainDataDir + self._key(ref,cur,strainType) +'.h5'
            with pd.HDFStore(strainData) as s:
                if self.verbose: print(s.keys())
                out = s.get(str(replicate))
            return out

    def FL(self, ref: int, cur:int, replicate: int = 0):
        return self.getStrain('falkLanger', ref, cur, replicate)

    def boundary(self, ref:int ,cur: int): return self.getStrain('boundary', ref, cur)

    def computeStrain(self, strainCall: tuple,
                      forceRecompute: bool=False):
        """
        strainQueue is a list of tuple (ref:int, cur:int, strainType:str)
        that is unpacked and dispatched to the correct compute function
        The output is written to hdf file using a file structure of
        ./strainData/ref{ref:03}/cur{cur:03}/{strainType}.h5

        #https://stackoverflow.com/questions/15704010/write-data-to-hdf-file-using-multiprocessing
        """
        # I need a function that operates on a single elt of strainQueue which can be mapped over strainQueue using Pool map

        # unpack the tuple
        ref,cur,strainType = strainCall

        # should I compute or recompute?
        if os.path.exists(self._strainPath(ref, cur, strainType)) and forceRecompute == False:
            if self.verbose: print('already computed, passing')
            return True
        else:
            if self.verbose: print('Computing ref = {}, cur = {}, strainType is {}'.format(ref,cur,strainType))
            if strainType == 'falkLanger':
                strainDF = self.compute_falkLanger(ref,cur, **self.strain['falkLanger'])
            elif strainType == 'atomicStrain':
                strainDF = self.compute_atomicStrainOvito(ref,cur, **self.strain['atomicStrainOvito'])
            elif strainType == 'boundary':
                strainDF = self.compute_boundary(ref,cur, **self.strain['boundary'])

            self.writeStrain(strainDF,ref,cur,strainType)
            return True

    def computeStrainList(self, queue:list, forceRecompute: bool = False):
        """
        This is hopeless... I am running into locking errors in **loading** sed_stitched
        On the plus side each strain comp only takes a minute so I dont really need parallelization
        Additionally, it should be no problem to parallelize a single strain combo and dispatch worker
        for each atlantic city trial. There is jsut one call to hdf to get ref and cur positions, then ten workers
        deal with resampling positions and computing the strain on the resampled data. Even without parallelization
        a full expt of 300 frames and 10 atlantic city trials would only take 2-3 days.

        This will test that cur>ref by internal call to self._key() and whether it has been computed by passing
        forceRecompute
        """
        for strainCall in queue:
            self.computeStrain(strainCall, forceRecompute)
        return True

    def compute_atomicStrainOvito(self, ref: int, cur: int, cutoff: float=2.2, **kwargs) -> pd.DataFrame:
        """
        Computes the atomic strain using ovito atomic strain setting the reference frame at ref and
        current frame as cur

        If some of the kwargs are not identical to the param in hdf, then raise an error or write to a new hdf
        """
        # load the configurations ref, cur from pytables
        # select the columns
        # get the particle index
        # pass the values as numpy array to ovito
        # compute strain components, D2min, von Mises, rotation, eigenvectors(u,v,w), eigenValues (U,V,W)
        # save results to pyTables
        return True

    def compute_falkLanger(self, ref: int, cur: int,
                           nnb_cutoff: float=2.2,
                           verbose: bool= False, **kwargs) -> dict:
        """
        compute the strain using Falk-Langer and homebrew algorithm
        Same output as atomicStrain_ovito
        This should more or less be done with calls to da.static
        however, the wrapper for computeLocalStrain assumes the old multiindex of frame,particle id
        I am just going to refactor da.localStrain, but not modify da.computeLocalStrain

        Additionally, add option to compute local strain with fixed number of nearest neighbors and report center rg
        of particle with nnb shell.
        """
        p = {'falkLanger': {'nnb_cutoff': nnb_cutoff}}

        # get dataFrame of particle positions with index of particle id for the ref and cur configurations
        # these should be cleaned and in um, rheoSedHeight by default
        refConfig = self.sed(ref)
        curConfig = self.sed(cur)

        # da.computeLocalStrain assumes input numpy array [x,y,z]
        if self.posCoordinateSystem['coordTuple'] != ['x','y','z']:
            if verbose: print('Remapping position columns to xyz')
            xyzCoordSys = self.posCoordinateSystem # make a copy of everything in coordinate system
            xyzCoordSys['coordTuple'] = ['x','y','z'] #overwrite coordTuple key
            posList = self.posList(**xyzCoordSys) #generate correctly order xyz list

            #reorder columns in posDF before coverting to numpy
            refConfig = refConfig[posList]
            curConfig = curConfig[posList]

        # filter to get only complete trajectories between time points and make np input arrays.
        idx = curConfig.index.intersection(refConfig.index)
        refConfig = refConfig.loc[idx].to_numpy()
        curConfig = curConfig.loc[idx].to_numpy()

        # generate search tree to query neighbors on reference config.
        if verbose: print('generating search tree')
        refTree = cKDTree(refConfig)

        # query tree with all points to nnb cutoff set to default of 1st min in RDF
        if verbose: print('querying tree for nnb')
        nnbIdx = refTree.query_ball_point(refConfig, nnb_cutoff)

        # let's keep track of the number of nnb for each particle id
        nnb_count = pd.Series(np.array([len(nnbList) for nnbList in nnbIdx]), index=idx, name='nnb count')
        max_nnb = nnb_count.max()

        def padN(l, val, N): return np.pad(np.array(l), (0, N), mode='constant', constant_values=val)[0:N]

        # see caution warning da.localStrain
        nnbIdx_np = np.array([padN(nnbIdx[m], m, N=max_nnb + 1) for m in range(len(nnbIdx))])

        if verbose: print('computing local strain')
        localStrainArray_np = da.computeLocalStrain(refConfig, curConfig, nnbIdx_np)
        localStrainArray_df = pd.DataFrame(localStrainArray_np, columns=['D2_min', 'vonMises',
                                                                         'exx', 'exy', 'exz', 'eyy', 'eyz', 'ezz',
                                                                         'rxy', 'rxz', 'ryz'], index=idx).join(nnb_count)

        return localStrainArray_df

    def compute_Bagi(self,ref: int, cur: int, verbose: bool = False):
        """
        Compute the strain according to Bagi:
        https://doi.org/10.1016/j.ijsolstr.2005.07.016
        Also:
        Bagi, K. (1996). Stress and strain in granular assemblies. Mechanics of materials, 22(3), 165-177.

        """
        # compute particle displacements from ref to cur
        # compute voronoi tesselation on particles in ref
        # for each tetrahedron compute the relative displacements of edge i and
        return True


    def fitTopSurface(self): return True


    def compute_boundary(self, ref: int, cur: int, gapMethod: str = 'min', **kwargs) -> pd.DataFrame:
        """
        Compute the strain by computing the relative displacement of the boundary.
        Output dataFrame should be and filled with nan if particle is not present in both ref and cur
        (equiv to absent on either)

        particle id (index), dx (um) dy (um) dz (um), coord_str, mat, upper/lower flag

        on this dataFrame, a simple call in pandas should compute the relative displacment
        df[df['upper'] == 'upper'].mean() - df[df['lower'] == 'lower'].mean()

        and still leave the possbility of computing spatial (xy dependence later)

        fuck this thing is slow considering how simple a computation this is

        ToDo:
          - loop over current times
          - generate figure
          - deal with offset in time for comparison to avg local strain
               >> maybe it would be worth shifting at the difference step? curTop - refTop
          - decide on output, save particle positions in addition to aggregate data
          - format output depending on stepParam entry
          - save output with correct formatting, labels, pdf output etc.
               >> add some of these to setplotstyle() method.
          - generate a subplot to show residual difference of the two strain measures...prl style.
          - One variation is to take volume average only over the particles included in the region bounded
            by the boundary (DUH) This should remove high strain particles taht are near the grid in FL thereby
            lowering the average strain, which goes in the same direction as scaling the boundary strain by ~5%
        """

        # compute the gap in the reference frame before remapping variables ref and cur to dataframes
        gapDict = self.gap(ref)
        refN, curN = ref,cur
        # load ref and cur
        # note ref and cur are now dataFrames, not int as the call signature shows.
        # note this defaults to the col keys in self.sed()
        ref = self.sed(ref)
        cur = self.sed(cur)

        # get particle ids for lower and upper boundary
        #    upper
        refUpperIdx = ref.index.intersection(self.upperIdx)
        curUpperIdx = cur.index.intersection(self.upperIdx)
        #    lower
        refLowerIdx = ref.index.intersection(self.lowerIdx)
        curLowerIdx = cur.index.intersection(self.lowerIdx)

        # aggregate over particle index and compute displacement
        # is this the same as compute the displacement per particle then averaging? Yes, assuming the
        # particle ids are all the same...simple proof by induction
        dispUpper = cur.loc[curUpperIdx].mean() - ref.loc[refUpperIdx].mean()
        dispLower = cur.loc[curLowerIdx].mean() - ref.loc[refLowerIdx].mean()
        relDisp = dispUpper - dispLower

        # rename the column labels
        rename = {'{} (um, rheo_sedHeight)'.format(coord):
                      '<{}Upper> - <{}Lower> (um, rheo_sedHeight)'.format(coord,coord) for coord in ['x','y','z']}
        relDisp = pd.DataFrame(relDisp).T
        relDisp.rename(columns=rename, inplace=True)

        # add the displacement of upper and lower boundaries
        disp_frmt = '<{}{}> (um, rheo_SedHeight)'
        for coord in ['x','y','z']:
            # add the upper and lower boundary displacments to relDisp
            relDisp[disp_frmt.format(coord,'Lower' )] = dispLower['{} (um, rheo_sedHeight)'.format(coord)]
            relDisp[disp_frmt.format(coord,'Upper' )] = dispUpper['{} (um, rheo_sedHeight)'.format(coord)]

        # compute the strain gamma (simple, divide by gap)
        for method, gap in gapDict.items():
            for coord in ['x','y','z']:
                relDisp['strain 2e{}z, gap {}'.format(coord,method)] = \
                    relDisp['<{}Upper> - <{}Lower> (um, rheo_sedHeight)'.format(coord,coord)]/gap
                relDisp['gap ref {} (um, rheo_sedHeight)'.format(method)] = gap
                #relDisp['strain 2e{}z, gap {}'.format(method)] = relDisp['y (um, rheo_sedHeight)']/gap
                #relDisp['strain 2e{}z, gap {}'.format(method)] = relDisp['z (um, rheo_sedHeight)']/gap

        self.writeStrain(relDisp,refN,curN,'boundary')
        # append output to pyTable with index of tuple
        # return aggregate statistic
        return relDisp

    def avgStrain(self, forceRecompute: bool = False, save2hdf: bool = True):
        """
        Compute strain vs time with a fixed reference configuration at time t=0
        Should return a dataFrame that can be passed to seaborn and plotted
        """
        outPath = self.strainDataDir +'/frameAverage.h5'
        if forceRecompute is False and os.path.exists(outPath):
            return pd.read_hdf(outPath,'avgStrain')
        else:
            avgStrain_dict = {}
            for cur in range(1,self.frames):
                #strainDF = self.getStrain(0,cur,strainType)
                #avgStrain[cur] = strainDF[strainDF['nnb count'] > 9].mean()

                fl_ref = self.getStrain('falkLanger', 0, cur)
                fl_dt1 = self.getStrain('falkLanger', cur - 1, cur)
                bd_ref = self.getStrain('boundary', 0, cur)
                bd_dt1 = self.getStrain('boundary', cur - 1, cur)

                # apply selection criterion to remove large strain due to low nnb count
                refIdx = fl_ref[fl_ref['nnb count'] >= 9].index
                dt1Idx = fl_dt1[fl_dt1['nnb count'] >= 9].index

                # select if applicable and apply mean
                fl_ref = fl_ref.loc[refIdx].mean()
                fl_dt1 = fl_dt1.loc[dt1Idx].mean()
                bd_ref = bd_ref.mean()
                bd_dt1 = bd_dt1.mean()

                # now add output keys (this part should be incorporated into yaml file perhaps?
                avgStrain_dict[cur] = {
                    'Ref: mean fl 2exz (%)' : 200*fl_ref['exz'],
                    'Ref: mean fl 2eyz (%)' : 200*fl_ref['eyz'],
                    'Ref: mean fl 2ezz (%)' : 200*fl_ref['ezz'],
                    'Ref: mean fl vM' : fl_ref['vonMises'],
                    'Ref: boundary gap min mean gamma (%)' : 100*bd_ref['strain 2exz, gap min'],
                    'Ref: boundary gap mean mean gamma (%)' : 100*bd_ref['strain 2exz, gap mean'],
                    'Ref: boundary gap max mean gamma (%)' : 100*bd_ref['strain 2exz, gap max'],
                    'Ref: x displacement upper boundary (um)': bd_ref['<xUpper> (um, rheo_SedHeight)'],
                    'Ref: y displacement upper boundary (um)': bd_ref['<yUpper> (um, rheo_SedHeight)'],
                    'Ref: z displacement upper boundary (um)': bd_ref['<zUpper> (um, rheo_SedHeight)'],
                    'Ref: x displacement lower boundary (um)': bd_ref['<xLower> (um, rheo_SedHeight)'],
                    'Ref: y displacement lower boundary (um)': bd_ref['<yLower> (um, rheo_SedHeight)'],
                    'Ref: z displacement lower boundary (um)': bd_ref['<zLower> (um, rheo_SedHeight)'],
                    #'Ref: stress xz (mPa)': self.gelModulus*bd_ref['<xLower> (um, rheo_SedHeight)']/self.gelThickness,
                    #'Ref: stress yz (mPa)': self.gelModulus*bd_ref['<yLower> (um, rheo_SedHeight)']/self.gelThickness,
                    #'Ref: stress zz (mPa)': self.gelModulus*bd_ref['<zLower> (um, rheo_SedHeight)']/self.gelThickness,
                    'Ref: residual mean fl - boundary min (%)' : 200 * fl_ref['exz'] - 100 * bd_ref['strain 2exz, gap min'],
                    'Ref: residual mean fl - boundary mean (%)': 200 * fl_ref['exz'] - 100 * bd_ref['strain 2exz, gap mean'],
                    'Ref: residual mean fl - boundary max (%)' : 200 * fl_ref['exz'] - 100 * bd_ref['strain 2exz, gap max'],
                    'dt1: mean fl 2exz (%)' : 200 * fl_dt1['exz'],
                    'dt1: mean fl 2eyz (%)' : 200 * fl_dt1['eyz'],
                    'dt1: mean fl 2ezz (%)' : 200 * fl_dt1['ezz'],
                    'dt1: mean fl vM': fl_dt1['vonMises'],
                    'dt1: boundary gap min mean gamma (%)' : 100 * bd_dt1['strain 2exz, gap min'],
                    'dt1: boundary gap mean mean gamma (%)': 100 * bd_dt1['strain 2exz, gap mean'],
                    'dt1: boundary gap max mean gamma (%)' : 100 * bd_dt1['strain 2exz, gap max'],
                    'dt1: x displacement upper boundary (um)': bd_dt1['<xUpper> (um, rheo_SedHeight)'],
                    'dt1: y displacement upper boundary (um)': bd_dt1['<yUpper> (um, rheo_SedHeight)'],
                    'dt1: z displacement upper boundary (um)': bd_dt1['<zUpper> (um, rheo_SedHeight)'],
                    'dt1: x displacement lower boundary (um)': bd_dt1['<xLower> (um, rheo_SedHeight)'],
                    'dt1: y displacement lower boundary (um)': bd_dt1['<yLower> (um, rheo_SedHeight)'],
                    'dt1: z displacement lower boundary (um)': bd_dt1['<zLower> (um, rheo_SedHeight)'],
                    #'dt1: stress xz (mPa)':  self.gelModulus * bd_dt1['<xLower> (um, rheo_SedHeight)']/self.gelThickness,
                    #'dt1: stress yz (mPa)':  self.gelModulus * bd_dt1['<yLower> (um, rheo_SedHeight)']/self.gelThickness,
                    #'dt1: stress zz (mPa)':  self.gelModulus * bd_dt1['<zLower> (um, rheo_SedHeight)']/self.gelThickness,
                    'dt1: residual mean fl - boundary min (%)' : 200 * fl_dt1['exz'] - 100 * bd_dt1['strain 2exz, gap min'],
                    'dt1: residual mean fl - boundary mean (%)': 200 * fl_dt1['exz'] - 100 * bd_dt1['strain 2exz, gap mean'],
                    'dt1: residual mean fl - boundary max (%)' : 200 * fl_dt1['exz'] - 100 * bd_dt1['strain 2exz, gap max']
                }
            out = pd.DataFrame(avgStrain_dict).T
            if save2hdf: out.to_hdf(self.strainDataDir +'/frameAverage.h5', 'avgStrain',mode='a')
            return out

    def     compute_zBinDisp(self, frame: int, gelBins: int = 6, sedBins: int = 10, **kwargs):
            """ ### Boundary Slip ### """
            # Particle counts in gel per xy bin
            # ability to segment gel and sediment (bimodal histogram)
            # Displacement in shear direction vs. height (diverging color map with horizontal bins)
            # make color palette

            # ToDo:
            #  - turn into a function that loops over frames
            #  - Find a way of incorporating the number of samples in each cut into the plot
            #  - Change the logic so that the bin spacing is always the same across sed and gel so
            #    so that slopes on gel and sed deformation are easy to compare without computing.
            #    Not sure how to do this...maybe label material, concat, and then bin on both mat and dist from
            #    interface?

            # get the top surface height in um, rheo_sedHeight
            c_ref = da.fitSurface_singleTime(self.sed(0), self.upperIdx, '(um, rheo_sedHeight)')['c']
            c_cur = da.fitSurface_singleTime(self.sed(frame), self.upperIdx, '(um, rheo_sedHeight)')['c']

            # get the positions and compute displacement
            # .   sed, keys x,y position and distance from bottom of sediment...this likely requires sed query on clean instance
            clean = Cleaning(**self.abcParam)
            ref = clean.sed(0)
            ref = ref[ref['cleanSedGel_keepBool']][['{} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']]
                                                   + ['dist from sed_gel interface (um, imageStack)']]
            s = clean.sed(frame)
            s = s[s['cleanSedGel_keepBool']][['{} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']]
                                             + ['dist from sed_gel interface (um, imageStack)']]

            # only include particles that are below the grid (approx)
            ref = ref[ref['z (um, rheo_sedHeight)'] < c_ref]
            s = s[s['z (um, rheo_sedHeight)'] < c_cur]

            disp = (s - ref).dropna()
            disp.rename(
                columns={'{} (um, rheo_sedHeight)'.format(coord):
                             'disp {} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']},
                inplace=True)
            disp.rename(columns={'dist from sed_gel interface (um, imageStack)':
                                     'disp z sed_gel interface (um, imageStack)'}, inplace=True)

            # .  gel, keys x,y position and distance from bottom of sediment
            refGel = clean.gel(0)
            refGel = refGel[refGel['cleanSedGel_keepBool']][
                ['{} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']]
                + ['dist from sed_gel interface (um, imageStack)']]
            g = clean.gel(frame)
            g = g[g['cleanSedGel_keepBool']][['{} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']]
                                             + ['dist from sed_gel interface (um, imageStack)']]
            disp_gel = (g - refGel).dropna()
            disp_gel.rename(
                columns={'{} (um, rheo_sedHeight)'.format(coord):
                             'disp {} (um, rheo_sedHeight)'.format(coord) for coord in ['z', 'y', 'x']},
                inplace=True)
            disp_gel.rename(columns={'dist from sed_gel interface (um, imageStack)':
                                         'disp z sed_gel interface (um, imageStack)'}, inplace=True)

            # join pos and disp, concat, z- bin and aggregate
            interface_dist = 'dist from sed_gel interface (um, imageStack)'

            g = g.join(disp_gel)
            g['bin bottom sed'] = pd.cut(g[interface_dist], gelBins)
            g['bin mid'] = g['bin bottom sed'].map(lambda x: round((x.left + x.right) / 2, 1))

            s = s.join(disp)
            s['bin bottom sed'] = pd.cut(s[interface_dist], sedBins)
            s['bin mid'] = s['bin bottom sed'].map(lambda x: round((x.left + x.right) / 2, 1))

            # assign material str
            g['mat'] = 'gel'
            s['mat'] = 'sed'

            # concatenate and reset index
            tmp = pd.concat([s, g]).reset_index()
            return tmp

    def plots(self):
        """
        Makes the following plots read from yaml plot file that has paths to saved datasets that are passed to seaborn
        for each category of plot and indexed with short metaData captions listed in the yaml file.
        This file doesnt have to be separate for each step. The paths to datasets can be formattable strings or relative
        paths once the working directory is set up.

        The idea is that once this directory and plot list are set up, I could chdir to each step, call this function
        and have the plots for each step plotted. One call --> hundreds of plots all labeled correctly.
        And if I add something to the directory, it wont recompute...maybe set up a flagging system or check if file
        exists. Different variations that I havnnt included need to be added as separate keys or subkeys and filenames.

        Strain vs time
        --------------
        input: dataFrame of avg strains
        compute avg strain dict and then generate plots
        - ref = 0: Volume avg FL strain xz over whole sample vs frame
        - ref = 0: Volume avg FL strain xz over gap min region
        - ref = 0: boundary strain with min, max, and mean gap vs frame

        - ref = 0: boundary strain with min gap and vol avg FL strain over min gap region comparison
        - ref = 0: Residual of boundary strain with min gap and vol avg FL strain over min gap region comparison
        - ref = 0: boundary strain with scaled by multiplicative factor and vol avg FL strain over min gap region comparison
        - ref = 0: Residual of boundary strain with scaled by multiplicative factor and vol avg FL strain over min gap region comparison


        +++ some kind of time correlation analysis for different strain components and/or D2 min and vM ***

        Strain distributions
        --------------------
        input: strain for each particle at each time
        - ref = 0: PDF of strain distribution exz
        - ref = 0: PDF of vM strain distribution
        - ref = cur -1: PDF of strain distribution exz
        - ref = cur -1: PDF of vM strain distribution
        - ref = cur -1: PDF of vM with some kind of cutoff or additional overlay of what
                        noise distribution from atlantic city sampling is.
        - ref = cur - 1: Volume avg strain time overlay (maybe every 5th frame?)
        - ref = cur - 1: Volume avg strain in two different deformation regimes


        Stress vs strain
        ----------------
        input: gel tracer
        - ref = 0: stress vs volume avg FL strain on loading
        - ref = 0: stress vs volume avg FL strain on unloading
        - ref = 0: stress vs volume avg FL strain on full cycle

        - ref = 0: stress vs boundary strain with min gap on loading
        - ref = 0: stress vs boundary strain with min gap on unloading
        - ref = 0: stress vs boundary strain with min gap on full cycle

        - ref = cur -1: absolute stress measurement vs. volume avg FL strain
                        -> does the rate of STZ activation increase with stress?


        Boundary Slip
        -------------
        - ref = 0: displacements binned in

        Inclusions
        ----------
        - ref = cur -1: number of particles with crtically large strain vs. frame
        - ref = cur -1: number of cluster of at least 12 particles with critcially large avg strain vs frame

        Orientation
        -----------
        - polar plot of orientation
        """
        return True


if __name__ =='__main__':
    os.chdir('/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018e')
    paramPath = {'globalParamFile': '../tfrGel10212018A_globalParam.yml',
                 'stepParamFile': './step_param.yml'}
    strain = Strain(**paramPath)

    strain.compute_boundary(0,1)



