from particleLocating import dplHash_v2 as dpl
import pandas as pd
from joblib import Parallel, delayed
import glob
import trackpy
import numpy as np
from scipy.spatial import cKDTree
import numba
from numba import prange
import trackpy as tp
from tqdm import tqdm


class ParticleStitch(dpl.dplHash):
    def __init__(self,metaDataPath,computer='IMAC', lsqCentroid = 'lsq'):
        self.dpl = dpl.dplHash(metaDataPath)
        self.computer= computer
        #self.locations = pd.DataFrame(columns = ['z (px, hash)', 'y (px, hash)', 'x (px, hash)',\
        #                                         'dz (px)', 'dy (px)', 'dx (px)',\
        #                                          'image mass', 'hashValue', 'material', 'n_iteration', 'cost',\
        #                                          'z (um)', 'y (um)', 'x (um)'\
        #                                          ]\
        #                              )
        self.locations = pd.DataFrame()
        self.lsqCentroid = lsqCentroid

    @staticmethod
    def csv2DataFrame(path, hv, mat, frame, sep=' '):
        """
        Read the csv file to a dataFrame and populate some additional columns

        Will add default behavior to just skip if path doesn't exist. Will return empty dataFrame
        which could be added to concat in addHashValue
        Zsolt, Oct 28 2021

        Will add default behavior to typecast empty csv files. from global param file.
        add dtype dict keyword parameter
        >> dict(allKeys.dtypes[set([x for x in subset.dtypes.keys()]).intersection(allKeys)])
        will return a dictionary that can be passed to dtype recasting in pandas.
        Zsolt Oct 31 2021

        I decided not to implement either fo the two changes listed above as it is easier, for the moment
        to just coerce to the correct dtype just prior to the point where writing object dtypes to pyTables is
        problematic.
        Oct 31 2021
        """
        df = pd.read_csv(path, sep=sep)
        df['hashValue'] = int(hv)
        df['material'] = str(mat)
        df['frame'] = int(frame)
        return df.rename(columns = {'z': 'z (px, hash)',
                                    'y': 'y (px, hash)',
                                    'x': 'x (px, hash)'})

    def addHashValue(self,hv,recenterDict = {'flag': True, 'coordStr':'(um, imageStack)'}, path=None):
        """
        for a given hashValue, will add append the locations to the self.locations and add column
        specifying the hashValue and material string with no other modification
        :return:
        """
        if self.lsqCentroid == 'lsq': pxLocationExtension = '_' + self.dpl.sedOrGel(hv) + "_trackPy_lsqRefine.csv"
        elif self.lsqCentroid == 'centroid': pxLocationExtension = '_' + self.dpl.sedOrGel(hv) + "_trackPy.csv"
        if path is None:
            path = self.dpl.getPath2File(hv, kwrd='locations', extension=pxLocationExtension, computer=self.computer)
        else:
            fName = self.dpl.getPath2File(hv, kwrd='locations', extension=pxLocationExtension, computer=self.computer, fileNameOnlyBool=True)
            path +=  fName
        mat = self.dpl.sedOrGel(hashValue=hv)
        frame = int(self.dpl.queryHash(hv)['index'][-1])

        df = self.csv2DataFrame(path, hv, mat, frame)
        if recenterDict['flag'] == True:
            df = self.recenterHash(df,hv,coordStr=recenterDict['coordStr'])
        self.locations = pd.concat([self.locations,df],ignore_index=True)
        return True

    def recenterHash(self,df,hv, coordStr = '(um, imageStack)'):
        """
        This function operates on a dataFrame of particle locations
        Takes particle locations from hashValue and, depending on the material
        transforms the pixel coordinates to coordinates in a choice of abosolute reference coordinates
        specified by :coordStr :
           '(um, rheo_sedHeight)'
               - gel: xy are centered on the image, and z is the height of the particle in the gel
                      in the full reference stack...ie z is height above coverslip
               - sed: xy are centered on the image, and z is the height above the top surface of gel.
           '(um, rheo_sedDepth)'
               - gel: xy are centered on the image, and z is the height of the particle in the gel
                      in the full reference stack...ie z is height above coverslip
               - sed: xy are centered on the image, and z is the depth below the shear plate.
            '(px, imageStack)'
               - gel: xy are image coordinates with top left corner as origin and
                      z is pixel height in image stack
               - sed: same coordinate system as gel.This is a straightforward transformation
            '(um, imageStack)'
               - same as '(px, imageStack)' but units are um
            '(px, hash)'
               - raw output coordinates of locating applied for hashed image. This is image coordinates
                 (ie xy is top left, and x is fastest axis) and **all** hash locations will have
                 different origins in the reference frame.
            'all'
                - do all the coord strings
        The coordStr is a suffix on the axis label for the output dataFrame
        The location of the gel is specfied in the yaml file and it is constant. Any tilt will be
        dealt with later
        :return: dataFrame with additional columns
        """
        # load some paramters
        computer = self.computer
        gelTopRef = self.dpl.metaData['imageParam']['gelSedimentLocation_fullStack']
        gelTopImageStack = self.dpl.metaData['imageParam']['gelSedimentLocation']
        shearPostImageStack = self.dpl.metaData['imageParam']['shearPostLocation']
        px2MicronDict = self.dpl.metaData['imageParam']['px2Micron']
        if self.dpl.metaData['postDecon']['upScaling']['bool'] == True:
            # we upscaled the image and so the output from particle locations are in
            # different pixel unit than the raw image
            upScaleDict = self.dpl.metaData['postDecon']['upScaling']['dim']
            px2MicronDict['x'] = px2MicronDict['x']/upScaleDict['x']
            px2MicronDict['y'] = px2MicronDict['y']/upScaleDict['y']

        imgDim = {'x': self.dpl.metaData['imageParam']['xDim'],\
                  'y': self.dpl.metaData['imageParam']['yDim'], \
                  'z': self.dpl.metaData['imageParam']['zDim']}
        # This path should be paired to the location and log directories
        origin,dim = self.dpl.integrateTransVect(hv,computer=computer)

        # add a step to rename col for centroid locations
        if self.lsqCentroid == 'centroid':
            for c in ['z','y','x']:
                coord = '{}_centroid (px)'.format(c)
                df['{} (px, hash)'.format(c)] = df['{}'.format(coord)]

        if coordStr =='(px, imageStack)':
            df['x '+ coordStr] = df['x (px, hash)'] + origin[0]
            df['y '+ coordStr] = df['y (px, hash)'] + origin[1]
            df['z '+ coordStr] = df['z (px, hash)'] + origin[2]
        elif coordStr == '(um, imageStack)':
            df['x '+ coordStr] = px2MicronDict['x']*(df['x (px, hash)'] + origin[0])
            df['y '+ coordStr] = px2MicronDict['y']*(df['y (px, hash)'] + origin[1])
            df['z '+ coordStr] = px2MicronDict['z']*(df['z (px, hash)'] + origin[2])
        elif coordStr == '(um, rheo_sedDepth)' or \
                coordStr == '(um, rheo_sedHeight)':
            df['x '+ coordStr] = px2MicronDict['x']*(  df['x (px, hash)']
                                                     + origin[0]
                                                     - imgDim['x']/2.0 )
            # note y-axis is flipped relative to img to give right handed coord systm
            df['y '+ coordStr] = px2MicronDict['y']*(  df['y (px, hash)']
                                                       + origin[1]
                                                       - imgDim['y']/2.0)*-1
            if self.dpl.sedOrGel(hv) == 'sed' and coordStr == '(um, rheo_sedHeight)':
                df['z '+ coordStr] = px2MicronDict['z']*(  df['z (px, hash)']
                                                         + origin[2]
                                                         - gelTopImageStack)
            # note z-axis is flipped to give depth in the sample
            elif self.dpl.sedOrGel(hv) == 'sed' and coordStr == '(um, rheo_sedDepth)':
                df['z ' + coordStr] = px2MicronDict['z'] * (df['z (px, hash)']
                                                            + origin[2]
                                                            - gelTopImageStack
                                                            - shearPostImageStack)*-1
            elif self.dpl.sedOrGel(hv) == 'gel':
                print('Caution! Converting gel hash coordinates to rheoSedDepth')
                print(' must be checked to ensure that yaml file is correct and')
                print(' gelSedimentLocation_fullStack has the initial z-slice at')
                print(' the location of the coverslip. Another option is to use')
                print(' piezoPos and assume gel is of constant thickness and interface')
                print(' may drift. Doing this correctly will involve dataprocessing to')
                print(' to identify drift amoong possible z motion of tracers due to')
                print(' normal forces and remove the drfit')

                df['z '+ coordStr] = px2MicronDict['z']*(  df['z (px, hash)']
                                                           + origin[2]
                                                           - gelTopImageStack
                                                           + gelTopRef)
        elif coordStr == 'all':
            self.recenterHash(df,hv,coordStr='(um, rheo_sedHeight)')
            self.recenterHash(df,hv,coordStr='(um, rheo_sedDepth)')
            self.recenterHash(df,hv,coordStr='(px, imageStack)')
            self.recenterHash(df,hv,coordStr='(um, imageStack)')
        else:
            print("Coordinate system not recognized in recenterHash")
            print("input param :coordStr={}".format(coordStr))
            raise KeyError
        return df

    @staticmethod
    @numba.jit(nopython=True, nogil=True, parallel=True)
    def sumError(z_std, y_std, x_std, zPx, yPx, xPx):
        """
        compute sum of squared errors in um.
        :z_std, y_std, and x_std are arrays of output lsq_refine and each value represents
                                 the ** pixel** uncertainty in the position of the particle
                                 in each coordinate
        :zPx, yPx, and xPx are float32 pixel to micron multiplicatiive
                          conversion factors.
        :return an array of sqrt(sum(*_std)**2)
        """
        out = np.empty_like(z_std)
        for n in prange(len(z_std)):
            error = np.sqrt((z_std[n] * zPx) ** 2 + (y_std[n] * yPx) ** 2 + (x_std[n] * xPx) ** 2)
            out[n] = error
        return out

    @staticmethod
    def computeError(df, pos_keyList=('z_std', 'y_std', 'x_std'), px2Micron=[0.15, 0.115, 0.115]):
        """
        Wrapper function around np_sumError to handle dataFrame inputs
        """
        try:
            z = df[pos_keyList[0]].to_numpy()
            y = df[pos_keyList[1]].to_numpy()
            x = df[pos_keyList[2]].to_numpy()
            zPx = px2Micron[0]
            yPx = px2Micron[1]
            xPx = px2Micron[2]
            totalError = ParticleStitch.sumError(z, y, x, zPx, yPx, xPx)
        except KeyError:
            print('locationStitch.computeError did not find key {} in location datFrame'.format(pos_keyList[0]))
            print('assigning -1 to totalError column and continuing')
            print('This will result in overlaps being resolved by just picking one of the two randomly (lowest index?)')
            totalError = np.empty(df.shape[0])
            totalError[:] = -1.0

        return pd.Series(totalError, index=df.index, name='totalError')

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def minError_flagDouble(pairList, errorList):
        """
        :pairList list of pairs of integers specificying iloc location in pandas dataFrame where two positions
                                            are within some cutoff distance used to generate the pairs
        :errorList: list of total std errors with same indexing scheme as the parent dataFrame from which
                                             the pairList was derived
        :return pair of values (True, False) for example, on what to set the keepBool value
        """
        out1 = np.empty_like(pairList)
        out2 = np.empty_like(pairList)
        for n in range(len(pairList)):
            p1 = pairList[n, 0]
            p2 = pairList[n, 1]
            errorPair = (errorList[int(p1)], errorList[int(p2)])
            out1[n, 0] = p1
            out1[n, 1] = bool(errorPair[0] < errorPair[1])
            out2[n, 0] = p2
            out2[n, 1] = bool(errorPair[1] <= errorPair[0])
        return out1, out2

    @staticmethod
    def removeDoubles(df,
                      cutoff,
                      material,
                      frame,
                      posKeyList=('z (um, rheo_sedHeight)', 'y (um, rheo_sedHeight)', 'x (um, rheo_sedHeight)'),
                      px2Micron=(0.15, 0.115, 0.115)):
        """
        Wrapper function that carries out a few stesp to remove double hits

        :df dataframe of locating output. Can include both sed and gel and multiple frames.
                                          Really should be a dask dataframe for out of mem computations
        :cutoff, float32 distance in microns below which particle pairs should be anaylzed for removing one of the pairs
        :material, str, either 'sed' or 'gel'
        :frame, int, what frame number to remove doubles from. zero-indexed as standard for trackpy
        :posKeyList, tuple of strings, keys for dataFrame df giving the z,y,x positions. df[posKeyList[0]] returns a series
                                       of particle positions
        : px2Micron, tuple of float32, mutiplicative factors converting pixel to microns. Passed to compute error to
                                       compute the total squared error to determine which particle to flag for removal
        """
        # split into sed and gel for a given timeFrame and delete the combined list
        df_partial = df[(df['material'] == material) &
                        (df['frame'] == frame)].reset_index(drop=True)

        # do some filtering based on ilastik keys
        # new, Jul 20 2022
        if material == 'gel':
            try:
                # Note that you have to reset_index of the dataFrame as the removeDoubles routine will involved
                # iloc and numpy arrays to match the keepBool column. This bug took me all day to find. Sigh.
                # Zsolt, Aug 2 2022
                df_partial = df_partial[(df_partial['gel_Tracer'] > df_partial['gel_Background'])].reset_index(drop=True)
            except: KeyError('Could not filter with ilastik keys. Stitching centroid?')

        # create the search tree (fast scipy cKDTree)
        tree = cKDTree(df_partial[list(posKeyList)])

        # query the pair list (also fast)
        pairs = np.array(sorted(tree.query_pairs(cutoff)))

        # compute total Error for every particle (accelerated by numba)
        # will fail for empty dataFrames
        df_partial['totalError'] = ParticleStitch.computeError(df_partial, px2Micron=px2Micron)

        # call minError_flagDouble (numba wrapped in python)
        flag1, flag2 = ParticleStitch.minError_flagDouble(pairs, df_partial['totalError'].to_numpy())

        # combine and remove doubles using groupby and apply on dataframe
        flag_df = pd.DataFrame(np.concatenate((flag1, flag2), axis=0), columns=['index', 'keepBool']). \
            set_index('index').groupby('index', as_index=True).apply(pd.DataFrame.product)

        # format the dataFrame to have boolean values in keepBool index and fill out the array for all the particles that were never flagged for possible removal
        df_partial['keepBool'] = 1  # set all the values to keep
        df_partial[
            'keepBool'] = flag_df  # for each index in flag_df set df_partial.loc[index]['keepBool'] to flag_df.loc[index]
        df_partial['keepBool'] = df_partial['keepBool'].fillna(
            1)  # if it was never check for double hits, keepBool value is nan and you must keep it
        df_partial['keepBool'] = df_partial['keepBool'].replace(1, True)
        df_partial['keepBool'] = df_partial['keepBool'].replace(0, False)
        return df_partial

    def stitch(self,
               hvList,
               cutOff=0.25,
               recenterDict ={'flag': True, 'coordStr':'(um, imageStack)'},
               **kwargs):
        """
        stitches all the hashValues in hvList

        Params
        ~~~~~~~~~
        hvList: list, of hashValues. Will be used to form fileNames
        cutoff: float, passed to removeDoubles. A distacne in the same units as coordStr in recenterDict
        recenterDict: dictionary, of parameters pass to addHashValue, importantly specifying whether to recenter
                                  and what coord system (as specified by coordStr) to return the results
        **kwargs: dict, optional argument for later use.

                  'path': Currently added keyword argument 'path' to specify the the path
                          the location csv files are at. If not specified will default to None and resume
                          previous functionality
                          - Zsolt July 21 2021
        """

        # loop over hashValues
        try: path = kwargs['path']
        except KeyError: path=None
        for hv in hvList: self.addHashValue(hv,recenterDict=recenterDict, path=path)

        loc = self.locations
        if recenterDict['flag'] == False: posLabels = ['z', 'y','x']
        else:
            coordStr = recenterDict['coordStr']
            posKeyList = ['z {}'.format(coordStr),
                         'y {}'.format(coordStr),
                         'x {}'.format(coordStr)]
        mat = self.dpl.sedOrGel(hvList[0])
        frame = self.dpl.queryHash(hvList[0])['index'][-1]
        px2Micron_dict = self.dpl.metaData['imageParam']['px2Micron']
        px2Micron = (px2Micron_dict['z'],
                     px2Micron_dict['y'],
                     px2Micron_dict['x'])

        # remove doubles
        stitch = self.removeDoubles(loc,
                                    cutoff=cutOff,
                                    material=mat,
                                    frame=frame,
                                    posKeyList=posKeyList,
                                    px2Micron=px2Micron)
        #stitch = stitch.astype({'frame':'int8',
        #                        'hashValue':'int8',
        #                        'n_iteration':'int8',
        #                        'material':'str'})
        self.locations = stitch
        return stitch

    def df2h5(self, df, mat, stem):
        """ send all columns of df to hdf5 dataStore using tp.PandasHDFStore"""
        fName = self.dpl.metaData['fileNamePrefix']['global']
        fName +='{}_stitched.h5'.format(mat)
        with tp.PandasHDFStore(stem+'/{}'.format(fName)) as s:
            s.put(df)
        return stem+'/{}'.format(fName)

    def stitchAll(self, tMin= None, tMax = None, cutOff=0.25, matList= None):
        hash_df = self.dpl.hash_df
        if tMax is None and tMin is None:
            tMin = 0
            tMax = hash_df['t'].max()
        if matList is None: matList = ['sed', 'gel']
        else: matList = [matList]
        for t in range(tMin, tMax+1):
            for mat in matList:
                hvList = hash_df[(hash_df['t'] == t) & (hash_df['material'] == mat)].index
                stitch = self.stitch(hvList, cutOff=cutOff)
                path = self.dpl.getPath2File(0,kwrd='locations', computer=self.computer, pathOnlyBool=True)
                if self.lsqCentroid == 'lsq': fName = self.dpl.metaData['fileNamePrefix']['global']+'stitched_{}'.format(mat)+'_t{:03d}.h5'.format(t)
                elif self.lsqCentroid == 'centroid': fName = self.dpl.metaData['fileNamePrefix']['global']+'stitched_centroid_{}'.format(mat)+'_t{:03d}.h5'.format(t)

                # if hdf file already exists, it should be removed before writing to disk.
                # Otherwise, it will likely throw an error as the hdf file is corrupted. Default is to append, so
                # this effectivelty forces an overwrite Zsolt, Jun 2022
                # Better solution: dataFrame.to_hdf(mode='w')
                hdfName = path + '/{}'.format(fName)
                #if os.path.exists(hdfName): os.remove(hdfName)

                stitch.to_hdf(hdfName, str(t),mode='w')
                print("stitched material {} at time t={}".format(mat,t))
        return True

    def parStitchAll(self,tMin, tMax, n_jobs=16):
        return Parallel(n_jobs=n_jobs)(delayed(self.stitchAll)(tMin=x, tMax=x) for x in tqdm(range(tMin,tMax+1)))


    #def splitNSave(self,
    #               df = self.locations,
    #               coordStr=('(um imageStack)', '(px, hash)' ),
    #               path = '',
    #               col_partitions = {'locations': ['z','y','x','frame','hashValue'],
    #                                 'locationOverlap': ['z','y','x','frame', 'hashValue'],
    #                                 'locationHash': ['z','y','x', 'frame', 'hashValue'],
    #                                 'locatingExtra': {'gel': ['mass', 'raw_mass', 'n_iteration', 'disc_size','frame'],
    #                                                    'sed': ['mass', 'raw_mass', 'n_iteration', 'size','frame']},
    #                                 'locatingError': ['z_std', 'y_std','x_std', 'cost', 'totalError','frame'],
    #                                 'errorExtra': ['background','background_std','signal', 'signal_std','ep','frame']
    #                                 }
    #               ):
    #    prefix = self.dpl.metaData['fileNamePrefix']['global']
    #    prefix +='_{}'.format(mat)
    #    # select the right columns and rows if applicable
    #    for key in col_partitions:
    #        prefix += '_{}.h5'.format(key)
    #        if key == 'locations' or key == 'locationOverlap':
    #            col_partitions[key] = ['z {}'.format(coordStr[0]),
    #                                   'y {}'.format(coordStr[0]),
    #                                   'x {}'.format(coordStr[0]),
    #                                   'frame', 'hashValue']
    #            with tp.PandasHDFStore(stem+'/{}'.format(prefix)) as s:
    #                columns = col_partitions[key]
    #                if key =='locations': tmp = df[columns][df['keepBool']]
    #                elif key =='locationOverlap': tmp = df[columns][~df['keepBool']]
    #                s.put(tmp)

    #        elif key == 'locationHash':
    #            col_partitions[key] = ['z {}'.format(coordStr[1]),
    #                                   'y {}'.format(coordStr[1]),
    #                                   'x {}'.format(coordStr[1]),
    #                                   'frame', 'hashValue']
    #        else: pass
    #        #with tp.PandasHDFStore(stem+'/{}'.format(fName)) as s:
    #        #    tmp = df[]
    #        #    s.put()
    #    return True
        # keep the index in all cases
        # export using pd.to_hdf()


        # write to hdf5 or parquet file possibly multiple files to able to load some but not all columns :
        #  -> bare minimum of locations in um for particles to keep

        #  -> locations in um that were removed in stitching
        #  -> frame number
        #  -> errors from refinement
        #  -> locating extras (mass, cost, hv, px locations)
        #  -> everything else that is not null and is not kept around for book keeping
        #     ->

    @staticmethod
    def dataFrameLocation2xyz(df,outDir, columns = ['x','y','z'], particleStr = 'X'):
        """
        outputs a datFrame with location labels to xyz file format
        The function does not do any selection of the data beyond putting
        the values from columns input and formatting the xyz file correctly.
        ___________
        df: the dataFrame, probably some variation on the output from trackpy locatin
        columns: a list of column labels to output. These have to be keys in the input data frame
                 They could 'x (px, hash)' or 'x (um, sedDepth)' for example.
        outDir: /path/to/wehre/file/should/be/saved/fileName.xyz
        """
        # write the header for the xyz file
        n_particle = len(df.index)
        df['particleString'] = particleStr
        with open(outDir,'w') as f:
            f.write(str(n_particle)+'\n\n')
            # if no path is provided to df.t_csv(), it returns a str of the output
            f.write(df.to_csv(columns=['particleString']+columns,sep =' ',index=False, header=False))
        return outDir

    def trimEdges(self,hv):
        """
        Eliminates particles in hv that are within 1 diameter of the edge of the sample
        unless that edge is the sed/gel interface

        All computations are done on (px, hash) coordinates
        :param hv:
        :return:
        """
        # find the bounds for the input hashValue
        # what is 1 diameter in terms of pixels?
        # sub defitition edgeBool(pos) returns True particle is on edge
        # load dataFrame and apply edgeBool() on all 6 edges in parralel
        # find unique values in the return list
        # return the dataFrame with edge particles dropped.

    def _getCompleteHashValues(self,locDir=None, fName_regExp=None):
        """
        Looks in the location directory specified in locDir (if None, default is to look in yamlMetaData)
        and returns a list of hashValues that are complete
        :return: list of hashValues that have been located
        """
        if locDir == None: locDir = self.dpl.getPath2File(0,kwrd='locations', \
                                                          computer=self.computer, \
                                                          pathOnlyBool=True)
        # get a list of all the csv files in the directory
        if fName_regExp == None:
            fName_regExp = '*_trackPy_lsqRefine.csv'
        locationFiles = glob.glob(locDir+'/{}'.format(fName_regExp))
        # do some string processing to extract just the hashValues.
        def fName2hv(fName):
            """ parses str:FName and returns int:hashValue"""
            start = fName.find('_hv')
            hv = int(fName[start +3: start + 8])
            return hv
        return [fName2hv(fName) for fName in locationFiles]

    def _findDoubleHits(self, hv1,outputStr='stitch'):
        """
        Given two hashValues, returns a list of double hits and checks that the double hits
        are within the expected overlap region of the two hashValues.
        The algorithm proceeds as:
        -> read in dataFrame: df1 associated with locations for input hashValue
        -> find all overlapping hashValue of hv1 that have been completed
        -> set a permanent index on df1
        -> looping over all nnb that have hashValue larger than df1 but at the same time point
           -> read in that dataFrame as df2
           -> reshift to absolute pixel coordinates
           -> create a psuedoTime index on df2 as 2 and df1 as 1
           -> find pseudo trajectories that are of length 2 using trackpy
           -> remove those pseduo trajectories from df1 by referening the permanent index on df1
           -> go to the next nnb
        -> This process will yield a dataFrame df1 that will not have any double hits with future
           dataFrames...but it does not gaurantee that I am keeping the most accurate (ie most interior location)

        I think this algorithm is indepndent for hashValue, which is a bit surprising in that stitching
        is not obviously a embarrassingly parralel, of course there is alarger memory footpring as
        I  have to read in nnb hashVlaue multiple times if that hashValue has multiple overlaps

        outputStr:
          -> 'stitch': add the datFrame to self.locations by concatenation after removing double hits
                       return None
          -> 'df_noDoubleHits': return the dataFrame with doubleHits removed
          -> 'df_split': return a pair (df_noDoubleHits, df_onlyDoubleHits)

        ToDo:
          [ ] add some output options including outputing the overlapping particles and warnings if some hashValues
              from nnb list are missing because those locations have not been completed. Also some log dict as
              output would be good.
          [ ] implement a parallel for loop to loop over input hashValues
          [ ] how are the results going to be stored? Exported? Are there going to be any other analysis
              that I should do within the par for loop?
          [ ] Add some reporting on the status of number of nnb left to check
          [ ] This analysis step should also be reading off of a yaml parameter file, albeit I think that file
              should be different from the dpl file...or maybe one file with two separate pipelines:
              -> decon particle locating (flatfield,decon,postDeconCombined)
              -> stitching, visualization, and tracking
          [ ] what is the check that the algorithm works?
              -> Visualize the particles in ovito and look for overlaps. Maybe output an xyz file with particle
                 and doublehit as different atom types?
              -> check that the particles removed were all in the overlap region and visualize the overlap
                 region to ensure that all particle were there and the overlap region was densely populated
                 removed double hits to ensure that no double hits were missed
          [ ] add some cropping (maybe) to try to select the "best" particle in the overlap region
              -> maybe this should be done later, after stitching. How much better is the best particle
                 given that the center overlap threshold is 0.1 pixels? A: I think this is the wrong question...
              -> Possibly severe as the strict overlap may be leaving some particles that are on the edge
                 and incorrectly located in one of the stacks. These edge particles really should be deleted as the
                 positions are wrong.
              -> Check that you dont crop particles at the sed/gel interface on either the sed or gel side
                 If memory serves, smartCrop and the hashing regions handle this well, but this should be
                 explicitly checked by visualizeing the locating quality on the interface hashes.

        :return:
        """
        # get list of nnb hashValues and select:
        #   - only those hashValue larger that hv1.
        #   - present in completed hashValues
        #     -> maybe I dont need to check this beforehand and can just use try/except
        #   - same material as input hashValue
        mat = self.dpl.sedOrGel(hv1)
        nnb = sorted(self.dpl.getNNBHashValues(hv1)[mat]) # sorted(list) returns the list, list.sort() returns None
        completeHVList = self.getCompleteHashValues()
        # initialize df2 outside for loop and add particle index that is fixed to df1 only, as higher indices dont matter
        df1 = self.csv2DataFrame(hv1)
        df1.reset_index()
        #df1.rename(columns={'index': 'index_hv'})
        df1['pseudoTime'] = 1
        df1['hashValue'] = hv1
        #df1['index_hv'] = np.arange(0,df1['hashValue'].count()) # add a permanent index specific to the hashValue
        # note, i could have used any key above that had a value for every row. I did not need to use 'hashValue'
        df1['index_hv'] = np.arange(0,len(df1.index)) # add a permanent index specific to the hashValue
        self.recenterHash(df1, hv1) # dont have to reassign recenter values to df1 as df1 is modified in place

        # now loop over the nearest neighbor hashValues that have hashValue loarger than hv1
        print("Starting stitch on hashValue {}".format(hv1))
        for hv2 in nnb:
            if (hv2 in completeHVList) and hv2>hv1:
                df2 = self.csv2DataFrame(hv2)

                # add pseudotime on which to apply tp.link
                df2['pseudoTime'] = 2

                # add hashValue as column to keep track of what hashValue this is
                df2['hashValue'] = hv2

                # add permanent index, amybe I dont need this for df2
                # df2['index_hv'] = np.arange(0,
                #                             df2['hashValue'].count())  # add a permanent index specific to the hashValue

                # apply recenterHash, not df2 is modified in place
                self.recenterHash(df2, hv2)

                # merge to get one dataFrame with different time points
                df = pd.concat([df1,df2])

                # run trackpy.link and then search for complete trajectories
                """
                # it also would be prudent to plot a histogram of the actual distances
                #   which be very small, on the order of the locating error
                #   and an estimate of the number of doulbe hits given the spatial overlap
                #   of the hashValues. Some of the hv in nnb may not actually have overlap...not sure
                #   and defineitely worth checking as it would save time
                #
                # particles that need to dropped here ahve pseudoTime = 0 and should have a
                # permanent index that has been unaltered. Find that permanent index and remove
                # indices from df1 and go onto to next nnb hashVlaue with modified df1.
                """
                doubleHits = trackpy.link(df,search_range=0.5,\
                                             pos_columns=['z (um, imageStack)',\
                                                          'y (um, imageStack)', \
                                                          'x (um, imageStack)'],\
                                             t_column='pseudoTime')
                grouped = doubleHits.reset_index(drop=True).groupby("particle")
                # filtered is without a question working as I think it should.
                # It is giving me the particles that are within the search_range in doubleHits
                filtered = grouped.filter(lambda x: x['pseudoTime'].count() == 2)
                #doubleHitIDList = filtered.set_index('pseudoTime', drop=False)['particle'].unique()
                #index_hvList_doubleHit = df[(df['pseudoTime']==1) \
                #                            & (df['particle'].isin(doubleHitIDList))]['index_hv']
                index_hvList_doubleHit = [int(elt) for elt in list(filtered[pd.notnull(filtered['index_hv'])]['index_hv'])]
                df1 = df1[~df1['index_hv'].isin(index_hvList_doubleHit)]
                #df1_onlyDoubleHits = df1[df1['index_hv'].isin(index_hvList_doubleHit)]

                # from here, take the elts of doubleHitIDList and:
                #  -> find the index_hv values
                #  -> remove those values from df1
                #  -> return df1 to test against the next nnb hashValue

            else: pass
        # rerun on hv to next nnb
        # add smaller hv to self.locations, but without this fake time column
        # and correctly handle the following cases:
        #  - no particle is completely removed as overlapping particle will be presnet in larger hv
        #      and likely have no overlap when repeated on larger hv
        #  - possible chain of overlaps where a single particle is in 4 distinct hashValues
        # store the double hits in a seperate dataFrame with columns

        # return df1, the dataFrame that can be placed by simple concantenation onto the "quilt" of positions without
        # introducing double hits
        if outputStr == 'df_noDoubleHits': return df1
        elif outputStr == 'df_split':
            print("not yet coded")
            #return (df1_noDoubleHits,df1_onlyDoubleHits)
        elif outputStr == 'stitch':
            #self.locations = pd.concat([self.locations,df1])
            #return self.locations
            return df1
        else:
            print("outputStr {} is not recognized!".format(outputStr))
            raise ValueError

    def _parStitch(self,n_jobs=8,matStr='sed'):
        """
        Cycle through all the available hashValues, stitch them by adding to self.locations
        :param n_jobs: number of cores to use for par for loop
        :param matStr: 'sed' or 'gel' or 'all'
        :return: none
        """
        # get a sorted list of complete hashValue specific to the input matStr.
        completedHV = self.getCompleteHashValues()
        if matStr == 'sed' or matStr =='gel':
            allHashValue = set([elt \
                                for elt in [int(keyStr) for keyStr in self.dpl.hash.keys()]\
                                if self.dpl.sedOrGel(elt) == matStr])
            stitchList = sorted([hv for hv in completedHV if hv in allHashValue])
        elif matStr == 'all': stitchList = completedHV
        else:
            print('matStr input to parStitch must be \'sed\' \'gel\' or \'all\', not {}'.format(matStr))
            raise KeyError

        self.locations = pd.concat(Parallel(n_jobs=n_jobs)(delayed(self.findDoubleHits)(hv) for hv in stitchList))
        return self.locations

if __name__ == '__main__':
    yamlTestingPath = '/Users/zsolt/Colloid/SCRIPTS/tractionForceRheology_git/TractionRheoscopy' \
                      '/metaDataYAML/tfrGel09052019b_shearRun05062019i_metaData_scriptTesting.yaml'
    tmpDir ='/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/buf'
    inst = ParticleStitch(yamlTestingPath,computer='MBP')
    inst.getCompleteHashValues()
    #inst.findDoubleHits(41)
    #dfSed = inst.parStitch()
    inst.dataFrameLocation2xyz(inst.locations,tmpDir+'/tmpAllSed.xyz',\
                               columns=['x (um, imageStack)','y (um, imageStack)','z (um, imageStack)'])

    """
    June 27 2021
    
    Commands to stitch 
    
    >>> stitch_inst = locationStitch.ParticleStitch(yamPath)
    >> stitch_inst.stitchAll()
    
    This will save an h5 file for every time point with all columns. There is no tracking, only stitching.
    """



