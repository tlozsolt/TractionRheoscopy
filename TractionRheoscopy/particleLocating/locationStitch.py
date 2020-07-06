import dplHash_v2 as dpl
import pandas as pd
from joblib import Parallel, delayed
import glob
import trackpy
import numpy as np
import locating, hashStitch


class ParticleStitch(dpl.dplHash):
    def __init__(self,metaDataPath,computer='ODSY'):
        self.dpl = dpl.dplHash(metaDataPath)
        self.computer= computer
        self.locations = pd.DataFrame(columns = ['z (px, hash)', 'y (px, hash)', 'x (px, hash)',\
                                                     'image mass', 'hashValue', 'material',\
                                                     'z (um)', 'y (um)', 'x (um)'\
                                                     ]\
                                          )

    def getCompleteHashValues(self):
        """
        Looks in the location directory specified in yamlMetaData and returns a list of hashValues
        that are complete
        Will raise a warning if list is not complete
        :return: list of hashValues that have been located
        """
        locDir = self.dpl.getPath2File(0,kwrd='locations',\
                                       computer=self.computer,\
                                       pathOnlyBool=True)
        # get a list of all the csv files in the directory
        locationFiles = glob.glob(locDir+'/*_trackPy.csv')
        # do some string processing to extract just the hashValues.
        def fName2hv(fName):
            """ parses str:FName and returns int:hashValue"""
            start = fName.find('_hv')
            hv = int(fName[start +3: start + 8])
            return hv
        return [fName2hv(fName) for fName in locationFiles]

    def csv2DataFrame(self,hv):
        pxLocationExtension = '_' + self.dpl.sedOrGel(hv) + "_trackPy.csv"
        path = self.dpl.getPath2File(hv,\
                                     kwrd='locations',\
                                     extension=pxLocationExtension,\
                                     computer=self.computer)
        df = pd.read_csv(path)
        df['hashValue'] = hv
        df['material'] = self.dpl.sedOrGel(hashValue=hv)
        return df.rename(columns = {'z (px)': 'z (px, hash)',
                             'y (px)': 'y (px, hash)',
                             'x (px)': 'x (px, hash)'})

    def addHashValue(self,hv):
        """
        for a given hashValue, will add append the locations to the self.locations and add column
        specifying the hashValue and material string with no other modification
        :return:
        """
        df = self.csv2DataFrame(hv)
        #df['hashValue'] = hv
        #df['material'] = self.dpl.sedOrGel(hashValue=hv)
        #df.rename(columns = {'z (px)': 'z (px, hash)',
        #                     'y (px)': 'y (px, hash)',
        #                     'x (px)': 'x (px, hash)'})
        self.locations = pd.concat([self.locations,df])
        return True

    def recenterHash(self,df,hv, coordStr = 'all'):
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
        origin,dim = self.dpl.integrateTransVect(hv,computer=computer)
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

    def findDoubleHits(self, hv1,outputStr='stitch'):
        """
        Given two hashValues, returns a list of double hits and checks that the double hits
        are within the expected overlap region of the two hashValues.
        The algorithm proceeds as:
        -> read in dataFrame: df1 associated with locations for input hashValue
        -> find all overlapping hashValue of hv1 that have been completed
        -> set a permanent index on df1
        -> looping over all nnb that have hashValue larger than df1
           -> read in that dataFrame as df2
           -> reshift to absolute pixel coordinates
           -> create a psuedoTime index on df2 as 2 and df1 as 1
           -> find pseudo trajectories that are of length 2 using trackpy
           -> remove those pseduo trajectories from df1 by referening the permanent index on df1
           -> go to the next nnb

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


    def parStitch(self,n_jobs=8,matStr='sed'):
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
        elif matStr == 'all': stitchList = completdHV
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



