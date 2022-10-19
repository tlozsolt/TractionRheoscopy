import yaml, os
import numpy as np
import pandas as pd
import math

class dplHash:
    """ a class for deconvolution and particle locating
      large xyzt confocal data in spatial chunks
  """

    def __init__(self, metaDataPath):
        """
      initialize the instance with a  hashTable dictionary and metaData dictionaryf keys
      self.hash(str(hashValue)) returns a dictionary with keys "xyztCropTuples" and "material"
      self.metaData gives a dictionary representation of the associated yaml file
    """
        with open(metaDataPath, 'r') as stream:
            self.metaData = yaml.load(stream, Loader=yaml.SafeLoader)
        # compute the total hash size
        timeSteps = self.metaData['imageParam']['timeSteps']
        # xy chunks are directly computed given chunk size, minimum overlap, and full xy dimensions
        xyDim_full = [self.metaData['imageParam']['xDim'], self.metaData['imageParam']['yDim']]
        # xyDim_crop = [self.metaData['deconvolution']['deconCropXDim'], self.metaData['deconvolution']['deconCropYDim']]

        # how big should each hash be in xy? Also check that sed and gel are the same size, but futre versions
        # might allow for different xy dimensions for sed and and gel. (probably not though)
        xyDim_gel = self.metaData['hash']['dimensions']['gel']['xyz'][:2]
        xyDim_sed = self.metaData['hash']['dimensions']['sed']['xyz'][:2]
        if xyDim_gel == xyDim_sed:
            xyDim_crop = xyDim_sed
        else:
            print("Material dependent xy dimensions for gel vs sed are not supported yet")
            raise KeyError

        # xyDim_minOverlap = [self.metaData['deconvolution']['minXOverlap'],self.metaData['deconvolution']['minYOverlap']]

        # read in min overlap for gel and sed
        xyzDim_minOverlap_gel = self.metaData['hash']['dimensions']['gel']['minOverlap']
        xyzDim_minOverlap_sed = self.metaData['hash']['dimensions']['sed']['minOverlap']

        # check that overlap is same for gel and sed, however future version mights allow for different overlaps
        if xyzDim_minOverlap_gel == xyzDim_minOverlap_sed:
            xyDim_minOverlap = xyzDim_minOverlap_gel[:2]
            zDim_minOverlap = xyzDim_minOverlap_gel[2]
        else:
            print(
                "Material dependent xyz overlaps for gel vs sed are not supported yet (but xy may be different from z)")
            raise KeyError

        # compute the number of hashes in x and y
        xyChunks = math.ceil((xyDim_full[0] - xyDim_minOverlap[0]) / (xyDim_crop[0] - xyDim_minOverlap[0]))

        # z chunks are computed separately for gel and sediment and based on min overlap and z dimension in gel and sed portions
        zDim_full = self.metaData['imageParam']['zDim']
        gelSedZPos = self.metaData['imageParam']['gelSedimentLocation']

        # Get the z bounds for gel and sediment taking into account the min overlap of gel into sed and vice-versa
        zDim_gelMinMax = (0, gelSedZPos + self.metaData['hash']['dimensions']['gel']['pxOverlap_w_sed'])
        zDim_sedMinMax = (gelSedZPos - self.metaData['hash']['dimensions']['sed']['pxOverlap_w_gel'], zDim_full)
        zDim_sed = zDim_sedMinMax[1] - zDim_sedMinMax[0]
        zDim_gel = zDim_gelMinMax[1] - zDim_gelMinMax[0]

        # how big are the chunks in z for sed and gel?
        zDimGel_crop = self.metaData['hash']['dimensions']['gel']['xyz'][2]
        zDimSed_crop = self.metaData['hash']['dimensions']['sed']['xyz'][2]

        # compute the number of chunks
        zChunks_sed = math.ceil((zDim_sed - zDim_minOverlap) / (zDimSed_crop - zDim_minOverlap))
        zChunks_gel = math.ceil((zDim_gel - zDim_minOverlap) / (zDimGel_crop - zDim_minOverlap))

        # create an empty dictionary with keys spanning the hash size
        hashSize = timeSteps * (zChunks_sed + zChunks_gel) * xyChunks * xyChunks
        self.metaData['hashDimensions'] = {'hashSize': hashSize, \
                                           'xyChunks': xyChunks, \
                                           'zChunks_sed': zChunks_sed, \
                                           'zChunks_gel': zChunks_gel, \
                                           'timeSteps': timeSteps}
        # Generate, for each hashValue, the xyz cutoffs...permutations on 3 choices, each, on x and y
        # to generate x points, we start at left (ie 0), \
        # add the cropped dimension until we exceed the full dimension, \
        # and then shift the rightmost point to end at rightmost extent. Same for y
        centerPtListXY = np.linspace(np.floor(xyDim_crop[0] / 2),
                                     np.ceil(xyDim_full[0] - xyDim_crop[0] / 2),
                                     num=xyChunks)
        centerPtListZ_gel = np.linspace(np.floor(zDim_gelMinMax[0] + zDimGel_crop / 2), \
                                        np.ceil(zDim_gelMinMax[1] - zDimGel_crop / 2), \
                                        num=zChunks_gel)
        centerPtListZ_sed = np.linspace(np.floor(zDim_sedMinMax[0] + zDimSed_crop / 2), \
                                        np.ceil(zDim_sedMinMax[1] - zDimSed_crop / 2), \
                                        num=zChunks_sed)
        xCrop = []
        for center in centerPtListXY:
            xCrop.append((int(center - xyDim_crop[0] / 2), \
                          int(center + xyDim_crop[0] / 2)))
        # for n in range(xyChunks):
        #  if n==0: leftPt=0
        #  elif n>0 and n<(len(range(xyChunks))-1): leftPt=n*xyDim_crop[0] - xyDim_minOverlap[0]
        #  else: leftPt = xyDim_full[0] - xyDim_crop[0]
        #  xCrop.append((leftPt,leftPt+xyDim_crop[0]))
        zCrop = []  # list of all the z-positions for gel and sediment
        material = []  # list of str specifiying whether material is sed or gel

        for center in centerPtListZ_gel:
            zCrop.append((int(center - zDimGel_crop / 2), \
                          int(center + zDimGel_crop / 2)))
            material.append('gel')
        for center in centerPtListZ_sed:
            zCrop.append((int(center - zDimSed_crop / 2), \
                          int(center + zDimSed_crop / 2)))
            material.append('sed')

        # for n in range(zChunks_gel):
        #  if n==0: bottomPt=0
        #  else: bottomPt=n*(zDimGel_crop - zDim_minOverlap)
        #  zCrop.append((bottomPt,bottomPt+zDimGel_crop))
        #  material.append('gel')
        # for n in range(zChunks_sed):
        #  if n==0: topPt = zDim_full
        #  else: topPt = zDim_full - n*(zDimSed_crop - zDim_minOverlap)
        #  zCrop.append((topPt-zDimSed_crop,topPt))
        #  material.append('sed')

        hashTable = {}
        hashInverse = {}
        # Do we need a hashInverse table? Yes because the job flow requires numbers to be passed to the
        # job queing. We could in principle make one hash with
        # key:values like '9,3,16':(9,3,16)
        # as opposed to:
        #      hash: '68':(9,3,16) and
        #      inverseHash: '9,3,16':68
        # but we cannot pass '9,3,16' to the job queuing as a hashValue
        # additionally, converting '68' to 68 is simplest possible type conversion \
        # that will most probably work as the default
        # on any level of the job control whether in python, bash or whatever the SLURM is written in.
        # This type conversion is also handled in this class by using the method queryHash
        keyCount = 0
        for t in range(timeSteps):
            for z in range(len(zCrop)):
                for y in range(len(xCrop)):
                    for x in range(len(xCrop)):
                        hashTable[str(keyCount)] = dict([('xyztCropTuples', [xCrop[x], xCrop[y], zCrop[z], t])])
                        hashTable[str(keyCount)]['material'] = material[z]
                        hashTable[str(keyCount)]['index'] = (x, y, z, t)
                        hashInverse[str(x) + ',' + str(y) + ',' + str(z) + ',' + str(t)] = keyCount
                        # are the keys unique? Yes but commas are required to avoid '1110' <- 1,11,0 pr 11,1,0 or 1,1,10
                        keyCount += 1
        """
    I have to check that 'the bounds are working the way I think they are. \
    In particular the overlaps need to added/subtracted somewhere and this, \
    I think, depends on the direction and start point from which I am counting.
    To generate z, we split case into sediment and gel, propagating from gel bottom \
    up until we past gel/sediment and propagate down from TEM grid until we get past gel/sediment.\
    No shifting at the end necessary since this is just going to influence \
    how much gel/sediment overlap is in each chunk. \
    We should make these into functions along with wrapping the dimension counting into functions as well.\
    Additionally, there is likely a better yaml implementation to put dimensions used into \
    an easier to use data structure. \
    Can we view line comments in yaml after importing? Definately yes if you make the comments \
    part of the data structure. Then everything is a value and text string...
    I feel like the rest of this is a nested for loop over 
    indices:  time, no. of gel and  no of sed chunks, xy chunks, 
    """
        # create hash_df to allow for indexing using pandas.
        # I dont want to touch self.hash as I would likely have to rewrite some of the access functions
        # This is however a better more elegant data structure for query in the hash table
        # what the hashValues for first and last time points?
        # >>> hash_df[(hash_df['t'] == 0) |  (hash_df['t']==89)].index
        hash_df = pd.DataFrame(hashTable).T
        hash_df['x'] = hash_df['index'].apply(lambda x: x[0])
        hash_df['y'] = hash_df['index'].apply(lambda x: x[1])
        hash_df['z'] = hash_df['index'].apply(lambda x: x[2])
        hash_df['t'] = hash_df['index'].apply(lambda x: x[3])

        # type cast hash_df index to integer as currently the index is str from hashTable keys that require
        # nonmutable type.
        _idx = hash_df.index.map(int)
        hash_df.set_index(_idx, inplace=True)

        self.hash_df = hash_df
        self.hash = hashTable
        self.invHash = hashInverse
        self.metaDataPath = metaDataPath
