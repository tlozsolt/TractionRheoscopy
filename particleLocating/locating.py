import trackpy as tp
import functools,yaml
from paintByLocations import particleGlyph as pg
#import flatField
import numpy as np
import pandas as pd
import scipy

# ToDo:
#  [+] write a iterative residual locating using a recursive function
#  [ ] export the dataFrame location results to text file and back to pandas
#  [ ] have the locating parameters be read from the yaml file.
#  [ ] maybe this should be combined with paint by locations given that I will be assessing
#      the accuracy of the locations using paint by locations.
#  [ ] should I make my own locations class? Is that really necessary? Maybe yes as I will want to
#      to save some of the metaData as well.
#  [ ] If I did make a class, what methods and attributes would I want to include?
#        [ ] locations dataFrame with hashed values
#        [ ] extra parameters dataFrame
#        [ ] locating parameters with easy export to yaml
#        [ ] locations that have been unhashed.
#        [ ] glyphs for paint by locations.
#        [ ] basically the best version of paint by locations to visualize what happened during dpl pipeline
#        [ ] methods for descriptive statistics like histrograms of extra parameters

def iterate(imgArray, paramDict,material):
    """
    This function applies the locatingFunc to the imageArray iteratively following Kate's work.
    The basic idea is to locate some particles, create a mask to zero out the positions of the located particles
    and then run locating again on the (partially) zeroed imgArray. Also, concantenate the particle locations
    :param imageArry:
    :param paramDict: read from yaml file; contains a dictionary for iterating and a dictionary of kwargs for locating
    :return: pandas data frame with particle locations and extra output from trackpy (ie mass, eccentricity, etc)
    """
    locatingParam = paramDict[material]
    iterativeParam = paramDict['iterative']
    particleBool = True # did we find additional particle on this iteration?
    # maybe this should be done in the while loop to account for changing parameters during iteration
    locList = []
    iterCount = 0
    maxIter = iterativeParam['maxIter']
    while particleBool == True and iterCount < maxIter:
        try: locateFunc = functools.partial(tp.locate, **locatingParam[iterCount])
        except IndexError:
            locateFunc = functools.partial(tp.locate, **locatingParam[-1])
        print("Iteration: {}".format(iterCount))
        iterCount += 1
        loc = locateFunc(imgArray).dropna(subset=['z']) # remove all the rows that have NAN particle positions
        print("{} particles located!".format(loc.shape))
        if loc.shape[0] == 0:
            particleBool=False
            break
        locList.append(loc) # add the dataframe to locList and deal with merging later
        loc['n_iteration'] = iterCount # add a new column to track what iteration the particle was located on.
        mask = createMask(loc,imgArray,iterativeParam['mask'][material])
        imgArray = imgArray*np.logical_not(mask)
    particleDF = pd.concat(locList).rename(columns={"x": "x (px)", "y": "y (px)", "z": "z (px)"})
    logDict = {'locating' : {'particles': particleDF.shape[0], 'iterations': iterCount}}
    return [particleDF, logDict]

def createMask(locDF, imgArray, glyphShape):
    """
    Create a mask of True values where there is a particle in locDF
    :param locDF: locations data frame from trackpy
    :param shape: shape of the image on
    :return:
    """
    glyphShape = np.array(glyphShape) # just to make sure that it is an array.
    glyph = pg(glyphShape,glyphShape + 2 )

    maskGlyph = glyph.mask
    deltaKernel = glyph.deltaKernel

    imgMask = np.zeros(imgArray.shape,dtype='bool')

    #paddedImg = scipy.signal.oaconvolve(imgArray,deltaKernel)
    # now set the coordinate centers to one
    xCoord = np.rint(locDF['x']).astype(int)
    yCoord = np.rint(locDF['y']).astype(int)
    zCoord = np.rint(locDF['z']).astype(int)
    imgMask[zCoord,yCoord,xCoord] = 1
    imgMask = scipy.signal.oaconvolve(imgMask,maskGlyph)
    # crop the mask
    [dz,dy,dx] = ((np.array(maskGlyph.shape) -1)/2).astype(int)
    imgMask = imgMask[dz:-dz,dy:-dy,dx:-dx]
    # Do we want to do anything with the possible values of say two or three that signify overlap between the particles?
    # if so, this needs to be dealt with before the boolean thresholding in the next line
    # Deal with machine precision
    imgMask[imgMask<0.1] = False
    imgMask[imgMask>0.1] = True
    # imgMask is now True if there was a particle there.
    # also note that this type of mask is essentially identical to the pixel training data that would be required
    # for a machine learning based image segmentation.
    return imgMask.astype(bool)

if __name__ == '__main__':
    inputImgPath ='/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/' \
                  'postDecon/tfrGel09052019b_shearRun05062019i_postDecon8Bit_hv00085.tif'
    metaDataPath = '/Users/zsolt/Colloid/SCRIPTS/tractionForceRheology_git/TractionRheoscopy/metaDataYAML/' \
                   'tfrGel09052019b_shearRun05062019i_metaData_scriptTesting.yaml'
    with open(metaDataPath,'r') as stream: metaData=yaml.load(stream,Loader=yaml.SafeLoader)
    #stack = pims.open(inputImgPath)
    #stack = flatField.zStack2Mem(inputImgPath)
    #inputDict = {'diameter': [7,19,19],'preprocess': False, 'minmass': 10000}
    paramDict = metaData['locating_trackpy']
    features = iterate(stack,paramDict,'sed')
