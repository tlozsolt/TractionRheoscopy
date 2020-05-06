import trackpy as tp
import functools,yaml
import numpy as np
import pandas as pd
import scipy


class particleGlyph:
    """
    A class for storing a an array of intensity with a strict ellipsoidal mask and defined intensity shading
    """
    def __init__(self,dim,boxDim):
        """
        :param dim: the dimensions of the ellipse (a,b,c)
        :param boxDim: the dimensions of the kernel.
        """
        # initalize a numpy array of dimensions dx,dy,dz
        # round up dz dy dz to odd numbers if input was even
        out = []
        for d in boxDim:
            if bool(d%2==0) == True: d=d+1
            out.append(d)
        [dz,dy,dx] = out
        glyphBox = np.zeros([dz,dy,dx],dtype='uint8')
        deltaKernel = np.zeros([dz,dy,dx],dtype='uint8')
        mask = np.zeros([dz,dy,dx],dtype='uint8')
        # put the ellipsoid in the center and define the shading. I think it should fall off linearly in intensity
        # scan through the indices in the box. If within the ellipsoid region, then decide on linear shading

        def ellipseInterior(pt,dim,center,out='bool'):
            (z,y,x) = pt
            (c,b,a) = [elt/2 for elt in dim]
            (cz,cy,cx) = center
            if out == 'bool':
                if ((x-cx)/a)**2 + ((y-cy)/b)**2 + ((z-cz)/c)**2 < 1: return True
                else: return False
            elif out == 'norm':
                return ((x-cx)/a)**2 + ((y-cy)/b)**2 + ((z-cz)/c)**2
            else: print("Unrecognized out kwrd: "+ out)

        def linearShade(x,left,right,min,max ):
            """ as x goes from left to right, the output will go continuously from min to max"""
            m = (max - min)/(right - left)
            b = max - m*right
            return m*x+b

        center = [elt/2 for elt in [dz,dy,dx]]
        deltaKernel[int(np.rint(center[0])), \
                    int(np.rint(center[1])), \
                    int(np.rint(center[2]))] = 1
        for z in range(dz):
            for y in range(dy):
                for x in range(dx):
                    n = ellipseInterior((z,y,x),dim,center,out='norm')
                    if n < 1:
                        glyphBox[z,y,x] = linearShade(n,0,1,10,0)
                        mask[z, y, x] = 1
        self.glyph = glyphBox
        self.mask = mask
        self.deltaKernel = deltaKernel
        self.shading = 'linearShading'
        self.dim = glyphBox.shape
        self.boundingBoxDim = boxDim

def zStack2Mem(path):
    with ski.external.tifffile.TiffFile(path) as tif:
        data = tif.asarray()
    return data

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
    glyph = particleGlyph(glyphShape,glyphShape + 2 )

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
    stack = zStack2Mem(inputImgPath)
    paramDict = metaData['locating']
    features = iterate(stack,paramDict,'sed')
