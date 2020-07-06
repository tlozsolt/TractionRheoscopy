import numpy as np
import pims
from scipy import ndimage
import skimage as ski
import time

"""
    This file is contains helper functions to access tiff stacks
    and apply a flat field correction following
    'https://en.wikipedia.org/wiki/Flat-field_correction'
    This is implemented with the softmatter.py code (ie pandas, numpy, etc) 
    
    What should this script/function/class do?
      + Read in 1024x1024 16 bit tiff stack at a specified path given in YAML metaData file 
      + Read in whatever is required to subtract background etc, including Tiff stack or image of 
            flatfield and masterDark
      + Carry out the flatfield correction 
         correctedImage = (raw - dark)*avg(flat - dark)/(flat - dark)
      + Write the result to a 16bit tiff 
      + possibly crop in xy and z or just z depending on resources required by deconvolution. We had 6 hour runs to
            decon 730 x 730 x 50(?) pixels, however now the original image size is 1024x1024 with unchanged z-pixels
            due to binning directly on the camera. 
      + It should also deal with the stitching raw images and have some logical way of addressing the overlap regions
            between adjacent crops...including z, which I might add will likely not be a fixed number for all stacks
            if the z-stack size changes. 
"""


def readMetaData(path2metaData):
    """ this function will read the relevant paths to metaData files
        maybe the reading relevant paths is best thought of a configuration parser?
        Dont know. Will look up.
        It should really be the class initiation of flatfield.py
    """
    return True

def openZstack(path):
    #return(Image.open(path))
    return pims.open(path)

def zStack2Mem_pimsIter(stackIter):
    """ Takes an iterable lazy loading PIMS Frame object and returns a numpy ndarray.
        Dont know if this is a good idea, or if there is a better way to z-project without loading into memory.
        In any case, this is not necessarily a terrible idea if its carried out on individual time points.
    """
    stackMem = np.array([np.zeros((1024,1024))])
    for slice in stackIter:
        #print(stackMem.shape)
        stackMem = np.concatenate((stackMem, np.array([slice])), axis=0)
    return stackMem[1:]

def zStack2Mem(path,stackBool=True):
    """
    Load a tiff stack of tiff series of slices into memory as numpy arry
    :param path: path to stack, or search path with * for z-slice information
    :param stackBool: True is data is saved as a single stack, false if saved as a series.
    :return:
    """
    if stackBool == True:
      with ski.external.tifffile.TiffFile(path) as tif:
        data = tif.asarray()
      return data
    elif stackBool == False:
        from skimage import io
        # load an image collection
        ic = io.ImageCollection(path)
        return ic.concatenate()
    else:
        raise ValueError('stackBool must be either True or False, depending on if the image data is saved as stack or series')


def getTiffStackDim(stack):
    """ Returns a triple (x,y,z) of the stack dimensions in pixels """
    x,y = stack.__dict__['_im_sz']
    z = len(stack)
    return (x,y,z)

def cropStack(stack,cropIndex):
    """ Takes a stack:nparray and cropIndex:nparray and returns a cropStack:nparray
         cropIndex is a triple of integer tuples specifying the crop dimensions
         The intger tuple is clopen [xmin,xmax)
    """
    print(stack.shape)
    xmin,xmax = cropIndex[0]
    ymin,ymax = cropIndex[1]
    try:
      zmin,zmax = cropIndex[2]
      return stack[zmin:zmax,ymin:ymax,xmin:xmax]
    except IndexError:
      return stack[ymin:ymax,xmin:xmax]


def getSliceDim(slice):
    return slice.shape

def array2tif(array,path,metaData=None):
    """ This function creates a tif file from the input array
        It inherits everything (size and dtype) from the underlying array
        INPUT:
          array: numpy array, either 2D slice, zStack, or time series
          path: path to write the file
        OUTPUT:
          -write the file and returns the path complete with filename
    """
    ski.external.tifffile.imsave(path,array,description=metaData)
    return path

def avgSlice(slice):
    """ takes in a numpy array and returns the avg pixel value"""
    return np.mean(slice)

def zProject(stack):
    """ takes a z-stack array and average each xy pixel along z"""
    return np.sum(stack,axis=0)/stack.shape[0]

def avgXYPixel(stack):
    """ takes z-stack array and returns a list of of the avg pixel intensity for every slice"""
    # Yes, you do need to sum axis=1 twice and not axes 1 and 2 because the outer sum call is on a modified stack
    return np.sum(np.sum(stack,axis=1),axis=1)/(stack.shape[1]+stack.shape[2])

def zGradAvgXY(stack):
    """ takes a z-stack array and returns a list of the gradient along z after average xy pixel intensities"""
    return np.gradient(avgXYPixel(stack))

def gaussBlur(slice,sigma=50):
    """ applies a guassian blur and returns the result

        INPUT
        slice: pims frame object or numpy array
        sigma: integer in pixels giving the standard deviation of the gaussian blur. Default is 50px

        OUTPUT:
        blurred image as a numpy array
    """
    return ndimage.gaussian_filter(slice, sigma)

def gaussBlurStack(stack,sigma=50,dz=1,parBool=False):
    """ I want to do something like:
        emptyOutStack[index] = functionOnSlice(fullStack[index])
        This does not currently work
        Additionally, parallelization should be considered a decorator.
        I want to take a function that operates on a slice and make
        a parallel version of the function that will operate on many
        slices in a stack in parrallel and then return the whole shebang as
        an array.
        Additionally, we want a version of the decorator that will take
        a function that takes in one particle and its nnb and returns some
        derived quanitity like local strain and create a decorator that
        will return a modified version of the function that will take a list
        of particles and calculate in parallel, the derived single particle quantity
        and return an array of all of them.
        And i want to write this decorator to work for whatever fuutre fucntion
        that operatoes on a slice or single particle entry or dislocation segment
        or.. some other qunaitity like grains or particle that have coordination
        of a sepcificed range.

        singleFunction(database enetry) vs decorated function(database, list of entries) execute in parallel
    """
    """ Apply a 2D gaussian blur to each slice in a stack"""
    flatStack = np.empty(stack.shape,dtype=stack.dtype)
    for z in range(0,flatStack.shape[0],dz):
        flatStack[z,:,:] = gaussBlur(stack[z,:,:],sigma = sigma)
        #print("gaussBlur on slice z: ",z)
    return flatStack

def correctImageSlice(rawSlice, masterDark, flatSlice):
    """ Apply flatfield correction to single image"""
    out = (rawSlice - masterDark)*avgSlice(flatSlice-masterDark)/(flatSlice - masterDark)
    return out.astype('uint16')

def correctImageStack(rawStack, masterDark, flatStack):
    """ Apply flatfield correction to a whole stack"""
    # declare a numpy array with the right dimensions (same as rawStack)
    # Compute the avgSlice for the whole raw stack
    # carry out the flat fielding operation as an array (not matrix)
    rawStackFloat = rawStack.astype('float32')
    masterDarkFloat = masterDark.astype('float32')
    flatStackFloat = flatStack.astype('float32')
    m = np.mean(flatStackFloat-masterDarkFloat,axis=(1,2))
    #print("computed avg for all slices")
    out = (rawStackFloat - masterDarkFloat)/(flatStackFloat - masterDarkFloat)
    #print("computed out before multiplication by m")
    for slice in range(out.shape[0]):
        out[slice] = out[slice]*m[slice]
        #print("Corrected z slice:", slice)
    #return out.astype('uint16')
    from threshold import arrayThreshold as at
    return at.recastImage(out,'uint16')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # prompt the user for a path to yaml file
    # Test opening the file and signal if successful
    # create a tmp directory and output tiff stacks from each processing step
    # tell the user to check the tmp directory and verify that the files look ok
    start_time = time.time()
    path = '/Users/zsolt/Colloid/DATA/testImagesCodeDebug/'
    rawPath = '/Users/zsolt/Colloid/DATA/testImagesCodeDebug/tfrGel09052019b_shearRun05062019i_t0499.tif'
    darkPath = '/Users/zsolt/Colloid/DATA/testImagesCodeDebug/calibration/darkImage_65ms_camera2922_zStack.tif'
    microRawPath = '/Users/zsolt/Colloid/DATA/testImagesCodeDebug/micro_3x3x3pixels.tif'
    corPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/flatField/'
    corPath += 'tfrGel09052019b_shearRun05062019i_flatField_hv00012.tif'
    deconPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/'
    deconPath += 'decon/tfrGel09052019b_shearRun05062019i_decon_hv00012.tif'
    raw = openZstack(rawPath)
    dark = openZstack(darkPath)
    cor = zStack2Mem(corPath)
    decon = zStack2Mem(deconPath)
    """
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    """
    avgIntensity_z = avgXYPixel(decon)
    plt.plot(avgIntensity_z)
    plt.ylabel('XY avg intensity')
    plt.xlabel('z slice')
    plt.show()
    pixelZGrad = zGradAvgXY(decon)
    plt.plot(pixelZGrad)
    plt.ylabel('grad_z(avg px intensity)')
    plt.xlabel('z slice')
    plt.show()
    maxValue = max(pixelZGrad)
    maxIndex = list(pixelZGrad).index(maxValue)
    print("Maximum grad in avg xy pixels taken along z occurs at: ",str(maxIndex), "with value: ", str(maxValue))

    # print("opening file at path: ", rawPath)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # #for elt in raw.__dict__: print(elt,raw.__dict__[elt])
    # #for slice in raw: print(np.mean(slice))
    # #print(zStack2Mem(f))
    # darkStack = zStack2Mem(darkPath)
    # print("read in dark stack to memory")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # zProjDark = zProject(darkStack)
    # print("zProject dark stack")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # flat = lambda elt: gaussBlur(raw[elt],sigma=50)
    # corIm = correctImageSlice(raw[0],zProjDark,flat(0))
    # print("correct image slice")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # array2tif(corIm, path + 'tmp.tif')
    # print("array2tif image")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # rawStack = zStack2Mem(rawPath)
    # print("zStack2Mem on raw")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # flatStack = gaussBlurStack(rawStack,dz=1)
    # print("guass blur rawStack")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # array2tif(flatStack,'flatStack.tif')
    # #flatStack = zStack2Mem('flatStack.tif')
    # corStack = correctImageStack(rawStack,zProjDark, flatStack)
    # array2tif(corStack,path + 'correctedStack.tif')
    #print(correctImage(raw[0],zProjDark,flat(0)))
    # This is a test of git moving:
