import numpy as np
import flatField
import dplHash_v2 as dpl

"""
Inputs 
- folder location of images to stitch and some framework to extract hashValue from fileName 
- hashTable that was used to generate these images 
- file with hashValue and pixel translation vectors
  -- maybe this file should be created at the end of **every** image processing step in the same format even if the
     images for that processing step are never stitched and the check for completeness of all hashValues should be the 
     same as compiling many individual files into a single file that of (hashValue: translation vector) pairs. 
- folder with files of particle location and some framework to extract hashValue from fileName. 
- output directory for stitched images, cross sections, etc 
- output for particle locations (with or without double hits removed)

Steps:
- given a timestep, get a list of the required hashValues
-initalize a stitched Stack
- for each hashValue, get the translation Vector in pixels
- update the pixel values in the stitched image after translating the origin by the translation vector
- continue for all hashValues in the given timestep
- save tiff stack
- maybe output the cross section slices to tiff as well

What am I going to do if the hashValue chunks overlap? 
Stupid: who cares, make sure crop out obvious garbage and then just overwrite the pixel values 
Simple: if there is an existing nonzero pixel value, avg the values
Complicated: compute the nnb hashValue and implement a specified matching function for the overlap region 
"""

def getHashNNB(hashValue,rasterParam):
    """
    This a complicated function that will return the nnb hashValues for a speicfied hashValue. It handles all edge
    cases and whether the material is gel or sediment and if those need to be stitched different. That is to say, the
    output vector of nnb hash value has three types of value: number, different material, or edge.

    This is complicated discrete function of the rasterParam (how many x and y chunks in plane and how many z chunks total)
    but note that it is translation invariant and there is a lot of symmetry in the edge cases.

    Actually, who cares about "computing" this function when it is **almost** tabluated when the hash was computed.
    To get the table, I just need to keep track of the dummy indices used to count when cropping in x, y, and z.
    These will be (i,j,k) values and the nnb are just (i+1,j,k),(i-1,j,k),(i, j+1,k),... etc
    I also probably only need the 6 face sharing nnb but maybe this should an option.

    returns a 26-tuple of nnb hashes with the possiblity that some entries are empty if the hashValue is an edge case
    :param hashValue:
    :return:
    """

def readLog(hashValue):
    """
    Read the log file for a given hashValue
    return a dictionary of the pipeline steps as keys
    """


def loadImage():
    """ read image from fileName path into numpy array
        return the array
    """
    return True

def getTranslationVector(hashValue, step = "postDecon"):
    """
    for a given hashValue, get the translation vector after specified processing step
    note that the translation vector can in principle depend on any cropping done on the image
    during image processing...in fact the cropping only needs to keep track of the updated translation vector
    as the image dimensions can be determined on the fly.

    :param hashValue: number of hashValue
    :return: translation vector (x,y,z) to add to (0,0,0) coordinate of numpy array
    """
    return True

def missingHV():
    """
    returns True if no missing hashValues; otherwise returns a list of missing hashValues.
    Flow control is such that I am going to call one, possibly batch, job to stitch and reslice **all** the
    stacks even if that sacrifices time in which I am waiting for a single hashValue to complete while all other
    jobs are done.
    :return:
    """
    return False

def initStitchedImage():
    """
    initiliaze a numpy array with the correct dimensions for the final stitched stack
    options and control:
        - gel, sed, or all
        - default dtype should be 8bit tiff
    :return: initialized array
    """


