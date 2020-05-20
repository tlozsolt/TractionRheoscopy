import numpy as np
import math
import pandas as pd

"""
-> take in a dataFrame of ground truth atomic positions 
-> generate pixel grid
-> get closest pixel to ground truth position
-> add gaussian thermal fluctuations

Analysis:
-> compute structure 
-> compute defects
-> compute strain using Falk-Langer.

Questions:
-> how much does the computed structure FCC/HCP/Other fluctuate with pixel bias
-> At what point does pixel bias swamp out strain
-> How much does defect identification fluctuate with pixel bias?
    -> Do you transiently locate defects if pixel bias is too large
    -> Do dislocation types fluctuate with pixel bias along?
-> What fraction of strain fluctuations are caused by pixel bias?
-> What if both the current state and reference configuration are skewed by pixel bias? 
    -> I would guess the inaccurate reference configuations would lead to persistantly wrong
       strain measurements over time.
"""

def makePixelGrid(dim_min=np.array([0,0,0]),dim_max=np.array([20,20,20]), px2Micron=np.array([0.15,0.23,0.23])):
    """
    :param px2Micron: The physical dimensions of a pixel in microns in zyx coordinates
    :param dim: The dimensions of the grid in um

    :return: an array of values
    """
    return np.array([np.arange(dim_min[0], dim_max[0], px2Micron[0]),\
                     np.arange(dim_min[1], dim_max[1], px2Micron[1]),\
                     np.arange(dim_min[2], dim_max[2], px2Micron[2])])


def find_nearest(array,value):
    """
    Find the nearest entry in array to input value

    Copied from stack overflow:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def addPixelBiasing(locDF,px2Micron=np.array([0.15, 0.23, 0.23]), subPixelRatio = np.array([1, 1, 1]),column_str=' (um, imageStack)'):
    """
    Takes in a location dataFrame and returns columns for particle positions (in microns)
    after scrambling location by synthetic pixel biasing

    :param locDF: location of particle positions
    :param px2Micron: zyx, physical size of pixels in um
    :param subPixelRatio: how much subpixel accuracy do you want?
                        -> ratio = 1: nearest pixel accuracy
                        -> ratio = 0.1: nearest tenth of a pixel scrambling
                        -> ratio = 2: super pixel binning and then scramble to superpixel
    :return: location dataFrame with new columns on scrambled locations
    """
    xPos = locDF['x'+ column_str].values
    yPos = locDF['y'+ column_str].values
    zPos = locDF['z'+ column_str].values
    dim_max = [zPos.max(), yPos.max(), xPos.max()]
    dim_min = [zPos.min(), yPos.min(), xPos.min()]

    px_Array = makePixelGrid(dim_min=dim_min,dim_max=dim_max, px2Micron = px2Micron*subPixelRatio)

    locDF['z (um, pxBiasSramble)'] = np.array([find_nearest(px_Array[0],elt) for elt in zPos])
    locDF['y (um, pxBiasSramble)'] = np.array([find_nearest(px_Array[1],elt) for elt in yPos])
    locDF['x (um, pxBiasSramble)'] = np.array([find_nearest(px_Array[2],elt) for elt in xPos])
    return locDF

if __name__ == "__main__":
    import synthetic_fcc as synthFCC
    fcc = synthFCC.stackSequence('ABC'*5,dim=25)
    fcc_background = synthFCC.stackSequence('ABC'*5,dim=25)
    fcc_no_core = fcc_background[(fcc_background['x'] >  14) |
                                 (fcc_background['x'] < -14) |
                                 (fcc_background['y'] >  14) |
                                 (fcc_background['y'] < -14)]
    stackingFaultCore = synthFCC.stackSequence('ABCABABCBCACABC')
    merge_noDoubles = synthFCC.weld(fcc_no_core, stackingFaultCore,1.55)
    addPixelBiasing(merge_noDoubles, column_str='',subPixelRatio=[0.2,0.2,0.5])
    merge_noDoubles.to_csv('/home/zsolt/buf/stackingFaults_PixelBias_x2_y2_z5.csv', sep=' ',
               columns=['x (um, pxBiasSramble)', 'y (um, pxBiasSramble)', 'z (um, pxBiasSramble)'])
    #merge_noDoubles.to_csv('/home/zsolt/buf/stackingFaults_synthetic.csv', sep=' ',
    #           columns=['x', 'y', 'z'])
    print(fcc.info())

