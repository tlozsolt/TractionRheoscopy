from particleLocating.flatField import zStack2Mem
from particleLocating.flatField import zProject
from particleLocating.flatField import array2tif
import numpy as np

def computeMasterDark(rawDarkPath: str, rawDarkFrmt: str, outDir: str, fileN: int=None,  verbose: str = 'False'):
    """
    Computes a masterDark frame from a time series of dark frames
    saved under rawDarkPath with fileName rawDarkFrmt.
    fileN gives the number of time stacks but time is likely an entire stack of
    frames and not an individual frame.

    Computation is done iterativly by loading the files in the timeSeries, computing the
    average per pixel along with number of z-slices and then computing the final average at the end.
    This in principle makes it possible to parallelize with multiple nodes, but that is
    not yet implemented.

    rawDarkPath: str, example '/Volumes/Project/zsolt/tfrgel23042022/calibration/darkTiff
    rawDarkFrmt: str, example '/tfrGel23042022_65ms_t{:05d}.tif' # note that t{:05d} is format
                      string for left zero padded to total length 5 digit and can be called
                      with rawDarkFrmt.format(n) for time point n

    Zsolt, May 2022
    """
    # initialize list of dictionary containig stackNumber: z-slices
    avgFrameDict = {}

    # initialize dictionary of stackNumber: avgFrame
    zSliceDict = {}

    # get number stacks by file matching in directory
    if fileN is None:
        raise NotImplementedError('You must provide number of stacks. No file matching on directory is implemented Zsolt May 2022')

    # loop over stack numbers 0 to fileN
    for t in range(fileN):
        # format string to get full path to tiff stack
        fName = rawDarkPath + rawDarkFrmt.format(t)

        # load stack into memory
        if verbose: print('Loading frame {}'.format(t))
        stack = zStack2Mem(fName)

        # get stack dimensions in z
        zSlices = stack.shape[0]
        if verbose: print('{} slices loaded from time {}'.format(zSlices, t))
        #print(zSlices) # double check zyx indexing on import

        # compute per pixel average with some fast numpy function
        # already written!
        avgFrame = zProject(stack)

        # save to avgFrameDict and zSliceDict
        key = str(t)
        avgFrameDict[key] = avgFrame
        zSliceDict[key] = zSlices

    # weight each partial average by fractional count of frames
    N = sum(zSliceDict.values())
    for key, value in zSliceDict.items(): avgFrameDict[key] = value/N * avgFrameDict[key]

    # straight sum over pixel values
    masterDark = np.zeros(avgFrameDict['0'].shape)
    for key, value in avgFrameDict.items():
        masterDark = masterDark + value

    # write masterDark frame to output directory
    array2tif(masterDark, outDir)

    return zSliceDict, avgFrameDict, masterDark

# at the end of the for loop, compute weighted average of the
# frames with weights assigned by zSlices/total


if __name__ == '__main__':
    calibrationPath = '/Volumes/PROJECT/calibration/24MAY2022/darkImage_calibration_exp55ms_camera292320220523_155253_20220523_162439'
    frmtString = '/darkImage_calibration_exp55ms_camera292320220523_155253_t{:06d}.tif'
    outDir = '/Volumes/PROJECT/calibration/darkTiff/DEBUG_24MAY2022_zyla2923_55ms.tif'
    N = 61

    zSliceDict, avgFrameDict, masterDark = computeMasterDark(calibrationPath, frmtString, outDir,N, verbose=True)

