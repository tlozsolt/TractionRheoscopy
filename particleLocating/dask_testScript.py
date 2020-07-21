import dask_image.imread
import dask_image.ndfilters
import dask_image.ndmeasure
import dask.array as da
import cv2
from functools import partial
import numpy as np
import pandas as pd
from particleLocating import locating as tp_locating
from particleLocaint import threshold
from scipy import ndimage


#%%
def resize(array):
    """
    array has shape (N,450,450)
    to make it work with cv2.resize we need to:
    -transpose
    -send to cv2.resize
    -transpose back
    and apparently dask will handle the rest of the parallelization?
    """
    slice = array[...,:,:].transpose().compute()
    upscale = cv2.resize(slice,(900,900),interpolation=cv2.INTER_LANCZOS4)
    return upscale.transpose()

#%%
def mean2D(block):
    """
    This will work with map_block for a 2D chunk.
    Note the important aspects that amount to a wrapper:
    -you have to return an array, not a number
    -the array has to have the same number of dimensions (but not the same dimensions)
    as the block itself. This is the meaning of np.array([the answer])[:,None]
    where the final part [:,None] uses numpy slicing to take all the answer in the first
    dimensions and add a None type second dimension
    np.array([1,2,3]) has shape (3,)
    np.array([1,2,3)[:,None] has shape (3,1)
    """
    return np.array(np.mean(block))[:,None]
#%%
def _cropFFT(input_arr, cropDim_zyx):
    """
    Crops out FFT artifacts in input_arr and produces output_arr
    The dimension of what is cropped is passed cropDim
    """
    dim = np.array(input_arr.shape)
    #output_arr = np.empty(dim-cropDim_zyx)
    return input_arr[
           cropDim_zyx[0] : dim[0] - cropDim_zyx[0], \
           cropDim_zyx[1] : dim[1] - cropDim_zyx[1], \
           cropDim_zyx[2] : dim[2] - cropDim_zyx[2]]
#%%
# exammple function that takes a dask array, averages a chunk and returns the avg value and the array location
# of the block center
#def avgBlock3D(block): return np.array([np.mean(block)])[:,None,None]
def avgBlock3D(block,block_info=None):
    #print(block_info[None])
    return np.array([np.mean(block)])[:,None,None]

def getLocation(block, block_info=None):
    print(block_info[None]['array-location'])
    return block

from functools import partial
from particleLocating import curvatureFilter
from scipy import ndimage
from particleLocating import locating

tvFilter = partial(curvatureFilter.CF,filterType=0,total_iter=10)
mcFilter = partial(curvatureFilter.CF,filterType=1,total_iter=1)
gaussBlur = partial(ndimage.gaussian_filter,sigma=1)

def tvFilter_dask(daskArray):
    slice = daskArray[:,:,:].transpose().compute()
    print(daskArray.shape)
    filteredSlice = tvFilter(slice)
    return filteredSlice.transpose()

def tvFilter_squeeze(daskArray):
    """
    No use of transpose
    This should really be a decorator...I want something like:
    """
    slice = daskArray.squeeze()
    filteredSlice = tvFilter(slice)
    return filteredSlice[None,...]

def loc_partial(paramDict, material):
    return partial(tp_locating.iterate, paramDict=paramDict, material= material)

loc_global = pd.DataFrame(columns=['x','y','mass', 'iter'])
loc_global.fillna(0)
def iterateLocate(daskArray, paramDict=None, material='sed', loc_global = None):
    """
    THis is a first attempt at particle locating on dask array
    Maybe what this should do is write the location dataFrame to a commpon output just by concatenation
    """
    f = partial(locating.iterate,paramDict=paramDict,material=material)
    locDF, logDict = f(daskArray).compute()
    loc_global = pd.concat([loc_global,locDF])
    return daskArray


if __name__ == "__main__":
    # Start from tif output of decon and go serial or numpy compute as needed from a dask array
    # steps are:
    # [ ] read in meta data info
    from particleLocating import dplHash_v2 as dpl
    yamlMetaData = '/Users/zsolt/Colloid_git/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'
    dplInst = dpl.dplHash(yamlMetaData)
    hashValue = 0
    compute = 'IMAC'
    deconPath = dplInst.getPath2File(0,kwrd='decon',computer='IMAC')
    # [X] read in the dask image array
    #postDecon_da = dask_image.imread.imread(deconPath)
    # [+] smart crop
    # dask, this basically work out of the box, but note that images are read in with smartCrop
    # using dask_image.imread.imread(<path/).compute()...which is no different that the stanard method.
    # and there is a small bug currently with smartCrop applied to tilted interfaces.
    smartCrop_da, smartCrop_log = dplInst.smartCrop(hashValue,computer=computer, output='dask_array')
    smartCrop_da = da.from_array(smartCrop_da,chunks=(1,smartCrop_da.shape[1],smartCrop_da.shape[2]))

    # [ ] postDecon
    #     [ ] threshold: slightly tricky because the output is not an array. Maybe run threaded computation
    #         to global parameter that gets fed into interpolate?
    #     -> Covert to 16 bit using recast
    #     -> make maxEntropyThreshold(stack) a static method
    #     -> this is very close, I just need to carry out the interpolation using the same
    #        grid data thing but with logical indexing.
    #%%
    def maxEntropyThreshold(stack):
        """
        Computes the maximum entropy threshdold from image histogram as implemented in Fiji > Threshold > MaxEnt

        This follows:

        Reference:
        Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
        Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
        Graphics, and Image Processing 29, no. 3 (1985): 273–285.

        and kapur_threshold() function in pythreshold package.

        :param stack:
        :return:
        """
        hist, _ = np.histogram(stack, bins=range(2**16),density=True)
        c_hist = hist.cumsum()
        c_hist_i = 1.0 - c_hist

        # To avoid invalid operations regarding 0 and negative values.
        c_hist[c_hist <= 0] = 1
        # I think this is a logical index on the boolean expression: if c_hist<=0, set that value to 1
        c_hist_i[c_hist_i <= 0] = 1

        c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum() # add logical array hist<=0 to make sure you dont take log(0)
        b_entropy = -c_entropy / c_hist + np.log(c_hist)

        c_entropy_i = c_entropy[-1] - c_entropy
        f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

        return np.argmax(b_entropy + f_entropy)

    def maxEnt(chunk):
        out = np.empty(chunk.shape, dtype='float32')
        out[:] = np.nan
        cz, cy, cx = np.array((np.array(out.shape) - 1) / 2).astype(int)
        chunk1D = np.ndarray.flatten(chunk)
        out[cz, cy, cx] = np.array([maxEntropyThreshold(chunk1D)])[..., None, None]
        return out

    def applyThreshold(imgArray, thresholdArray, recastBool = True,scaleFactor=1.0):
        """
        This function does not compute a threshold. It just takes imgArray and thresholdArray
        and outputs an 16 bit image of the threshold with optional recasting to image to 16 bit depth.
        :return:
        """
        # change type to enable subtraction
        out = imgArray.astype('float32') - scaleFactor*thresholdArray.astype('float32')
        # clip everything below zero
        positive = out # make a deep copy in case we also want to return the thresholded parts.
        negative = out * -1
        positive[positive<0] = 0 # now use logical indexing to reassign all negative values to zero
        negative[negative < 0] = 0
        if recastBool == True:
            positive = threshold.arrayThreshold.recastImage(positive,'uint16') # rescale the dynamic range after thresholding to span 16bits
            negative = threshold.arrayThreshold.recastImage(negative, 'uint16')  # rescale the dynamic range after thresholding to span 16bits
        return positive,negative

    #%%
    thresh_compute = smartCrop_da.map_overlap(maxEnt,depth=(5,5,5),dtype='float32',boundary='reflect').compute()
    interp_indices = np.argwhere(~np.isnan(thresh_compute))

    values = thresh_compute[~np.isnan(thresh_compute)]
    indices = np.argwhere(~np.isnan(thresh_compute))
    points = (indices[:, 0], indices[:, 1], indices[:, 2])
    # note that I have to use nearest interpolation as the dask chunks chunk centers are not on the edges...
    # ... but then again if this was mirrored the edges would give the same values as an interior interpolation
    # other option is to fill with avg of values as opposed to nan
    threshold_array = griddata(points, values, (zz, yy, xx), method='linear', fill_value = np.mean(values)).astype('uint16')
    aboveThresh, belowThresh = applyThreshold(smartCrop_da.compute(),threshold_array)
    # Now clean up garbage....

    #%%
    #     [+] curvature filters: already done with dask array and map blocks
    #     [+] upscale: already done with dask array and map blocks.
    # [ ] locate iterative
    # [ ] Visualize
    #
    # [ ] form dask array from np array and chunk into z-slices
    path = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/smartCrop/tfrGel10212018A_shearRun10292018f_smartCrop_hv00000.tif'
    pos16bit_dask = dask_image.imread.imread(path)
    #pos16bit = np.random.randint(1,12000,size=(25,45,45),dtype='uint16')
    #pos16bit_dask = da.from_array(pos16bit, chunks=(1,45,45))
    # [ ] apply cv2.resize to get upscaled image
    #%%
    upscaled_dask = da.from_array(resize(pos16bit_dask),chunks=(1,900,900))
    print("Upscaling complete!")
    # note resize currently scales to 900 as a hard coded value
    # [ ] apply curvature filters

    #%%
    #[+] Guassian blur
    a = gaussBlur(upscaled_dask)
    print('blur complete')
    #     [ ] total variation
    # This variation does not need transose which is the right way to think about it
    # I also think we can get away with lazy computation and composing the functions if I leave off compute() suffix
    # Its not clear that any of the modifications to curvature filters besides assignment += and /= mattered
    a = a.map_blocks(tvFilter_squeeze, dtype='float32').compute()
    print('tv filter complete')
    #     [ ] Mean Curvature
    a = mcFilter(a)
    print('mean curvature complete')
    # I have to check that this actually give me the results I am expecting. Very fast though
    # Note also that these commands need to be chained in order to give the result we care about
    # as currently nothing that is computed is saved
    # Also, the format should be to form the correct partial function from the yaml metaData file
    # presumably with a double splat dict expansion applied to a wrapper function that is built to
    # handle the naming conventions in the yaml metaData keys. All parameters should be kwarg except
    # the array.
    # [X] run tp.locate on dask array
    #     -> Do we really need to do this? Its not really an array operation and I dont know the output dimensions
    #        beforehand.
    #     -> It is a convenient idea though. Rechunk with overlaps larger than that particle search diameter and carry
    #        out particle locating on the chunks. Compile the particle positions at the end.
    #     -> If that worked, we could even run all the image analysis, stitch together the images, form a dask array
    #        and run the particle locating without having to deal with double hits or missed particles
    import trackpy as tp

    # decon
    #from particleLocating import dplHash_v2 as dpl
    import dask_image

    #yamlMetaData = '/Users/zsolt/Colloid_git/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'
    #hv = 0
    #inst = dpl.dplHash(yamlMetaData)
    #postDecon_arr = inst.smartCrop(0, computer='IMAC', output='dask_array')
    #a6x6 = da.from_array(np.arange(6 ** 3).reshape((6, 6, 6)), chunks=3)
    #a6x6.map_blocks(avgBlock3D, chunks=(1, 1, 1)).compute()
    #a6x6.map_blocks(getLocation, chunks=(3, 3, 3), dtype=np.int).compute()

    #from particleLocating import threshold

    # rechunk to volumes
    #threshold = postDecon_arr.rechunk(chunks=(50, 100, 100))

    # read in decon files
    #deconOutPath = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/tfrGel10212018A_shearRun10292018f_decon_hv00003.tif'
    #array = dask_image.imread.imread(deconOutPath,chunks=1)
    # apply smart crop (this needs to be rewritten to remove the requirement of reading from file, or add reading file
    # using dask image directory to smartCrop as an option)

    # map overlap to compute threshold
    # rechunk to make z-slices
    # resize to get correct xy size
    # rechunk to make x-slice
    # asym resize to get correction yz size (I think this should work)
    # rechunk into z-slices and subtract disk arrays to compute threshold
    # recast image
    # upscale across z-slices
    # rechunks into volumes
    # call trackpy on the  volumetric chunks
    # I think this should return particle positions with the correct coordinates specific to each chunk
    # form a dask array for the visualization using oaconvolve from particle positions and glyph
    # save visualization hdf5 as three channels (raw, locInput, locations) and visualize with napari

    # - convert to dask array and chunk in z
    # - form a dask array and chunk into overlapping regions
    # recast the image as uint16 for thresholding
    # - compute threshold
    # - apply threshold...maybe we need to compute at every point if it so fast.
    # or...scale up the resolution of the image with cv2.resize
    # - rechunk into slices
    # - upscale slices
    # - rechunk into overlapping regions with overlap region given by particle size keeping in mind the upscaling
    # - tp.locate apply to all the chunks
    # The goal is not use to multiprocessing at all and just chain the outputs.




