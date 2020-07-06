"""
Curvature filters copied from
YuanhaoGong/ CurvatureFilter

These are the same 2D curvature filters implemented in the Mosaic package for Fiji.

Cite:
@ARTICLE{gong:cf,
    author={Yuanhao Gong and Ivo F. Sbalzarini},
    journal={IEEE Transactions on Image Processing},
    title={Curvature filters efficiently reduce certain variational energies},
    year={2017},
    volume={26},
    number={4},
    pages={1786-1798},
    doi={10.1109/TIP.2017.2658954},
    ISSN={1057-7149},
    month={April},}

"""
import numpy as np
import flatField
from joblib import Parallel,delayed
from scipy import ndimage
from skimage import exposure
import threshold, pyFiji

def update_Bern(img, row, col):
    img_ij = img[row:-1:2, col:-1:2]
    img_prev = img[row - 1:-2:2, col:-1:2];
    img_next = img[row + 1::2, col:-1:2]
    img_left = img[row:-1:2, col - 1:-2:2];
    img_rigt = img[row:-1:2, col + 1::2]

    d1 = (img_prev + img_next) / 2.0 - img_ij;
    d2 = (img_left + img_rigt) / 2.0 - img_ij
    d_m = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    img_ij[...] += d_m


def update_MC(img, row, col):
    img_ij = img[row:-1:2, col:-1:2];
    img_ij8 = 8 * img_ij
    img_prev = img[row - 1:-2:2, col:-1:2];
    img_next = img[row + 1::2, col:-1:2]
    img_left = img[row:-1:2, col - 1:-2:2];
    img_rigt = img[row:-1:2, col + 1::2]
    img_leUp = img[row - 1:-2:2, col - 1:-2:2];
    img_riUp = img[row - 1:-2:2, col + 1::2];
    img_leDn = img[row + 1::2, col - 1:-2:2];
    img_riDn = img[row + 1::2, col + 1::2];

    d1 = 2.5 * (img_prev + img_next) + 5.0 * img_rigt - img_riUp - img_riDn - img_ij8;
    d2 = 2.5 * (img_prev + img_next) + 5.0 * img_left - img_leUp - img_leDn - img_ij8;
    d3 = 2.5 * (img_left + img_rigt) + 5.0 * img_prev - img_leUp - img_riUp - img_ij8;
    d4 = 2.5 * (img_left + img_rigt) + 5.0 * img_next - img_leDn - img_riDn - img_ij8;

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))

    d /= 8

    img_ij[...] += d


def update_GC(img, row, col):
    img_ij = img[row:-1:2, col:-1:2];
    img_prev = img[row - 1:-2:2, col:-1:2];
    img_next = img[row + 1::2, col:-1:2]
    img_left = img[row:-1:2, col - 1:-2:2];
    img_rigt = img[row:-1:2, col + 1::2]
    img_leUp = img[row - 1:-2:2, col - 1:-2:2];
    img_riUp = img[row - 1:-2:2, col + 1::2];
    img_leDn = img[row + 1::2, col - 1:-2:2];
    img_riDn = img[row + 1::2, col + 1::2];

    d1 = (img_prev + img_next) / 2.0 - img_ij;
    d2 = (img_left + img_rigt) / 2.0 - img_ij
    d3 = (img_leUp + img_riDn) / 2.0 - img_ij;
    d4 = (img_leDn + img_riUp) / 2.0 - img_ij
    d5 = (img_prev + img_leUp + img_left) / 3.0 - img_ij;
    d6 = (img_prev + img_riUp + img_rigt) / 3.0 - img_ij
    d7 = (img_leDn + img_left + img_next) / 3.0 - img_ij;
    d8 = (img_rigt + img_riDn + img_next) / 3.0 - img_ij

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d5)) + d5 * (np.abs(d5) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d6)) + d6 * (np.abs(d6) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d7)) + d7 * (np.abs(d7) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d8)) + d8 * (np.abs(d8) < np.abs(d))

    img_ij[...] += d


def update_TV(img, row, col):
    img_ij = img[row:-1:2, col:-1:2];
    img_ij5 = 5 * img_ij
    img_prev = img[row - 1:-2:2, col:-1:2];
    img_next = img[row + 1::2, col:-1:2]
    img_left = img[row:-1:2, col - 1:-2:2];
    img_rigt = img[row:-1:2, col + 1::2]
    img_leUp = img[row - 1:-2:2, col - 1:-2:2];
    img_riUp = img[row - 1:-2:2, col + 1::2];
    img_leDn = img[row + 1::2, col - 1:-2:2];
    img_riDn = img[row + 1::2, col + 1::2];

    d1 = img_prev + img_next + img_left + img_leUp + img_leDn - img_ij5;
    d2 = img_prev + img_next + img_rigt + img_riUp + img_riDn - img_ij5;
    d3 = img_left + img_rigt + img_leUp + img_prev + img_riUp - img_ij5;
    d4 = img_left + img_rigt + img_leDn + img_next + img_riDn - img_ij5;
    d5 = img_leUp + img_prev + img_riUp + img_left + img_leDn - img_ij5;
    d6 = img_leUp + img_prev + img_riUp + img_rigt + img_riDn - img_ij5;
    d7 = img_leDn + img_next + img_riDn + img_rigt + img_riUp - img_ij5;
    d8 = img_leDn + img_next + img_riDn + img_left + img_leUp - img_ij5;

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d5)) + d5 * (np.abs(d5) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d6)) + d6 * (np.abs(d6) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d7)) + d7 * (np.abs(d7) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d8)) + d8 * (np.abs(d8) < np.abs(d))

    d /= 5

    img_ij[...] += d


def CF(inputimg, filterType=2, total_iter=10):
    """
    This function applies Curvature Filter on input image for 10 iterations (default).

    Parameters:
        inputimg: 2D numpy array that contains image data.
        filterType: indicate with filter to use, GC filter by default
        total_iter: number of iterations, default is 10.

    Return:
        2D numpy array, the input image is not modified.
    """
    outputimg = np.copy(inputimg)
    localFunc = {
        0: update_TV,
        1: update_MC,
        2: update_GC,
    }
    update = localFunc.get(filterType)
    for iter_num in range(total_iter):
        # four sets from domain decomposition
        update(outputimg, 1, 1)
        update(outputimg, 2, 1)
        update(outputimg, 1, 2)
        update(outputimg, 2, 2)
    return outputimg

def tvFilter_stack(stack,iter=10,n_jobs=16):
    """
    total variation curvature filter in 2D
    Simple wrapper function to convert data type to 32 bit float
    and apply iteration of total variation without having to
    remember that totalVariation is filter type 0.
    Also, this is parallel for each slice.
    :param stack: numpy array
    :param iter: number of iterations
    :param n_jobs: number of threads in joblib
    :return: filtered stack, possbily recast as 16bit
    """
    #convert to float
    stack = stack.astype('float32',copy=True)
    parOut = Parallel(n_jobs=n_jobs)(delayed(CF)(stack[z,:,:],filterType=0,total_iter=iter)\
                                     for z in range(stack.shape[0]))
    return np.array(parOut)

def mcFilter_stack(stack, iter=10, n_jobs=16):
    """
    mean curvature filter in 2D applied slice by slice to stack
    This should be used to rgualrize for surface that are minimal (like a sphere, not like a cylinder)
    Gaussian Curvature will regualrize fro surface that are developable (can be unwrapped to plane w/o distortion)
    total variation will yield piece wise constant.

    :param stack:
    :param iter:
    :param n_jobs:
    :return:
    """
    stack = stack.astype('float32',copy=True)
    parOut = Parallel(n_jobs=n_jobs)(delayed(CF)(stack[z,:,:],filterType=1,total_iter=iter) \
                                     for z in range(stack.shape[0]))
    return np.array(parOut)

def gcFilter_stack(stack, iter=10, n_jobs=16):
    """
    mean curvature filter in 2D applied slice by slice to stack
    This should be used to rgualrize for surface that are minimal (like a sphere, not like a cylinder)
    Gaussian Curvature will regualrize fro surface that are developable (can be unwrapped to plane w/o distortion)
    total variation will yield piece wise constant.

    :param stack:
    :param iter:
    :param n_jobs:
    :return:
    """
    stack = stack.astype('float32',copy=True)
    parOut = Parallel(n_jobs=n_jobs)(delayed(CF)(stack[z,:,:],filterType=1,total_iter=iter) \
                                     for z in range(stack.shape[0]))
    return np.array(parOut)

def gaussBlur_stack(stack,sigma=1,n_jobs=4):
    """
    simple wrapper on gauss blur from numpy
    :param stack:
    :param sigma:
    :param n_jobs:
    :return:
    """
    # convert to float
    stack = stack.astype('float32')
    #ndimage.gaussian_filter(slice, sigma)
    parOut = Parallel(n_jobs=n_jobs)(delayed(ndimage.gaussian_filter)(stack[z,:,:],sigma=sigma)\
                                     for z in range(stack.shape[0]))
    return np.array(parOut)

def equalize_adaptHist_stack(stack,clip_limit=0.03,n_jobs=4):
    stack = stack.astype('uint16')
    parOut = Parallel(n_jobs=n_jobs)(delayed(exposure.equalize_adapthist)(stack[z,:,:],clip_limit=clip_limit)\
                                     for z in range(stack.shape[0]))
    return np.array(parOut)

if __name__ == '__main__':
    testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
                  'OddysseyHashScripting/pyFiji/testImages'
    inputImgPath ='/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/' \
                  'postDecon/tfrGel09052019b_shearRun05062019i_postDecon8Bit_hv00002.tif'
    img = flatField.zStack2Mem(inputImgPath)
    #img = img.astype('uint16')
    #tvFiltered = threshold.arrayThreshold.recastImage(tvFilter_stack(img,iter=50),'uint16')
    adaptEqualize = threshold.arrayThreshold.recastImage(equalize_adaptHist_stack(img,clip_limit=0.03),dtypeOut='uint8')
    #equalizeSlice = exposure.equalize_adapthist(img[75],clip_limit=0.03)
    #equalizeSlice = threshold.arrayThreshold.recastImage(equalizeSlice,'uint16')
    print(pyFiji.send2Fiji([img,adaptEqualize],wdir=testImgPath))


