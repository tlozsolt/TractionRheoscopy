import cv2
import numpy as np
import flatField
import yaml
from joblib import Parallel, delayed

def resize(slice,paramDict):
    """
    Resizes a slive to the desired output size.
    :param slice: single slice from ndarray with 32bit float dtype, assumed to be the output from decon or smartCrop
    :param paramDict: read from metaData yaml file.
    :return: rescaled image, possibly a dictionary of
    """

    out_size = (slice.shape[0]*paramDict['dim']['y'], \
                slice.shape[1]*paramDict['dim']['x'])

    if paramDict['interp_method'] == 'lanczos':
        return cv2.resize(slice, out_size, interpolation = cv2.INTER_LANCZOS4)
    elif paramDict['interp_method'] == 'cubic':
        return cv2.resize(slice, out_size, interpolation = cv2.INTER_CUBIC)
    else:
        print("interpolation method {} is not recognized. Choose \'lanczos\' or \'cubic\' ".format(paramDict['interp_method']))
        raise ValueError

def resize_stack(stack,paramDict):
    if paramDict['parallel']['bool'] == True: n_jobs = paramDict['parallel']['n_jobs']
    else: n_jobs=1
    return np.array(Parallel(n_jobs=n_jobs)(delayed(resize)(stack[z,:,:],paramDict) for z in range(stack.shape[0])))

if __name__ == "__main__":
    imgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2'\
              '/OddysseyHashScripting/decon/decon_rawOutput.tif'
    yamlPath = '/Users/zsolt/Colloid/SCRIPTS/tractionForceRheology_git'\
               '/TractionRheoscopy/metaDataYAML'\
               '/tfrGel09052019b_shearRun05062019i_metaData_scriptTesting.yaml'

    # read in the image stack
    stack = flatField.zStack2Mem(imgPath)

    # read in the parameter dictionary from yaml file
    with open(yamlPath, 'r') as stream: metaData = yaml.load(stream, Loader=yaml.SafeLoader)
    paramDict = metaData['postDecon']['upscaling']

    # resize the image
    resizedImg = resize_stack(stack,paramDict)

    # visulize the stack
    import pyFiji
    from threshold import arrayThreshold

    testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/'\
                  'OddysseyHashScripting/pyFiji/testImages'
    resized8bit = arrayThreshold.recastImage(resizedImg, 'uint8')
    pyFiji.send2Fiji(resized8bit,wdir=testImgPath)




