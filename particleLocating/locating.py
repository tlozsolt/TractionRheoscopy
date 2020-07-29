import trackpy as tp
import functools,yaml
from functools import partial
from particleLocating.paintByLocations import particleGlyph as pg
import numpy as np
import pandas as pd
import scipy
import pickle
from dask.distributed import Client
from dask import dataframe as ddf


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

def lsq_refine(df_loc, np_image, **refine_lsq_Meta):
    """
    """
    meta = refine_lsq_Meta
    fit_fun = locatingMetaData['refine_lsq']['fit_func']
    if fit_fun == 'disc':
        disc_size = locatingMetaData['refine_lsq']['disc_size']
        return tp.refine_leastsq(df_loc, np_image,
                                 diameter=diam,
                                 fit_function=fit_func,
                                 param_val={'disc_size':disc_size},
                                 compute_error=True)
    elif fit_fun == 'gauss':
        return tp.refine_leastsq(df_loc, np_image,
                                 diameter=diam,
                                 fit_function=fit_func,
                                 compute_error=True)

def lsq_refine_combined(df_loc, np_image, **refine_lsq_meta):
    """
    Refine the particle positions in df_loc using np_image as input
    and dictionary of keywords from yaml file

    The position refinement is least squares minimzation of both guassian and hat (or ``disk``) functions
    with parameters defined in **each** step of iterative particle locating.

    There are many layers of the dictionaries. refine_lsq_meta should have two keys:
    gauss and disc.

    This returns a combined dataframe with particle locations and all other computed features
    in addition to refined positions and uncertainties for both guassian adn disc functions. The
    only columns that are deleted are exact replicas such as frame or mass.
    """
    # compute the refined positions

    N = df_loc.shape[0]
    dN = 1500
    metaGlobal = refine_lsq_meta['global']
    metaIteration = refine_lsq_meta['iteration']
    refine_dtypes = refine_lsq_meta['refine_dtypes']
    computer = refine_lsq_meta['computer']
    daskParam = refine_lsq_meta['dask_resources'][computer]
    mat = refine_lsq_meta['material']

    #compute some simple optimal paramters for dask workers and npartitions
    if daskParam['memory-limit'] == '4Gb': npart = int(2*daskParam['nprocs'])
    elif daskParam['memory-limit'] =='2Gb': npart = int(daskParam['nprocs'])
    else:
        print("Mem limit {} is not either `4Gb` or `2Gb`\n".format(daskParam['memory_limit']))
        print("Setting number of partitions equal to nprocs in dask_resources")
        npart = int(daskParam['nprocs'])

    def ddf_refine(ddf_chunk,np_imageArray, **tpRefineKwargs):
        df_chunk = tp.refine_leastsq(ddf_chunk, np_imageArray, **tpRefineKwargs)
        return df_chunk

    # start the dask client
    if daskParam['ip'] != 'auto': client = Client(daskParam['ip'])
    else: client = Client()
    client.restart()
    df_refine = pd.DataFrame({})
    for n in range(0,N,dN):
        client.restart()
        print(n + dN,N, '{:.2%}'.format((n +dN)/N))
        ddf_loc = ddf.from_pandas(df_loc.loc[n:int(n + dN-1)],
                                  npartitions=npart)
        if mat == 'sed': df_chunk = ddf_loc.map_partitions(partial(ddf_refine,np_imageArray = np_image,
                                                                   **metaIteration['gauss']),
                                                           meta=refine_dtypes).compute()
        elif mat == 'gel': df_chunk = ddf_loc.map_partitions(partial(ddf_refine,np_imageArray = np_image,
                                                                     **metaIteration['disc']),
                                                             meta=refine_dtypes).compute()
        else: raise ValueError("Material {} is not 'sed' or 'gel'."
                               "Dont know if to refine with gauss or disc".format(mat))
        df_refine = pd.concat([df_refine,df_chunk])
        #df_disc = ddf_loc.map_partitions(partial(ddf_refine,np_imageArray = np_image,
        #                                          **metaIteration['disc']),
        #                                  meta=refine_dtypes['disc']).compute()

        # which columns are duplicated in the output
        # This is really more of a post processing step. I think I should just save df_gauss and df_disc to a file
        # and then merge the dataFrames, removing duplicate columns later.
        #col_duplicate = [key for key in df_loc if df_loc[key].equals(df_gauss[key])]
        #df_refinelsq = df_disc.drop(columns=col_duplicate).join(
        #    df_gauss.drop(columns=col_duplicate),
        #    lsuffix = '_lsqHat',rsuffix='_lsqGauss' )
        #df_loc = df_loc.join(df_refinelsq)
    return df_loc, df_refine


def iterate(imgArray, metaData, material, metaDataYAMLPath=None):
    """
    This function applies the locatingFunc to the imageArray iteratively following Kate's work.
    The basic idea is to locate some particles, create a mask to zero out the positions of the located particles
    and then run locating again on the (partially) zeroed imgArray. Also, concantenate the particle locations
    :param imageArry: numpy nd image stack that just needs to be located. Already decon and thresholded, upscaled etc
    :param paramDict: the dictionary in metaDataYAML under the key 'locating'
    :material: 'sed' or 'gel' depending on the material being located
    :metaDataYAMLPath: str, path to metaDataYAML directory. Used to import df_locMicro.pkl to precompute output datatypes
    :return: pandas data frame with particle locations and extra output from trackpy (ie mass, eccentricity, etc)
    """
    global imgArray_refine, combined_dict
    paramDict = metaData['yamlMetaData']['locating']
    locatingParam = paramDict[material]
    iterativeParam = paramDict['iterative']
    daskParam = metaData['yamlMetaData']['dask_resources']
    particleBool = True # did we find additional particle on this iteration?
    # maybe this should be done in the while loop to account for changing parameters during iteration
    locList = []
    refineList = []
    iterCount = 0
    maxIter = iterativeParam['maxIter']
    refineBool = paramDict['refine_lsq']['bool']

    if paramDict['refine_lsq']['bool']:
        # if we are refining during iterative locating, create a deep
        # copy of imgArray and refine on that.
        imgArray_refine = imgArray.copy()

        # create the complicated nest of input dictionaries included a computation of return dtypes
        # with keys global, iteration, refine_dtypes
        compute_error = paramDict['refine_lsq']['compute_error']

        # This is risky...refine makes decisions based on whether feature is anisotropic or not. In particular
        # it assume (reasonably) that if the locating was anisotropic, the refinement should be anisotropic as well
        # .., it will fail if you try it with iso locating and aniso refinement
        #dict_refine = {'gauss': {'diameter': (5,5,5), 'fit_function': 'gauss', 'compute_error': compute_error},
        #                'disc': {'diameter': (17,23,23), 'fit_function': 'disc' , 'compute_error': compute_error ,
        #                         'param_val': {'disc_size': 0.6}}
        #               }
        try:
            if material == 'sed': dict_refine = locatingParam[-1]['refine_lsq']['gauss']
            elif material == 'gel': dict_refine = locatingParam[-1]['refine_lsq']['disc']
        except KeyError:
            print("You are trying mix sed and gel refinement functions...thats not yet implemented")
            raise

        #with open(metaDataYAMLPath+'/df_locMicro.pkl', 'rb') as f:
        #    df_locMicro = pickle.load(f)
        #df_refine_micro_gauss = tp.refine_leastsq(df_locMicro,imgArray,**dict_refine['gauss'])
        #df_refine_micro_disc = tp.refine_leastsq(df_locMicro,imgArray,**dict_refine['disc'])
        #refine_dtypes = {'gauss': df_refine_micro_gauss.dtypes.apply(lambda x: x.name).to_dict(),
        #                 'disc' :  df_refine_micro_disc.dtypes.apply(lambda x: x.name).to_dict()}
        combined_dict = {'global' : paramDict['refine_lsq'],
                         #'refine_dtypes' : refine_dtypes,
                         'dask_resources': daskParam,
                         'computer': metaData['computer'],
                         'material': material
                        }
    refine_dtypes = None
    while particleBool == True and iterCount < maxIter:
        iterCount += 1

        # try moving down the list, until you cant...
        try:
            param_all = locatingParam[iterCount]
            locParam = {k:v for k,v in locatingParam[iterCount].items() if k != 'refine_lsq'}
        # when you cant move down the list, just use the last entry for the tail
        except IndexError:
            locParam = {k:v for k,v in locatingParam[-1].items() if k != 'refine_lsq'}
            param_all = locatingParam[-1]

        # create the locating function with partial application
        locateFunc = functools.partial(tp.locate,**locParam )
        print("Iteration: {}".format(iterCount))
        loc = locateFunc(imgArray).dropna(subset=['z']) # remove all the rows that have NAN particle positions
        loc['n_iteration'] = iterCount # add a new column to track what iteration the particle was located on.
        print("{} particles located!".format(loc.shape))
        if loc.shape[0] == 0: break

        # now refine the positions
        if refineBool:
            if refine_dtypes == None:
                # we have give some metaData on return types
                loc_micro = loc[0:2]
                df_refine_micro = tp.refine_leastsq(loc_micro, imgArray, **dict_refine)
                refine_dtypes = df_refine_micro.dtypes.apply(lambda x: x.name).to_dict()
                combined_dict['refine_dtypes'] = refine_dtypes
            print("Carrying out least squares particle refinement!")
            try: combined_dict['iteration'] = locatingParam[iterCount]['refine_lsq']
            except IndexError: combined_dict['iteration'] = locatingParam[-1]['refine_lsq']
            loc, loc_refine = lsq_refine_combined(loc, imgArray_refine, **combined_dict)

        # add to output
        locList.append(loc) # add the dataframe to locList and deal with merging later
        refineList.append(loc_refine)
        if not refineBool: mask = createMask(loc,imgArray,iterativeParam['mask'][material])
        else: mask = createMask(loc_refine, imgArray, iterativeParam['mask'][material])
        imgArray = imgArray*np.logical_not(mask)
    particleDF = pd.concat(locList).rename(columns={"x": "x_centroid (px)",
                                                    "y": "y_centroid (px)",
                                                    "z": "z_centroid (px)"})
    refineDF = pd.concat(refineList)
    logDict = {'locating' : {'particles': particleDF.shape[0],
                             'iterations': iterCount,
                             'refine_lsq': paramDict['refine_lsq']['bool']}}
    return [particleDF, refineDF, logDict]

def createMask(locDF, imgArray, glyphShape,refineBool=True):
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

    # these coordinates might need to be shifted to not get IndexErrors
    coord = [zCoord,yCoord,xCoord]
    shiftCoord = []
    for n in range(len(coord)):
        if coord[n] > imgMask.shape[n]:
            # Note this will always shift to be in bounds, but the shift is arbitrarily large.
            # I am really assuming that the shifts are small, 1-2px, and rare.
            shiftCoord[n] = imgMask.shape[n] - 1
        elif coord[n] < 0: shiftCoord[n] = 0
        else: shiftCoord[n] = coord[n]

    # assign shifted coordinates to prevent IndexErrors
    imgMask[shiftCoord[0], shiftCoord[1], shiftCoord[2]] = 1

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
