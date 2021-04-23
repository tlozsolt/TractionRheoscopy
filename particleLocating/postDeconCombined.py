from particleLocating import dplHash_v2 as dpl
from particleLocating import locating, paintByLocations, flatField, pyFiji
from particleLocating import threshold
from particleLocating import curvatureFilter
from particleLocating import ilastik
import dask_image
import dask_image.imread
import dask.array as da
import gc
from functools import partial
from scipy.interpolate import griddata
import cv2
from scipy import ndimage
import os
import yaml
import numpy as np

"""
List of all postDecon steps that need to be consolidated

[ ] smartCrop 
    - convert bit type as input is 32 bit tiff.
    - crop FFT
    - crop more if hash is overlapping with gel/sed interface
    - record crop parameters to log yaml file
[ ] postDecon 
    - make sure this is either 8 or 16bit input
    - apply local, adapative threshold
    - this generates at least three img arrays that could be vidualized
        - interpolated threshold array
        - positive and negative images of vlaues above/below threshold 
    - filters
        - curvature filter
        - gaussian blur
        
[ ] locating
    - no image filtering
    - pandas data frame output maybe write locations to text and whole dataframe to hdf5
[ ] visualization (possible options) 
    - two color, raw and locations with glyph
    - two color, locations with glyph and thresholded image
    - more complex colormap that visualizes:
        - double hits, core/shell, and some normalized measure of how good the overlap is. 
        

The structure should be a class with attributes for all the arrays, but internal clearing of memory for
any attribute that is not going to be saved. It is also possible that given hte default memory handling in numpy 
and pandas that the arrays would be overwritten as they are modified that exceptions will be made for arrays to save
as opposed for excpetions made for arrays to be deleted to preserve memory. 
"""

class PostDecon(dpl.dplHash):
    # initialize attributes
    # I want the attirbutes to be, mostly image arrays and I want to be able to call functions embedded
    #     inherited classes, mostly dplHash
    #
    def __init__(self,metaDataPath,hashValue):
        self.dpl = dpl.dplHash(metaDataPath)
        self.hashValue = hashValue
        print("hashValue is {}".format(self.hashValue))

    def smartCrop(self, computer = 'ODSY', output='np.array'):
        hashValue = self.hashValue
        self.smartCrop = self.dpl.smartCrop(hashValue, computer = computer, output = output)
        self.dpl.writeLog(hashValue,'smartCrop',self.smartCrop[1],computer = computer)

    def postDecon(self, computer = 'ODSY', output = 'np.array'):
        hashValue = self.hashValue
        self.postDecon = self.dpl.postDecon_python(hashValue, computer = computer, input = self.smartCrop,output = output)
        self.postDecon[0] = np.pad(self.postDecon[0],[(4,),(4,),(4,)]) # This has an implicit hashValue
        self.dpl.writeLog(hashValue,'postDecon',self.postDecon[1],computer = computer)
        #self.dpl.writeLog(hashValue,'postDecon',computer = computer)

    def locations(self,computer= 'ODSY'):
        hashValue = self.hashValue
        paramDict = self.dpl.metaData['locating']
        mat = self.dpl.sedOrGel(hashValue)
        locatingInputImg = self.postDecon[0] # This has an implicit hashValue
        #locatingInputImg = np.pad(self.postDecon[0],[(13,),(15,),(15,)]) # This has an implicit hashValue
        # Lets pad the image to get eliminate the effect of diameter on the search region
        self.locations = locating.iterate(locatingInputImg, paramDict, mat)
        self.dpl.writeLog(hashValue,'locating',self.locations[1],computer = computer)
        #self.dpl.writeLog(hashValue,'locations',computer = computer)

    def saveLocationDF(self,computer = 'ODSY'):
        hashValue = self.hashValue
        pxLocationExtension = '_' + self.dpl.sedOrGel(hashValue) + "_trackPy.csv"
        locationPath = self.dpl.getPath2File(hashValue, \
                                         kwrd='locations', \
                                         extension=pxLocationExtension,\
                                         computer=computer)
        self.locations[0].to_csv(locationPath,index=False)

    # removed in attempt to fix compilation error (Zsolt, March 28 2021)
    #def visualize(self,computer='ODSY'):
    #    self.overlay = paintByLocations.locationOverlay(self.locations[0],self.postDecon[0], locatingprogram = 'trackpy')
    #    fName_glyph = self.dpl.getPath2File(self.hashValue,kwrd='visualize', computer=computer, extension='visGlyph.tif')
    #    fName_locInput = self.dpl.getPath2File(self.hashValue,kwrd='visualize', computer=computer, extension='visLocInput.tif')
    #    flatField.array2tif(self.overlay.inputPadded,
    #                        fName_locInput,
    #                        metaData = yaml.dump(self.dpl.metaData, sort_keys = False))
    #    flatField.array2tif(self.overlay.glyphImage,
    #                        fName_glyph,
    #                        metaData = yaml.dump(self.dpl.metaData, sort_keys = False))

class PostDecon_dask(dpl.dplHash):
    """
    Steps and methods:
    [+] Initial class as inheritance from dpl
    [+] read in dask array
    [+] carry out smart crop
    [+] carry out postDecon
        [+] threshold
        [+] resize
        [+] curvature filter
    [ ] locate
    [ ] visualize

    """
    def __init__(self,metaDataPath,hashValue, computer='IMAC'):
        self.dpl = dpl.dplHash(metaDataPath)
        self.hashValue = hashValue
        self.computer = computer
        self.metaDataPath = metaDataPath
        self.mat = self.dpl.sedOrGel(self.hashValue)
        self.init_da = self.tif2Dask(None)
        self.refine_array = None


    def tif2Dask(self,fPath):
        """
        Given the hashValue and yaml file for this instance, open the tif file to dask array and chunk in z
        """
        if fPath == None:
            step_kwrd = self.dpl.getPipelineStep('smartCrop')
            fPath = self.dpl.getPath2File(self.hashValue,kwrd=step_kwrd,computer=self.computer)
        print("load step {} from path {}".format(step_kwrd,fPath))
        return dask_image.imread.imread(fPath)

    def smartCrop_da(self, fullStack):
        """
        This function carries out smartCrop using dask arrays and delaying computation as long as possible
        It is nearly identical to dpl.smartCrop but without the file reading stuff

        It returns a dask array, not numpy but it will be cropped correctly.

        This function crops the output of decon to remove spurious deconvolution artifacts in XY and Z.
        Additional cropping in Z is done for hashvalues that contain chunks overlapping with gel/sediment interface
        to ensure the maxEntropy thresholding in postDecon_imageJ does what it should using the stack histogram of the
        cropped stack.
        The function is "smart" in the sense that some of the crop paraemters are determined by analyzing the intensity
        for the specific hashvalue called as opposed to reading a global uniform parameter from the yaml metaData.

        This should interface with the writeLog function by outputting the required YAML data
        smartCrop:
          cropBool: True
          origin: [vector combining relative shifts of fftCrop and sedGel]
          dim: [new dimensions of the image]
          time: when the file was written

        hashValue:int the hashValue on which to run smartCrop
        computer:str either ODSY, SS, MBP or IMAC
        output:str either log, np.array or dask_array. Output type np.array can be used with dask input
        """

        # read in the correct upstream input data
        metaData = self.dpl.metaData['smartCrop']

        # initialize the log parameters
        refPos = self.dpl.getCropIndex(self.hashValue)
        originLog = [0,0,0] # relative changes in origin and dimensions to be updated and recorded in writeLog()
        dimLog = [0,0,0]
        errorDict = {}

        # run ilastik before cropping
        if self.dpl.metaData['smartCrop']['ilastik'] == True and fullStack is None:
            ilastik_meta = self.dpl.metaData['ilastik']

            # create threshold class in order to access getPathIlastik function...this is a bug that should be a fixed
            pxThreshold = ilastik.ilastikThreshold(self.metaDataPath,computer=self.computer)
            pxThreshold._sethv(self.hashValue)

            #run classifier
            run_args = self.dpl.metaData['ilastik']['pxClassifier']['run_args']
            exec, project, decon = pxThreshold.getPathIlastik('exec'),\
                                   pxThreshold.getPathIlastik('project'),\
                                   pxThreshold.getPathIlastik('decon')
            pxClassifier = ilastik.pixelClassifier(run_args, exec, project)

            # make sure that output fileName is empty. If file is present, delete it
            if ilastik_meta['pxClassifier']['output_filename_format'] == '{dataset_dir}/{nickname}_probabilities.h5':
                #out_fullPath = '/n/holyscratch01/spaepen_lab/zsolt/mnt/serverdata/zsolt/zsolt/tfrGel10212018x/tfrGel10212018x/strainRamp/tfrGel10212018A_shearRun10292018f20181030_22333 PM_20181030_65210 PM/decon/tfrGel10212018A_shearRun10292018f_decon_hv00122_probabilities.h5'

                #path = '/n/holyscratch01/spaepen_lab/zsolt/mnt/serverdata/zsolt/zsolt/tfrGel10212018x/tfrGel10212018x/strainRamp/tfrGel10212018A_shearRun10292018f20181030_22333 PM_20181030_65210 PM/decon/'
                deconPath = self.dpl.getPath2File(self.hashValue,
                                                  kwrd='decon',
                                                  computer=self.computer,
                                                  pathOnlyBool=True)

                #nickname = 'tfrGel10212018A_shearRun10292018f_decon_hv00122'
                nickname = self.dpl.getPath2File(self.hashValue, kwrd='decon',
                                                 computer=self.computer, fileNameOnlyBool=True).split('.')[0]
                suffix = '_probabilities.h5'
                fullPath = '{path}/{stem}{suffix}'.format(path=deconPath, stem = nickname, suffix=suffix)
                if os.path.exists(fullPath):
                    print("Removing previous ilastik file, and running classifier again\n")
                    os.remove(fullPath)

            pxClassifier.run(pxClassifier.parseArgs(decon))

            # now run threshold
            pxThreshold.setHashValue(self.hashValue)
            fullStack = da.from_array(pxThreshold.threshold(self.mat))

        # if appropriate, crop out the uniform decon FFT artifacts in XY and Z
        if metaData['fftCrop']['bool'] == True:
            dim = fullStack.shape
            crop = (metaData['fftCrop']['X'], metaData['fftCrop']['Y'],metaData['fftCrop']['Z'])
            fullStack = fullStack[crop[2]:dim[0]-crop[2], \
                        crop[1]:dim[1]-crop[1], \
                        crop[0]:dim[2]-crop[0]] # crop is xyz while dim is zyx
            for i in range(len(refPos)):
                refPos[i] = (refPos[i][0] + crop[i], refPos[i][1] - crop[i])
            originLog = [originLog[n] + crop[n] for n in range(len(crop))]
            dimLog = [dimLog[n] + -2*crop[n] for n in range(len(crop))]

        # is this hashvalue contain a sed/gel interface? Is it mostly sed or gel?
        sedGel = self.dpl.sedGelInterface(self.hashValue)
        if sedGel[1] == True \
                and metaData['sedGelCrop']['bool'] == True \
                and metaData['sedGelCrop']['method'] == 'zGradAvgXY' \
                and metaData['ilastik'] == False:
            # Same cropping algo as dpl.smartCrop
            # need to compute zGradAvgXY in order to find out crop parameters.
            pixelZGrad = flatField.zGradAvgXY(
                fullStack.rechunk(chunks = fullStack.shape).compute()
                )
            maxValue = max(pixelZGrad)
            maxIndex = list(pixelZGrad).index(maxValue) # this is the z index of the max grad (?) I think
            # Now do some quality control on this max value:
            # is the max Value large enough?
            if maxValue < metaData['sedGelCrop']['minValue']:
                msg = "maximum gradient for sedGelCrop is below the minValue listed in metaData"
                #print(maxIndex,maxValue,metaData['sedGelCrop']['minValue'])
                errorDict["maxGradError"] = {'msg': msg,
                                             'maxValue': maxValue,
                                             'maxIndex': maxIndex,
                                             'minValue': metaData['sedGelCrop']['minValue']}
                for key in errorDict["maxGradError"]: print(key,errorDict['maxGradError'][key])
            # Is the index close to where the purported sed/gel interface is?
            sedGelDeviation = abs((maxIndex + refPos[2][0] - metaData['sedGelCrop']['offset'])
                                  - self.dpl.metaData['imageParam']['gelSedimentLocation'])
            if sedGelDeviation > metaData['sedGelCrop']['maxDev']:
                msg = "The purported gel/sediment location is {} is further than expected".format(sedGelDeviation)
                #raise KeyError
                errorDict['sedGelDeviationError'] = {'sedGelDeviation': sedGelDeviation,
                                                     'maxDev': metaData['sedGelCrop']['maxDev'],
                                                     'maxIndex': maxIndex,
                                                     'z ref pos': refPos[2][0],
                                                     'nominal gel/sed loc': self.dpl.metaData['imageParam']['gelSedimentLocation']}
                for key in errorDict['sedGelDeviationError']: print(key, errorDict['sedGelDeviationError'][key])
            # crop the stack using the maxindex and uniform offset
            offset = metaData['sedGelCrop']['offset']
            zSlices = fullStack.shape[0]
            if sedGel[0] == 'sed': # crop from the bottom
                fullStack = fullStack[maxIndex - offset : zSlices ,:,:] # We need to use the offset to relax the cropping
                refPos[2] = (refPos[2][0] + maxIndex - offset,refPos[2][1])
                originLog[2] = originLog[2] + maxIndex - offset
                dimLog[2] = dimLog[2] - (maxIndex - offset)
                print("Warning, on smartCrop, check to make sure you have enough slices to crop given offset")
            elif sedGel[0] == 'gel':
                fullStack = fullStack[0:maxIndex - offset,:,:]
                refPos[2] = (refPos[2][0], maxIndex - offset)
                dimLog[2] = dimLog[2] - (len(pixelZGrad) - maxIndex + offset)
                print("Warning, on smartCrop, check to make sure you have enough slices to crop given offset")

        return [fullStack, {'smartCrop': {'origin' : originLog, 'dim' : dimLog, 'refPos' : refPos}}]

    def threshold_da(self,input_da, **thresholdMeta):
        """
        Carries out thresholding on input_da and returns a dask array of the thresholded values
        """

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
            hist, _ = np.histogram(stack, bins=range(2 ** 16), density=True)
            c_hist = hist.cumsum()
            c_hist_i = 1.0 - c_hist

            # To avoid invalid operations regarding 0 and negative values.
            c_hist[c_hist <= 0] = 1
            # I think this is a logical index on the boolean expression: if c_hist<=0, set that value to 1
            c_hist_i[c_hist_i <= 0] = 1

            c_entropy = (hist * np.log(
                hist + (hist <= 0))).cumsum()  # add logical array hist<=0 to make sure you dont take log(0)
            b_entropy = -c_entropy / c_hist + np.log(c_hist)

            c_entropy_i = c_entropy[-1] - c_entropy
            f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

            return np.argmax(b_entropy + f_entropy)

        def maxEnt(chunk):
            """
            Computes maxEntropy threshold on chunk and returns and array of the same size
            that has NAN at all values other the center of the array which has the threshold value
            When this function is applied to dask array with map_overlap, it will produce the output
            for thresholding
            """
            out = np.empty(chunk.shape, dtype='float32')
            out[:] = np.nan
            cz, cy, cx = np.array((np.array(out.shape) - 1) / 2).astype(int)
            chunk1D = np.ndarray.flatten(chunk)
            out[cz, cy, cx] = np.array([maxEntropyThreshold(chunk1D)])[..., None, None]
            return out

        def applyThreshold(imgArray, thresholdArray, recastBool=True, scaleFactor=1.0):
            """
            This function does not compute a threshold. It just takes imgArray and thresholdArray
            and outputs an 16 bit image of the threshold with optional recasting to image to 16 bit depth.
            :return:
            """
            # change type to enable subtraction
            out = imgArray.astype('float32') - scaleFactor * thresholdArray.astype('float32')
            # clip everything below zero
            positive = out  # make a deep copy in case we also want to return the thresholded parts.
            negative = out * -1
            positive[positive < 0] = 0  # now use logical indexing to reassign all negative values to zero
            negative[negative < 0] = 0
            if recastBool == True:
                positive = threshold.arrayThreshold.recastImage(positive,
                                                                'uint16')  # rescale the dynamic range after thresholding to span 16bits
                negative = threshold.arrayThreshold.recastImage(negative,
                                                                'uint16')  # rescale the dynamic range after thresholding to span 16bits
            return positive, negative

        # ToDo:
        #  - uodate to take in kwarg from metaData
        #  - also rechunk the data
        rechunk_nzyx = thresholdMeta['local']['dask']['rechunk_nzyx']
        depth_delta = thresholdMeta['local']['dask']['depth_delta']
        boundary = thresholdMeta['local']['dask']['boundary']


        chunks_dim = np.ceil(
            np.array(input_da.shape)/np.array(rechunk_nzyx)
                              ).astype(int)
        input_da = input_da.rechunk(chunks=tuple(chunks_dim.astype(int)))
        # now what the smallest chunk size? Modular division
        #depth_dim = chunks_dim % np.array(input_da.shape) \
        #        + np.array(depth_delta)
        depth_dim = ((np.array(input_da.shape) % chunks_dim) + chunks_dim + np.array(depth_delta)) % chunks_dim
        #           ((             largest remainder       ) + unwrap and shift down             ) mod to get greatest lower bound
        thresh_compute = (input_da.map_overlap(maxEnt,
                                              depth=tuple(depth_dim.astype(int)),
                                              dtype='float32',
                                              boundary=boundary)).compute()

        # split thresh_compute into values and corresponding indices, and format for scipy.griddata
        values = thresh_compute[~np.isnan(thresh_compute)]
        indices = np.argwhere(~np.isnan(thresh_compute))
        points = (indices[:, 0], indices[:, 1], indices[:, 2])
        client.cancel([input_da,thresh_compute])
        # note that I have to use nearest interpolation as the dask chunks chunk centers are not on the edges...
        # ... but then again if this was mirrored the edges would give the same values as an interior interpolation
        # other option is to fill with avg of values as opposed to nan
        zz,yy,xx = np.mgrid[0:input_da.shape[0]:1, 0:input_da.shape[1]:1, 0:input_da.shape[2]:1]
        threshold_array = griddata(points, values, (zz, yy, xx), method='nearest', fill_value=np.mean(values)).astype(
            'uint16')
        aboveThresh, belowThresh = applyThreshold(input_da.compute(scheduler='threads'), threshold_array)
        # now clean up garbage
        del values, indices, points, threshold_array, belowThresh
        gc.collect()
        return da.from_array(aboveThresh, chunks=(1,aboveThresh.shape[1],aboveThresh.shape[2]))

    def resize_da(self,input_da,**kwargs):
        """
        Resizes xy slices with scale factor in yamlMetaData
        ToDo:
          [+] Review double splat notation and useage for reading parameters from input dictionary...like how to convert
              input dictionary kwrds to local variables that can be referenced.
          [+] Can **kwargs be instead **yamlDict or **metaData to be more descriptive variable name?
          [+] Also double check the dimensioning works here and note that I will need to specify the output chunk size
              in order to run map blocks on this data.
        """
        postDecon_meta = kwargs
        dim = (postDecon_meta['upScaling']['dim']['x']*input_da.shape[2],
               postDecon_meta['upScaling']['dim']['y']*input_da.shape[1])
        interp_method = postDecon_meta['upScaling']['interp_method']

        slice = input_da.squeeze() # get a true 2D slice.
        dict_interp = {'lanczos': cv2.INTER_LANCZOS4,
                       'cubic': cv2.INTER_CUBIC,
                       'linear': cv2.INTER_LINEAR,
                       'nearest': cv2.INTER_NEAREST,
                       'bilinear': cv2.INTER_AREA}
        upscale = cv2.resize(slice,dim,interpolation=dict_interp[interp_method])
        return upscale[None,...]

    def postThresholdFilter_da(self,input_da,**postDeconMeta):
        """
        Carries out postDecon filter steps using dask map_blocks
        Return a dask array that can be computed at the **end** of the filtering steps
        ToDo:
          [+] fill in the functions from dask_testScript.py
          [+] cyclce through the list of filters to be applied in yaml metaData.
          - check that chaining and delayed compute on multiple filters works
        """

        # define the filter functions to be applied to each slice of the input dask array
        def tvFilter(input_da, iter):
            f =  partial(curvatureFilter.CF,filterType=0,total_iter=iter)
            filteredSlice = f(input_da.squeeze())
            return filteredSlice[None,...]

        def mcFilter(input_da, iter):
            f= partial(curvatureFilter.CF,filterType=1,total_iter=iter)
            filteredSlice = f(input_da.squeeze())
            return filteredSlice[None,...]

        def gaussianBlur(input_da, sigma):
            f = partial(ndimage.gaussian_filter,sigma=sigma)
            filteredSilce = f(input_da.squeeze())
            return filteredSilce[None,...]

        # Convert input to float32 just in case
        input_da = input_da.astype('float32')
        # rechunk along z
        #input_da = input_da.rechunk(chunks=(1,input_da.shape[1], input_da.shape[2]))

        for filterDict in postDeconMeta['postThresholdFilter'][self.mat]['methodParamList']:
            # cycle through the list of filters and reassign outputs to input_da
            print("Carrying out filter {}".format(filterDict['method']))

            # is it totalVariation?
            if filterDict['method'] == 'totalVariation':
                n_iter = filterDict['iter']
                f = partial(tvFilter,iter=n_iter)
                input_da = input_da.map_blocks(f,dtype='float32')

            # is it gaussian blur?
            elif filterDict['method'] == 'gaussianBlur':
                sigma = filterDict['sigma']
                f = partial(gaussianBlur,sigma=sigma)
                input_da = input_da.map_blocks(f, dtype='float32')

            # is it mean curvature?
            elif filterDict['method'] == 'meanCurvature':
                n_iter = filterDict['iter']
                f = partial(mcFilter,iter=n_iter)
                input_da = input_da.map_blocks(f,dtype='float32')

            # Whatever it is, I havnt implemented it yet
            else: raise KeyError("method {} not recognized or implemented".format(filterDict['method']))

        # note the return type is not yet computed.
        return input_da

    def iterativeLocate_da(self, input_da,**locatingMeta):
        pass

    def postDecon_dask(self):
        """
        In principle this function carries out all steps after deconvolution:
            [+] smartCrop
            [+] threshold
            [+] postThreshdoldFilter
            [-] upscale
            [+] locate
            [+] refine
            [-] visualize
        However the steps marked [-] have not been implemented yet
        This will also automatically log the results, however the job control could be improved.
        """
        computer=self.computer

        print("Warning, calling postDecon_dask from postDeconCombined.PostDecon_dask\n"
              "Job control over steps has not been implemented in any way apart from\n"
              "directly editing the function.\n"
              "Zsolt - Jul 20 2020\n")
        print("Initialiizing dask cluster")
        global client
        if computer =='IMAC':
            from dask.distributed import Client, LocalCluster
            #node = LocalCluster(n_workers=8, threads_per_worker=24,
            #                    ip='tcp://localhost:8786',
            #                    memory_limit='4Gb')

            #~~~~~~~~~~~~~~~~~~ Not working, March 26 2021
            #client = Client(node)
            #IMAC_ip = self.dpl.metaData['dask_resources'][computer]['ip']
            #client = Client(IMAC_ip)
            #~~~~~~~~~~~~~~~~~~~~~~~

            nprocs = self.dpl.metaData['dask_resources'][computer]['nprocs']
            nthreads = self.dpl.metaData['dask_resources'][computer]['nthreads']
            mem = self.dpl.metaData['dask_resources'][computer]['memory-limit']
            node = LocalCluster(n_workers=nprocs,
                                threads_per_worker=nthreads,
                                memory_limit=mem,
                                silence_logs='INFO')
            client = Client(node)
            #client.restart()
            ## restart since the cluster on imac may not be fresh.
            client.restart()
        else:
            # This should work... on odsy.. works on test node...do I need to set memory
            # constraints?

            # not clear why I need the if __name__ =="__main__" clause, but
            # it solved the problem. Im suprised it worked.
            # See https://github.com/dask/distributed/issues/2520
            from dask.distributed import Client, LocalCluster
            print("Starting LocalCluster inside __name__ =='__main__:' condition")
            nprocs = self.dpl.metaData['dask_resources'][computer]['nprocs']
            nthreads = self.dpl.metaData['dask_resources'][computer]['nthreads']
            local_dir = self.dpl.metaData['dask_resources'][computer]['local_directory']
            #local_dir = self.dpl.getPath2File(0,kwrd='postDecon',pathOnlyBool=True, computer=computer)

            node = LocalCluster(n_workers=nprocs,
                                threads_per_worker=nthreads,
                                local_directory=local_dir,
                                silence_logs='INFO')
            client = Client(node)
            #client.restart()

        #client.restart()
        if self.dpl.metaData['smartCrop']['ilastik'] == False:
            da_decon = self.init_da
            # carry out smart crop
            print("Starting smart crop")
            da_smartCrop, log_smartCrop = self.smartCrop_da(da_decon)
            client.cancel(da_decon)
        elif self.dpl.metaData['smartCrop']['ilastik'] == True:
            da_smartCrop, log_smartCrop = self.smartCrop_da(None)
            if self.dpl.metaData['locating']['refine_lsq']['refine_array'] == 'smartCrop':
                self.refine_array = threshold.arrayThreshold.recastImage(da_smartCrop,'uint16')


        # log the changes
        self.dpl.writeLog(self.hashValue, 'smartCrop', log_smartCrop, computer=computer)

        # Carry out the threshold
        print("Starting thresholding")
        da_threshold = self.threshold_da(da_smartCrop, **self.dpl.metaData['postDecon']['threshold'])
        client.cancel(da_smartCrop)

        # post threshold filter
        print("Starting post threshold filtering")
        metaData = self.dpl.metaData['postDecon']
        da_postThresholdFilter = self.postThresholdFilter_da(da_threshold, **metaData)
        np_postThresholdFilter = da_postThresholdFilter.compute(scheduler='threads')

        # carry out locating and refinement, iteratively
        print("Starting locating and refinement, iteratively")
        client.restart()
        metaDataFolder = self.dpl.getPath2File(0,kwrd='metaDataYAML',computer=computer,pathOnlyBool=True)
        df_loc, df_refine, log_locating = locating.iterate(np_postThresholdFilter,
                                                           {'yamlMetaData': self.dpl.metaData,
                                                            'computer': computer},
                                                           self.mat,
                                                           metaDataYAMLPath=metaDataFolder,
                                                           daskClient = client,
                                                           imgArray_refine = self.refine_array)

        # integrate pxProb channels and add as columns to output locations df_refine
        pxIntegrate = ilastik.ilastikIntegrate(self.metaDataPath, computer=self.computer)
        #set hashvalue
        pxIntegrate.setHashValue(self.hashValue)
        #read pxProb
        pxIntegrate._readPxProb(None)
        df_ilastik = pxIntegrate.integratePxProb(df_refine)
        loc_idx = df_refine.index
        df_refine = df_refine.join(df_ilastik.set_index(loc_idx))

        # set hashvalue
        pxIntegrate.setHashValue(self.hashValue)
        #read pxProb and crop
        pxIntegrate._readPxProb(None)
        # integrate with df_refine, and maybe check that column labels are correct
        df_refine_ilastik = pxIntegrate.integratePxProb(df_refine)

        # save input image  to visualize directory
        print("Saving input images to scratch visualize ")
        fName_locInput = self.dpl.getPath2File(self.hashValue,kwrd='visualize', computer=computer, extension='locInput.tif')
        flatField.array2tif(np_postThresholdFilter,
                            fName_locInput,
                            metaData = yaml.dump(self.dpl.metaData, sort_keys=False))

        # save locations to csv
        print("Saving locations, both refined and centroid")
        pxLocationExtension = '_' + self.dpl.sedOrGel(self.hashValue) + "_trackPy.csv"
        refineLocExt = '_' + self.dpl.sedOrGel(self.hashValue) + "_trackPy_lsqRefine.csv"
        locationPath = self.dpl.getPath2File(self.hashValue, kwrd='locations', extension=pxLocationExtension, computer=computer)
        refinePath = self.dpl.getPath2File(self.hashValue, kwrd='locations', extension=refineLocExt, computer=computer)
        df_loc.to_csv(locationPath, index=False, sep=' ')
        df_refine.to_csv(refinePath, index=False, sep=' ')
        self.dpl.writeLog(self.hashValue, 'locating', log_locating, computer=computer)

        client.close()
        node.close()
        self.results = {'df_loc': df_loc,
                        'df_refine':df_refine,
                        'np_postThresholdFilter':np_postThresholdFilter,
                        'log_locating': log_locating}
        return df_loc, df_refine, np_postThresholdFilter, log_locating


if __name__ == '__main__':
    yamlTestingPath = '/Users/zsolt/Colloid/SCRIPTS/tractionForceRheology_git/TractionRheoscopy' \
                      '/metaDataYAML/tfrGel09052019b_shearRun05062019i_metaData_scriptTesting.yaml'
    testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
                  'OddysseyHashScripting/pyFiji/testImages'
    inst = PostDecon(yamlTestingPath,44)
    print(inst.dpl.queryHash(44))
    print("running smart Crop")
    inst.smartCrop(computer='MBP')
    print("running post Decon")
    inst.postDecon(computer='MBP')
    print("running locations")
    inst.locations(computer='MBP')
    print("running visualize")
    inst.visualize()
    print("saving particle locations to file")
    inst.saveLocationDF(computer='MBP')

    import seaborn as sb
    from matplotlib import pyplot as plt
    import numpy as np
    import yaml
    import time
    import locationStitch as ls
    locDF = inst.locations[0]
    for n in range(5):
      sb.distplot(locDF[locDF['n_iteration'] == n]['mass'],kde=False)
    plt.show()

    sb.distplot(locDF[locDF['n_iteration'] == 1]['x (px)'].apply(np.modf)[0], kde=False, norm_hist=True)
    sb.distplot(locDF[locDF['n_iteration'] == 1]['y (px)'].apply(np.modf)[0], kde=False, norm_hist=True)
    sb.distplot(locDF[locDF['n_iteration'] == 1]['z (px)'].apply(np.modf)[0], kde=False, norm_hist=True)
    plt.show()



    moment = time.strftime("%Y-%b-%d_%H_%M", time.localtime())
    fName = 'paramSearch_hv00044_{}'.format(moment)

    lsInst = ls.ParticleStitch(yamlTestingPath,computer='MBP')
    lsInst.locations = lsInst.csv2DataFrame(44)
    lsInst.recenterHash(lsInst.locations,44,coordStr='(um, imageStack)')
    lsInst.dataFrameLocation2xyz(lsInst.locations,testImgPath+'/{}.xyz'.format(fName),\
                                 columns=['x (um, imageStack)',\
                                          'y (um, imageStack)',\
                                          'z (um, imageStack)'])
    with open(testImgPath+'/{}.yaml'.format(fName),'w') as f:
        yaml.dump(lsInst.dpl.metaData,f)




# ToDo:
#  [+] incorporate this class into scripting framework
#  [+]  rewrite writeLog function to work with postDeconCombined.
#      -> make output of all pipeline steps the same as smartCrop...namely return a dictionary to log
#      -> check that writelog is going to take handle whatever keys and dict its is given
#      -> make sure that only the relevant information is being recorded...dont record refPos if \
#         there was no cropping becasue then it will seem like refPos is accurate as of that step
#         which is not true. We want to record incremetnal chagnes in origin and dim and then intgrate
#         those through the pipeline
#  [+] decide on and implement a method for serializing particle locating output
#    - xyz files of particle locations in um, possibly recentered and in absolute coordinates
#      -> after stitching
#    - csv file of particle locations in px along with other output of dataframe
#      -> csv file of full dataFrame. Its fast and I am sure that everyone will be able to read it
#    - some other serialization format like hdf5 or feather or pickle to serialize the dataFrame itself.
#     -> Again, not worth it. Maybe after the whole shebang is stitched.
#  - stitch particle locations, possibly by using trackpy on two overlapping hashValues that have shifted back
#    back to reference coordinates.
#  - implement a standard visualization, possibly default is that the image is dark if the particle and raw
#    image overlap, with xz and yz crosssection... maybe scale the glyph to have the same intergrated mass as the
#    located particle...and then subtract the located particle from the glyph and recenter to midrange of 8bit depth
