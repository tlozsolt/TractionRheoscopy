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

class hashStitch:
    """
    A class for creating stitched images that are outputed from hashed decon particle locating image processing
    """
    def __init__(self,yamlPath):
        # initalize the dplInstance associated with the yamlPath
        self.dplInst = dpl.dplHash(yamlPath)
        self.metaData = self.dplInst.metaData
        # initialize the stitched image with three options: sed gel or both
        self.imageParamDict = self.metaData['imageParam']
        (gelBottom,sedGelInt,sedTop) = (0,self.imageParamDict['gelSedimentLocation'],self.imageParamDict['zDim'])
        (xDim,yDim) = (self.imageParamDict['xDim'],self.imageParamDict['yDim'])
        self.sedCanvas = (np.zeros((sedTop - sedGelInt,yDim, xDim),dtype=np.float64),\
                          np.zeros((sedTop - sedGelInt,yDim, xDim),dtype=int)\
                          )
        self.gelCanvas = (np.zeros((sedGelInt,yDim, xDim),dtype=np.float64), \
                          np.zeros((sedGelInt,yDim, xDim),dtype=int) \
                          )
        self.allCanvas = (np.zeros((sedTop,yDim, xDim),dtype=np.float64), \
                          np.zeros((sedTop,yDim, xDim),dtype=int) \
                          )
        # I think we need a new data construct that is pair of arrays for each canvas.
        # The canvas includes an array of stitched img values (ie just the stitched image)
        # and an array that counts the number times that pixel has been updated.
        # and maybe also a function that describes how the values are merged:
        # average, replacement, max, or linear weight in overlap region.
        # I think this could be implemented as some kind of discrete convolution on the counting matrix.
        # just like averaging is a convolution with a kernel [0.3,0.3,0.3]

    def clearCanvas(self):
        # just resets the canvas values to initial zeros, without reinit the whole instance
        (gelBottom,sedGelInt,sedTop) = (0,self.imageParamDict['gelSedimentLocation'],self.imageParamDict['zDim'])
        (xDim,yDim) = (self.imageParamDict['xDim'],self.imageParamDict['yDim'])
        self.sedCanvas = (np.zeros((sedTop - sedGelInt,yDim, xDim),dtype=np.float64), \
                          np.zeros((sedTop - sedGelInt,yDim, xDim),dtype=int) \
                          )
        self.gelCanvas = (np.zeros((sedGelInt,yDim, xDim),dtype=np.float64), \
                          np.zeros((sedGelInt,yDim, xDim),dtype=int) \
                          )
        self.allCanvas = (np.zeros((sedTop,yDim, xDim),dtype=np.float64), \
                          np.zeros((sedTop,yDim, xDim),dtype=int) \
                          )


    def openImage(self,hv,step='postDecon', computer = 'ODSY'):
        fPath = self.dplInst.getPath2File(hv,kwrd=step,computer=computer)
        return flatField.zStack2Mem(fPath)

    def integrateTransVect(self,hv,step='postDecon',computer='ODSY'):
        """
        Reads the yaml log file and returns the integrated translation vector for that specific hashValue
        :param hv:  hashValue
        :param step: step in pipeline that we should integrate up through. This should be done by searching the log \\
                     file and not by reading the pipeline from yaml as, in principle the pipeline from yaml could
                     reflect a job that was restarted.
        :param computer:
        :return: tuple giving global (x,y,z) of hash chunk origin.
        """
        # open the yaml file and parse it
        with open(self.dplInst.getPath2File(hv,kwrd='log',computer=computer),'r') as stream:
            yamlLog = yaml.load(stream,Loader=yaml.SafeLoader)
        pipeline = self.dplInst.metaData['pipeline']
        # create a list of pipeline steps that were both flagged as true and would be included in the log
        keyList = [list(elt.keys())[0] for elt in pipeline]
        bool = [list(elt.values())[0] for elt in pipeline]
        pipelineSteps = [keyList[n] for n in range(len(keyList)) if bool[n]==True and keyList[n]!='rawTiff']
        origin = np.array([0,0,0])
        dim = np.array([0,0,0])
        # I need to integrate up through step listed in arguement as a safegaurd against incomplete jobs.
        for key in pipelineSteps:
          if key != step:
            try:
              logEntry = yamlLog[key]
              origin += logEntry['origin'] # note, relies on type conversion of logEntry to numpy.array
              dim += logEntry['dim']
            except KeyError:
              print("There is something wrong when comparing pipeline to log. Perhaps incomplete job?")
              raise KeyError
          else: # break out of the for loop after you hit step kwrd arguement
              origin += logEntry['origin'] # note, relies on type conversion of logEntry to numpy.array
              dim += logEntry['dim']
              break
        return (origin,dim)

    def hashStitch(self,subCanvas,hashImg):
        """
        Stitches hashImg onto subCanvas and averages values if multiple hashes overlap
        This function does not do **any** translation and assume subcanvas has the same dimensions
        as imArray.
        At a single pixel this function updates the pixel to be:
            subCanvas[i,j,k] = (count[i,j,k]*subCanvas[i,j,k] + hashImg[i,j,k])/(count[i,j,k] + 1)
        :param subCanvas: pair of numpy arrays of image (dtype = float32) and imgCounter (dtype=int) \
                       where imgCounter just counts how many times that pixel has been updated.
        :param hashImg: the image array, likely 8bit that you would like to add to the canvas
        :return: subCanvas with imArray included.
        """
        # Check dtypes and dimensions
        if subCanvas[0].shape != hashImg.shape:
            print("hashStitch called but subCanvas and hashImg have different dimensions:")
            print("subCanvas[0].shape: ", str(subCanvas[0].shape))
            print("subCanvas[1].shape: ", str(subCanvas[1].shape))
            print("hashImg.shape: ", str(hashImg.shape))
            raise IndexError
        # apply the averaging function, if possible w/o a for loop
        subCanvas = ((subCanvas[1][:]*subCanvas[0][:] + hashImg[:])/(subCanvas[1][:] + 1),\
                     subCanvas[1][:] + 1
                     )
        return subCanvas
        # return the subcanvas datatype.

    def placeHash(self,hashValue,step='postDecon',computer='ODSY',canvasType = 'all'):
        """
        Takes an image array and places
        :param hashValue:
        :param step: string describing what pipeline step to stitch
        :param computer: string giving the computer, either 'MBP' or 'ODSY'
        :param canvas: string specifying if you want to place Hash on 'sed;, 'gel', or 'all'.
        :return:
        """
        # get the integrated origin and dim
        origin,dim = self.integrateTransVect(hashValue,step=step,computer=computer)
        imageHash = self.openImage(hashValue,step=step,computer=computer)
        # decide on whether you want to placeHash on sed, gel, or both based on the canvas arguement
        if canvasType =='all':
            # careful to check xyz or zyx listing. I think origin, dim is xyz while allStitch in zyx indexing.
            # cut out the subCanvas using origin and dim
            subCanvas = [self.allCanvas[i][origin[2]:origin[2] + dim[2],\
                                           origin[1]:origin[1] + dim[1],\
                                           origin[0]:origin[0] + dim[0]]\
                         for i in range(len(self.allCanvas))]
            # apply hashStitch
            subCanvas = self.hashStitch(subCanvas,imageHash)
            # place the subCanvas back in using origin and dim
            for i in range(len(self.allCanvas)):
                self.allCanvas[i][origin[2]:origin[2] + dim[2],\
                                  origin[1]:origin[1] + dim[1],\
                                  origin[0]:origin[0] + dim[0]]\
                = subCanvas[i]
            # return the full canvas
            return self.allCanvas
        elif canvasType != 'all':
            print("canvasType is: ", str(canvasType))
            print("Stitching on sed or gel seperately has not yet been implemented")
            raise TypeError
        else: pass

    def makeQuilt(self, dplInst, time = 'all', step='postDecon',computer = 'ODSY',canvasType='all'):
        """
        This function takes in an instance of dplHash class object and some extra flags (step and computer)
        and creates a stitched image of all the available hashes in the given folder keeping in mind
        that it will create seperate z-stacks for each timepoint
        It is designed to run to completion, and if there are errors like missing files it will print
        a warning but otherwise continue.
        :param dplInst:
        :param time: either a string 'all', a single int, or a list of int specifying which time points to stitch
        :param step: string from pipeline, probably rawTiff or postDecon
        :param computer: string ODSY or MBP
        :return: none, but extensive print statements if some hashValues were not found.
        """
        # [+] with time flag: if all -> generate a list of times and call makeQuilt on the list recursively
        #     if time is a list, make a for loop and call makeQuilt recursively on each elt
        if time == 'all':
            time = [n for n in range(dplInst.metaData['imageParam']['timeSteps'])]
            makeQuilt(dplInst, time = time, step=step, compute=computer)
        elif type(time) == list:
            for t in time:
                makeQuilt(dplInst, time=t, step=step, compute=computer)
                # I think I need to rezero the self.allCanvas after I complete a single time
                # just create another function that reinitializes the canvas. clearCanvas()
                self.clearCanvas()
        else:
            # now kwrd imput variable time is definitely a single time step
            hvListOneTime = dplInst.time2HVList(time,'all')# List of all hashValues for the given time step.
            for hv in hvListOneTime:
                try:
                    self.placeHash(hv,step=step,computer=computer,canvasType=canvasType)
                except FileNotFoundError:
                    print("There are likely some missing hashValues")
                    print("time: ", str(time)," hv: ", str(hv), "step: ", step)
                    pass
            # Save the stitched image and also the image of the count stack.
            fOut = dplInst.getPath2File()


            # [ ] try to open the file, if failed print error statement but continue on
            #     for each file in a given time point, run the placeHash on allCanvas
            # [ ] save the result to an 8 bit tiff stack to a path specified in yaml file
            # [ ] we should add stitching as another pipeline step and change the pipeline control
            #     scripts to stop at locating as both tracking and stitching require complete jobs.

if __name__ == "__main__":
    #
