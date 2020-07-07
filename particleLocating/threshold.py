import numpy as np
from particleLocating import flatField, pyFiji
from scipy.interpolate import griddata
from joblib import Parallel,delayed

class arrayThreshold:
    """
    A class for applying adpative local threshdoling to image arrays.
    Most heavy lifting is done by import functions
    """
    def __init__(self,imgPath):
        try:
          self.imgArray = self.recastImage(flatField.zStack2Mem(imgPath),dtypeOut='uint16')
          self.thresholdArray = np.zeros(self.imgArray.shape)
        except ValueError:
          #print('Assuming input {} is a list of [stack,dict of log values]'.format(imgPath))
          self.imgArray = imgPath[0]
          self.thresholdArray = np.zeros(self.imgArray.shape)
          self.log = imgPath[1]

    def localThreshold(self,blockDim,edgeAugment = 'reflect',n_xyz = (3,15,15),parBool=True,n_jobs=4):
        """
        Apply a threshold by moving through the imageArray, break into blockDim, and apply thresold algo to block
        :param blockDim: int, size of the block in pixels,
        :param n_xyz: 3-tuple, number of samples in x,y,z to use in local threshold.
        :param edgeAugment: str, how should the local threshold work if blockDim is outside image bounds?
        :param thresholdFunc: what, probably adaptive function should we use to compute local threshold?
        :param parBool: boolean, should we use parallel processing using joblib package?
        :param n_jobs: int, number of paralell jobs to use if parBool==True.
                            Note I may be memory constrained and not core constrained for parallel processing.
        :return: (points, values), tuple containing image coordinates and computed threshold for block centered
                 at points. This is likely later interpolated to get a threshold array that is used as a mask
        """
        # for every pixel, find the block dim and mirror if necessary for edge case
        def makeBlock(array, blockDim, pxCoord):
            delta = np.floor(blockDim/2).astype(int)
            padShift = np.floor(blockDim/2).astype(int)
            padded = np.pad(array,np.floor(blockDim/2).astype(int),mode=edgeAugment)
            lowRange = pxCoord + padShift - delta
            upRange = pxCoord + padShift + delta + 1 # add one for zero indexing
            # There is likely a better way to do this that would naturally work for array of any dim
            # but for now I will be content with that fact that I didnt use a for loop
            if len(lowRange) == 3 :
                return padded[lowRange[0] : upRange[0],\
                              lowRange[1] : upRange[1],\
                              lowRange[2] : upRange[2]]
            elif len(lowRange) == 2:
                return padded[lowRange[0]: upRange[0],\
                              lowRange[1]: upRange[1]]
            elif len(lowRange) == 1:
                return padded[lowRange[0]: upRange[0]]
            else:
                print("adaptive Threshold does not work for array of dim larger than 3")
                raise IndexError

        def partialApplication(i, j, k):
            block = np.ndarray.flatten(makeBlock(image, blockDim, np.array([i, j, k])))
            # it would be best to implement to take **any** function as threshold algorithm, not just maxEnt as
            # current hardcoded.
            return (self.maxEntropyThreshold(block), i, j, k)

        image = self.imgArray
        nx,ny, nz = n_xyz
        if parBool == False:
            points_z, points_y, points_x = [], [], []
            values = []
            for i in list(np.linspace(0,image.shape[0],nz).astype(int)):
                for j in list(np.linspace(0, image.shape[1], ny).astype(int)):
                    for k in list(np.linspace(0, image.shape[2], nx).astype(int)):
                        block = np.ndarray.flatten(makeBlock(image,blockDim,np.array([i,j,k])))
                        points_z.append(i)
                        points_y.append(j)
                        points_x.append(k)
                        values.append(self.maxEntropyThreshold(block))
        elif parBool == True:
            parOut = Parallel(n_jobs=n_jobs)(delayed(partialApplication)(i,j,k)\
                                             for i in list(np.linspace(0, image.shape[0], nz).astype(int))\
                                             for j in list(np.linspace(0, image.shape[1], ny).astype(int))\
                                             for k in list(np.linspace(0, image.shape[2], nx).astype(int))\
                                             )
            values = [elt[0] for elt in parOut]
            points_z = [elt[1] for elt in parOut]
            points_y = [elt[2] for elt in parOut]
            points_x = [elt[3] for elt in parOut]
        else: pass
        points = (np.array(points_z).astype(int),\
              np.array(points_y).astype(int),\
              np.array(points_x).astype(int))
        values = np.array(values).astype('uint16')
        return points,values

    def interpolateThreshold(self,points,values):
        """
        Function carries out an interpolation of volumetric image data given linked array of points and values
        :param points: 3-tuple of position arrays (position_z,position_y, position_x)
                       giving n-th (xyz) coordinate -> (points[2][n],points[1][n],points[0][n])
        :param values: linked array of sampled image values. n-th coordinate as listed above corresponds values[n]
        :return: image data that has been interpolated in 3D and output at uint16.
        """
        image = self.imgArray
        zz,yy,xx = np.mgrid[0:image.shape[0]:1, 0:image.shape[1]:1, 0:image.shape[2]:1]
        return griddata(points,values,(zz,yy,xx), method='linear').astype('uint16')

    def maxEntropyThreshold(self,stack):
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

        #self.thresholdArray[:] = np.argmax(b_entropy + f_entropy)

        return np.argmax(b_entropy + f_entropy)

    def linearShade(self, x, left, right, min, max):
        """ as x goes from left to right, the output will go continuously from min to max
            and saturate if x<left or x>right
        """
        if x < left: return min
        elif x > right: return max
        else:
            m = (max - min) / (right - left)
            b = max - m * right
            return m * x + b

    @staticmethod
    def recastImage(imgArray, dtypeOut):
        """
        output an array where each value has been recast to a new data type without any other change
        The entire dynamic range of the image is remapped to the output bit depth. There is no clipping.
        :param imgArray: np.array of image data
        :param dtypeOut: str specifying output data type. Currently either 'uint16' or 'uint8'
        :return:
        """
        if dtypeOut == 'uint16':
           min,max = 0.99*np.amin(imgArray) ,1.01*np.amax(imgArray)
           m = 2**16/(max-min)
           b = 2**16-m*max
           mArray = np.full(imgArray.shape,m)
           bArray = np.full(imgArray.shape,b)
           return np.array(np.multiply(mArray,imgArray) + bArray).astype('uint16')
        elif dtypeOut == 'uint8':
            min, max = 0.99*np.amin(imgArray), 1.01*np.amax(imgArray)
            m = 2 ** 8 / (max - min)
            b = 2 ** 8 - m * max
            mArray = np.full(imgArray.shape, m)
            bArray = np.full(imgArray.shape, b)
            return np.array(np.multiply(mArray, imgArray) + bArray).astype('uint8')
        else: raise ValueError('recasting is only availabe to uint8 and uint16, not dtypeOut=',dtypeOut)


    def applyThreshold(self,recastBool = True,scaleFactor=1.0):
        """
        This function does not compute a threshold. It just takes self.imgArray and self.thresholdArray
        and outputs an 16 bit image of the threshold with optional recasting to image to 16 bit depth.
        :return:
        """
        # change type to enable subtraction
        out = self.imgArray.astype('float32') - scaleFactor*self.thresholdArray.astype('float32')
        # clip everything below zero
        positive = out # make a deep copy in case we also want to return the thresholded parts.
        negative = out * -1
        positive[positive<0] = 0 # now use logical indexing to reassign all negative values to zero
        negative[negative < 0] = 0
        if recastBool == True:
            positive = self.recastImage(positive,'uint16') # rescale the dynamic range after thresholding to span 16bits
            negative = self.recastImage(negative, 'uint16')  # rescale the dynamic range after thresholding to span 16bits
        return positive,negative


if __name__ == "__main__":
  fPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/'\
      'OddysseyHashScripting/stitch/smartCrop/smartCrop/tfrGel09052019b_shearRun05062019i_smartCrop_hv00040.tif'
  testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
                'OddysseyHashScripting/pyFiji/testImages'
  instance = arrayThreshold(fPath)
  #%%
  #pyFiji.send2Fiji(instance.imgArray,wdir=testImgPath)
  instance.imgArray = instance.recastImage(instance.imgArray,'uint16')
  import seaborn as sns
  import matplotlib.pyplot as plt
  sns.distplot(instance.imgArray.ravel())
  plt.show()
  print(pyFiji.send2Fiji(instance.imgArray,wdir=testImgPath))
  #%%
  tmp = instance.localThreshold(50,n_xyz=(3,3,3),parBool=True,n_jobs=8)
  instance.thresholdArray = instance.interpolateThreshold(tmp[0],tmp[1])
  positive, negative = instance.applyThreshold()
  print(pyFiji.send2Fiji(positive,wdir=testImgPath))

