import numpy as np
import pandas
from scipy import signal
import pyFiji
import threshold
import flatField
import seaborn as sns
from matplotlib import pyplot as plt

class particleGlyph:
    """
    A class for storing a an array of intensity with a strict ellipsoidal mask and defined intensity shading
    """
    def __init__(self,dim,boxDim):
        """
        :param dim: the dimensions of the ellipse (a,b,c)
        :param boxDim: the dimensions of the kernel.
        """
        # initalize a numpy array of dimensions dx,dy,dz
        # round up dz dy dz to odd numbers if input was even
        out = []
        for d in boxDim:
            if bool(d%2==0) == True: d=d+1
            out.append(d)
        [dz,dy,dx] = out
        glyphBox = np.zeros([dz,dy,dx],dtype='uint8')
        deltaKernel = np.zeros([dz,dy,dx],dtype='uint8')
        mask = np.zeros([dz,dy,dx],dtype='uint8')
        # put the ellipsoid in the center and define the shading. I think it should fall off linearly in intensity
        # scan through the indices in the box. If within the ellipsoid region, then decide on linear shading

        def ellipseInterior(pt,dim,center,out='bool'):
          (z,y,x) = pt
          (c,b,a) = [elt/2 for elt in dim]
          (cz,cy,cx) = center
          if out == 'bool':
            if ((x-cx)/a)**2 + ((y-cy)/b)**2 + ((z-cz)/c)**2 < 1: return True
            else: return False
          elif out == 'norm':
            return ((x-cx)/a)**2 + ((y-cy)/b)**2 + ((z-cz)/c)**2
          else: print("Unrecognized out kwrd: "+ out)

        def linearShade(x,left,right,min,max ):
            """ as x goes from left to right, the output will go continuously from min to max"""
            m = (max - min)/(right - left)
            b = max - m*right
            return m*x+b

        center = [elt/2 for elt in [dz,dy,dx]]
        deltaKernel[int(np.rint(center[0])),\
                    int(np.rint(center[1])),\
                    int(np.rint(center[2]))] = 1
        for z in range(dz):
          for y in range(dy):
            for x in range(dx):
                n = ellipseInterior((z,y,x),dim,center,out='norm')
                if n < 1:
                    glyphBox[z,y,x] = linearShade(n,0,1,10,0)
                    mask[z, y, x] = 1
        self.glyph = glyphBox
        self.mask = mask
        self.deltaKernel = deltaKernel
        self.shading = 'linearShading'
        self.dim = glyphBox.shape
        self.boundingBoxDim = boxDim

class pxLocations:
  """
  A class for holding particle location information and some basic metaData.
  All heavy lifting is done by pandas as reccommended by the python soft-matter
  community
  """
  def __init__(self,file,units='px',program='trackpy'):
    try:
        locArray = np.loadtxt(file)
        if program == 'katekilfoil':
            locdataframe = pandas.DataFrame(locarray[:, :3], columns= \
                ['x (' + units + ')', 'y (' + units + ')', 'z (' + units + ')'])
            kilfoilextra = pandas.DataFrame(locarray[:, 3:7], columns= \
                ['integrated intensity (au)', 'eccentricity ?', 'zeropx overlap'])
            self.kilfoilext = kilfoilextra
            self.locations = locdataframe
        elif program == 'trackpy':
            locdataframe = pandas.DataFrame(locarray[:, :3], columns= \
                ['z (' + units + ')', 'y (' + units + ')', 'x (' + units + ')'])
            self.locations = locdataframe
    except ValueError:
        if isinstance(file,pandas.core.frame.DataFrame): self.locations = file
        else:
            print("input type {} is not recognized. either filepath to img or np array is expected".format(type(file)))
            raise TypeError

class locationOverlay:
  """
  a class for creating a storing synthetic images from an array of location outputs
  """
  def __init__(self,locationpath,locatinginputpath,locatingprogram = 'katekilfoil'):
    # initalize a numpy array with the correct pixel dimensions, maybe padded
    try: inputImg = flatField.zStack2Mem(locatinginputpath)
    except ValueError:
        if isinstance(locatinginputpath,(np.ndarray, np.generic)): inputImg = locatinginputpath
        else:
            print("inpute type {} is not recognized. either filepath to img or np array is expected".format(type(locationpath)))
            raise TypeError
    (dz,dy,dx) = inputImg.shape
    imgArray = np.zeros((dz,dy,dx),dtype='uint8')
    # import the locations from a text file
    l = pxLocations(locationpath,program=locatingprogram)
    g = particleGlyph([7,7,7],[9,9,7])
    # from the pixel locations, place a predefined glyph at the center
    # assign a pixel value of 1 to each pixel that corresponds to a coordinate in location
    xcoord = np.rint(l.locations['x (px)']).astype(int)
    ycoord = np.rint(l.locations['y (px)']).astype(int)
    zcoord = np.rint(l.locations['z (px)']).astype(int)
    if locatingprogram == 'katekilfoil': imgArray[zcoord,xcoord,ycoord] = 1
    elif locatingprogram == 'trackpy': imgArray[zcoord,ycoord,xcoord] = 1
    else: raise KeyError
    # i assume that would look something like imgArray[xCoord,yCoord,zCoord] = 1, where xCoord is 1d array of all the x coordinates mapped to integers
    # now convolve with the glyph using split and fft methods in scipy
    self.glyphImage = signal.oaconvolve(imgArray,g.glyph).astype('uint8')
    self.inputPadded = threshold.arrayThreshold.recastImage(signal.oaconvolve(inputImg,g.deltaKernel),'uint8')
    self.deltaImage = imgArray.astype('uint8')

    # save the image and the array of singles

    # fill in some metaData attributes that will be used for exporting

if __name__ == "__main__":
    #locationPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/' \
    #               'locations/tfrGel09052019b_shearRun05062019i_locations_hv00002_gel_pxLocations.text'
    #locationPath ='/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
    #              'OddysseyHashScripting/pyFiji/orientation_trackpy.text'
    output_location_path = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/' \
                           'locations/tfrGel09052019b_shearRun05062019i_postDecon8Bit_hv00085_trackpy.text'
    inputImgPath ='/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/' \
                  'postDecon/tfrGel09052019b_shearRun05062019i_postDecon8Bit_hv00085.tif'
    #inputImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/' \
    #               'pyFiji/testImages/orientation.tif'
    testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
                  'OddysseyHashScripting/pyFiji/testImages/'
    #l = pxLocations(output_location_path,program='trackpy')
    overlay = locationOverlay(output_location_path,inputImgPath,locatingProgram='trackpy')
    #sns.distplot(l.kilfoilExt['integrated intensity (au)'],kde=False)
    #plt.show()

    print(pyFiji.send2Fiji([overlay.glyphImage,overlay.inputPadded],wdir=testImgPath))


