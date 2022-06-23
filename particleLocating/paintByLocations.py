import numpy as np
import pandas
from scipy import signal
from particleLocating import pyFiji, threshold, flatField
from particleLocating import dplHash_v2 as dpl
from particleLocating import particleGlyph
import seaborn as sns
from matplotlib import pyplot as plt

#class particleGlyph:
#    """
#    A class for storing a an array of intensity with a strict ellipsoidal mask and defined intensity shading
#    """
#    def __init__(self,dim,boxDim):
#        """
#        :param dim: the dimensions of the ellipse (a,b,c)
#        :param boxDim: the dimensions of the kernel.
#        """
#        # initalize a numpy array of dimensions dx,dy,dz
#        # round up dz dy dz to odd numbers if input was even
#        out = []
#        for d in boxDim:
#            if bool(d%2==0) == True: d=d+1
#            out.append(d)
#        [dz,dy,dx] = out
#        glyphBox = np.zeros([dz,dy,dx],dtype='uint8')
#        deltaKernel = np.zeros([dz,dy,dx],dtype='uint8')
#        mask = np.zeros([dz,dy,dx],dtype='uint8')
#        # put the ellipsoid in the center and define the shading. I think it should fall off linearly in intensity
#        # scan through the indices in the box. If within the ellipsoid region, then decide on linear shading
#
#        def ellipseInterior(pt,dim,center,out='bool'):
#          (z,y,x) = pt
#          (c,b,a) = [elt/2 for elt in dim]
#          (cz,cy,cx) = center
#          if out == 'bool':
#            if ((x-cx)/a)**2 + ((y-cy)/b)**2 + ((z-cz)/c)**2 < 1: return True
#            else: return False
#          elif out == 'norm':
#            return ((x-cx)/a)**2 + ((y-cy)/b)**2 + ((z-cz)/c)**2
#          else: print("Unrecognized out kwrd: "+ out)
#
#        def linearShade(x,left,right,min,max ):
#            """ as x goes from left to right, the output will go continuously from min to max"""
#            m = (max - min)/(right - left)
#            b = max - m*right
#            return m*x+b
#
#        center = [elt/2 for elt in [dz,dy,dx]]
#        deltaKernel[int(np.rint(center[0])),\
#                    int(np.rint(center[1])),\
#                    int(np.rint(center[2]))] = 1
#        for z in range(dz):
#          for y in range(dy):
#            for x in range(dx):
#                n = ellipseInterior((z,y,x),dim,center,out='norm')
#                if n < 1:
#                    glyphBox[z,y,x] = linearShade(n,0,1,10,0)
#                    mask[z, y, x] = 1
#        self.glyph = glyphBox
#        self.mask = mask
#        self.deltaKernel = deltaKernel
#        self.shading = 'linearShading'
#        self.dim = glyphBox.shape
#        self.boundingBoxDim = boxDim

class pxLocations:
  """
  A class for holding particle location information and some basic metaData.
  All heavy lifting is done by pandas as reccommended by the python soft-matter
  community
  """
  def __init__(self,file,units='px',program='trackpy',locColumns=['z','y','x'], sep=','):
    try:
        if program == 'katekilfoil':
            locArray = np.loadtxt(file)
            locdataframe = pandas.DataFrame(locArray[:, :3], columns= \
                ['x (' + units + ')', 'y (' + units + ')', 'z (' + units + ')'])
            kilfoilextra = pandas.DataFrame(locArray[:, 3:7], columns= \
                ['integrated intensity (au)', 'eccentricity ?', 'zeropx overlap'])
            self.kilfoilext = kilfoilextra
            self.locations = locdataframe
        elif program == 'trackpy':
            #locdataframe = pandas.DataFrame(locArray[:, :3], columns= \
            #    ['z (' + units + ')', 'y (' + units + ')', 'x (' + units + ')'])
            locdataframe = pandas.read_csv(file,sep=sep)
            self.locations = locdataframe[locColumns]
    except ValueError:
        if isinstance(file,pandas.core.frame.DataFrame): self.locations = file
        else:
            print("input type {} is not recognized. either filepath to img or np array is expected".format(type(file)))
            raise TypeError

class locationOverlay():
    """
    a class for creating a storing synthetic images from an array of location outputs
    """

    def __init__(self,metaDataPath: str, computer: str='IMAC'):
        self.dpl = dpl.dplHash(metaDataPath)
        self.computer= computer
        self.metaData = self.dpl.metaData['paintByLocations']

    def loadFromFile(self, hv: int):
        self.hv = hv

        # read in all columns to self.locOutput
        # reserve self.locations for pandas slices of self.locOutput.loc[boolean selection][locColumns]
        sep,loc_columns = self.metaData['sep'], self.metaData['locColumns']
        locationInputPath = self.dpl.getPath2File(hv, kwrd='visualize', computer=self.computer, extension='locInput.tif')

        _refineLocExt = '_' + self.dpl.sedOrGel(hv) + "_trackPy_lsqRefine.csv"
        refinePath = self.dpl.getPath2File(hv, kwrd='locations', extension=_refineLocExt, computer=self.computer)

        #self.locations = pandas.read_csv(refinePath,sep=sep)[loc_columns]
        self.locOutput = pandas.read_csv(refinePath,sep=sep)
        self.inputImg = flatField.zStack2Mem(locationInputPath)

    def makeGlyphImg(self):
        """

        """

        # read in some metaData parameters
        mat = self.dpl.sedOrGel(self.hv)
        dim = self.metaData[mat]['dim']
        boxDim = self.metaData[mat]['boxDim']
        locColumns = self.metaData['locColumns']

        # create glyph object
        g = particleGlyph.particleGlyph(dim,boxDim)

        # name keys for indexing pandas df of locations
        z_key, y_key, x_key = locColumns

        # get input image dimensions and initialize output padded imgArray
        dz, dy, dx = self.inputImg.shape
        #deltaImg = np.pad(np.zeros((dz, dy, dx), dtype='uint8'), 1)
        # try padding after assignement of particle centers
        deltaImg = np.zeros((dz, dy, dx), dtype='uint8')

        # what are the positions that have centers within the edges of the
        # image so that I can do the convolution with glyph
        inbound_loc = self.locations[(self.locations[z_key] < dz) &
                                     (self.locations[y_key] < dy) &
                                     (self.locations[x_key] < dx)]

        # some simple rounding and type conversion to use coord as int indices
        zcoord = np.rint(inbound_loc[z_key]).astype(int)
        ycoord = np.rint(inbound_loc[y_key]).astype(int)
        xcoord = np.rint(inbound_loc[x_key]).astype(int)

        # assign center locations a value of 1
        deltaImg[zcoord, ycoord, xcoord] = 1

        # now pad to get the same dimesions
        deltaImg = np.pad(deltaImg,1)

        # convolve delta field with glyph
        self.glyphImage = signal.oaconvolve(deltaImg, g.glyph).astype('uint8')
        self.inputPadded = threshold.arrayThreshold.recastImage(
            signal.oaconvolve(np.pad(self.inputImg, 1), g.deltaKernel), 'uint8')
        self.deltaImage = deltaImg.astype('uint8')

    def send2pyFiji(self,**kwargs):
        pyFiji.send2Fiji([self.glyphImage, self.inputPadded, self.deltaImage], **kwargs)

    def runImac(self, hv: int):
        self.loadFromFile(hv)
        self.makeGlyphImg()
        self.send2pyFiji()

    """
    
    def __init__(self,locationpath,locatinginputpath,locatingprogram = 'katekilfoil',locColumns=['z','y','x'],sep=','):
      # initalize a numpy array with the correct pixel dimensions, maybe padded
      try: inputImg = flatField.zStack2Mem(locatinginputpath)
      except ValueError:
          if isinstance(locatinginputpath,(np.ndarray, np.generic)): inputImg = locatinginputpath
          else:
              print("inpute type {} is not recognized. either filepath to img or np array is expected".format(type(locationpath)))
              raise TypeError
      (dz,dy,dx) = inputImg.shape
      imgArray = np.pad(np.zeros((dz,dy,dx),dtype='uint8'),1)
      # import the locations from a text file
      l = pxLocations(locationpath,program=locatingprogram,locColumns=locColumns,sep=sep)
      g = particleGlyph([18,14,14],[22,18,18])
      # from the pixel locations, place a predefined glyph at the center
      # assign a pixel value of 1 to each pixel that corresponds to a coordinate in location

      # this should be changed to a dictionary based lookup, not a list of keys
      z_key, y_key, x_key = locColumns
      inbound_loc = l.locations[(l.locations[z_key] < dz) &
                                (l.locations[y_key] < dy) &
                                (l.locations[x_key] < dx)]
      #zcoord = np.rint(l.locations[z_key]).astype(int)
      #ycoord = np.rint(l.locations[y_key]).astype(int)
      #xcoord = np.rint(l.locations[x_key]).astype(int)
      zcoord = np.rint(inbound_loc[z_key]).astype(int)
      ycoord = np.rint(inbound_loc[y_key]).astype(int)
      xcoord = np.rint(inbound_loc[x_key]).astype(int)
      if locatingprogram == 'katekilfoil': imgArray[zcoord,xcoord,ycoord] = 1
      elif locatingprogram == 'trackpy': imgArray[zcoord,ycoord,xcoord] = 1
      else: raise KeyError
      # i assume that would look something like imgArray[xCoord,yCoord,zCoord] = 1, where xCoord is 1d array of all the x coordinates mapped to integers
      # now convolve with the glyph using split and fft methods in scipy
      self.glyphImage = signal.oaconvolve(imgArray,g.glyph).astype('uint8')
      self.inputPadded = threshold.arrayThreshold.recastImage(signal.oaconvolve(np.pad(inputImg,1),g.deltaKernel),'uint8')
      self.deltaImage = imgArray.astype('uint8')
      """

      # save the image and the array of singles

      # fill in some metaData attributes that will be used for exporting

if __name__ == "__main__":
    """
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
    """

    # input information
    metaPath = '/Volumes/PROJECT/tfrGel23042022/strainRamp/f/dplPath/tfrGel23042022_shearRun01052022f_imageStack_metaData.yaml'
    hv = 25 #pick a random example

    # initialize
    inst = locationOverlay(metaPath)

    # check hv information. sed/gel where in the sample it?
    inst.dpl.hash_df.loc[hv]

    # load images including locInput and locations
    inst.loadFromFile(hv)

    # select columns of self.locOutput
    inst.locations = inst.locOutput[inst.metaData['locColumns']]

    # make glyph image, store in class attribute
    inst.makeGlyphImg()

    # save to tiff and copy fiji commands to clipboard
    inst.send2pyFiji(wdir='/Volumes/PROJECT/tfrGel23042022/strainRamp/f/pyFiji')


