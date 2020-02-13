import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class particleGlyph:
    """
    A class for storing a an array of intensity with a strict ellipsoidal mask and defined intensity shading
    """
    def __init__(self,dim,boxDim):
        # initalize a numpy array of dimensions dx,dy,dz
        # round up dz dy dz to odd numbers if input was even
        out = []
        for d in boxDim:
            if bool(d%2==0) == True: d=d+1
            out.append(d)
        [dx,dy,dz] = out
        [a,b,c] = dim
        glyphBox = np.zeros([dx,dy,dz],dtype='uint8')
        # put the ellipsoid in the center and define the shading. I think it should fall off linearly in intensity
        # scan through the indices in the box. If within the ellipsoid region, then decide on linear shading

        def ellipseInterior(pt,dim,center,out='bool'):
          (x,y,z) = pt
          (a,b,c) = [elt/2 for elt in dim]
          (cx,cy,cz) = center
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

        center = [elt/2 for elt in [dx,dy,dz]]
        for z in range(dz):
          for y in range(dy):
            for x in range(dz):
                n = ellipseInterior((x,y,z),dim,center,out='norm')
                if n < 1: glyphBox[x,y,z] = linearShade(n,0,1,255,0)
        self.glyph = glyphBox
        self.shading = 'linearShading'
        self.dim = glyphBox.shape
        self.boundingBoxDim = boxDim

class pxLocations:
  """
  A class for holding particle location information and some basic metaData.
  All heavy lifting is done by pandas as reccommended by the python soft-matter
  community
  """
  def __init__(self,file,units='px'):
    locArray = np.loadtxt(fName=file)
    locDataFrame = pandas.DataFrame(locArray[:, :3], columns=['x (' + units +')', 'y (' + units +')', 'z (' + units +')'])
    kilfoilExtra = pandas.DataFrame(locArray[:, 3:7], columns=['integrated intensity (au)','eccentricity ?' , 'zeroPx Overlap')

  self.locations = locDataFrame
  self.kilfoilExt = kilfoilExtra


class locationOverlay:
  """
  A class for creating a storing synthetic images from an array of location outputs
  """
  def __init__(self,(dx,dy,dz),locationPath):
    # initalize a numpy array with the correct pixel dimensions, maybe padded
    imgArray = numpy.zeros((dz,dy,dx),dtype='uint8')
    # import the locations from a text file
    locations =
    # from the pixel locations, place a predefined glyph at the center
    # fill in some metaData attributes that will be used for exporting

if __name__ == "__main__":
    print("This is a little bit like paint by numbers")
    # Visuliaze a single glyph box
    ellipsoid = particleGlyph([2,2,4],[10,10,10])
    print(ellipsoid.dim)
    plt.gray() # set colormap to grayscale default
    for slice in range(ellipsoid.dim[2]):
      #img = Image.fromarray(ellipsoid.glyph[:,:,slice],'L')
      #img.show()
      plt.imshow(ellipsoid.glyph[:,:,slice])
      plt.show()
    # for a list of xyz coordinates, create and slice through images of glyphs at the centers with input dimensions.
    # save to tiff