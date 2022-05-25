import numpy as np

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
