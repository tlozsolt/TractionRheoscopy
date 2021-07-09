import numpy as np
import pandas as pd
import xarray as xa
import yaml

from particleLocating import flatField
from particleLocating import pyFiji

# create a class to store particle locations and the associated tiff images that were used as input
# to particle locating.

class cg2ml_train()

    def __init__(self, metaDataFile):
        self.meta = yaml.load(metaDataFile)
        self.locations = None # pandas dataFrame
        self.input_img = None # np array
        self.particle_param = None # dict read from yaml file

    def _loadImg(self):
        """
        initialize the img by reading the path from metaDataFile key imgPath
        """
        path = self.meta['imgPath']
        self.input_img = flatField.zStack2Mem(path)

    def _loadParam(self):
        """
        Load information on the particle size and array size we are going to cut out
        This information is fixed for the specific colliod sample as all particles are
        nominally the same size and the bounding box is likewise going to be the same
        for each particle. It does not include information that is specfic to each of
        particles.

        - x y and z dim of the particle in pixels
        - x y and z dim of the bounding box in pixels
        - ??
        -
        """
        self.particle_param = self.meta['particleParam']

    def _loadLoc(self):
        """
        This simply read the locations from an h5 file
        """
        path = self.meta['locationPath']
        self.locations = pd.read_hdf(path)

    def cutArray(self):
        """
        Given an array and a list of locations, split the array into small pieces of a given size with the
        array centered on the location of the particle

        Ideally this would be a xarray with keyed labels on numeric array
        Also, should work with xyzt labels
        Also, maybe the part where I loop over the particle should be implemented in numba
        and this function should be a wrapper?

        Output:
        shift vector returning piece to chunk, array of small piece
        """

if __name__ == "__main__":

    #%%
    a = np.random.random(15)
    d = 3 # chunk size
    dx = int(d/2) # 1
    stop = np.int(a.shape[0] -d/2 ) #
    start = int(d/2)

    out = np.zeros((np.int(n-2*dx + 1),d))

    for chunk in range(1,14,1):
        out[chunk] = a[chunk-dx:chunk+dx + 1]


    # simple test for chunking loop


