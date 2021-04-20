# decide on the crop parameters based on the raw image
# go to post-decon, and take only the sediment hv that are next to the gel
# from the particle located gel particles, zero them out. in fact, zero out the entire fitted plane
# check the stack
# locate the particles with the bottom of the image almost completely black.
# this is really a masking operation.

# precompute the top surface of the gel using existing da.fitTopSurface
# if not present, compute it
# class inheriticance of dplHash class
# for each sed hv that is overlapping with the gel
# determine the plane of the gel that is intersecting the hashvalue
# zero out the image below this plane, pixel by pixel
# run the same thresholding and what not procedure as the typical particle locating.

# v2: Is there a way to find the sed/gel interface without particle locating?
# it would be easiest to just make this part of the hashing process...
# make overlapping hashes and then mask the interfacial hashes, and then continue on
# Ideally this should be done during the background substraction phase, maybe even before deconvolution.

# what is masked the lcoated gel particles and then rehashed?

from particleLocating import dplHash_v2 as dpl
import data_analysis as da
import numpy as np
import numba
import trackpy as tp
import pandas as pd
import subprocess
from abc import ABC, abstractmethod
from typing import List, Dict
import yaml
import h5py
import tifffile
import pandas as pd
from trackpy import masks
from numba import jit
from typing import List, Tuple


class ilastikClassifier(ABC):

    def __init__(self, run_args: Dict, exec: str = '', project: str = ''):
        """
        Abstract class specifying the namespace and methods for any ilastik neural network
        that will be called within python as a subprocess.

        Arguments

            run_args: dictionary of keyword arguements that may need to reordered and formatted in order to run
                      prior to calling run -> subprocess.call(args).
                      This maybe different depending on the classifier type and hence is a flexible dictionary
                      within the namespace.
                      This is likely loaded from meta yaml file and parsed outside the class.
            exec: path to ilastik.sh
            project: path to trained neural network
        """
        self.exec = exec  # specify the path to the ilastik executable. This is the same for every ilastik classifier type
        self.project = project  # specify the path to trained neural network. This is also required for every ilastik classifier type
        self.run_args = run_args
        # what does super() do? In this case call the method __int__() from the parent class...but the parent class
        # is an abstract class so I am not sure.
        super().__init__()

    @abstractmethod
    def run(self, arg_list: List[str]):
        pass

    @abstractmethod
    def parseArgs(self, inputFile: str = ''):
        pass


class pixelClassifier(ilastikClassifier):
    # def __init__(self, exec, project ):
    #    # we need some kind of call to python subprocess module
    #    # I think that works in a straightforward way:
    #    # actually the initialization should only involve setting up the list of
    #    # string input arguments for calling subprocess, and not the actual call
    #    # which will be a run command based on the minimum requirements outlined
    #    # in the abstractmethod defined for all ilastikLoaders.
    #
    #    # UPDATE: you dont have to define init again for pixelClassifier with inheritance from
    #    # abstract class (ABC) ilastikLoader
    #    """
    #    >>> {ilasticExec}.sh --headless
    #                         --project=path/to/project/file.ilp
    #                         --export_source=Probabilities
    #                         --output_format=hdf5
    #                         --output_filename_format={dataset_dir}/{nickname}_probabilities.hdf
    #                         /path/to/tiff/stack.tif
    #    >>> ./run_ilastik.sh --headless --project=/Volumes/TFR/tfrGel10212018A_shearRun10292018f/interfaceLocating/pixelClassification_interfaceTimeSeries_deconCrop/MyProject.ilp /Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/tfrGel10212018A_shearRun10292018f_decon_hv00050.tif
    #    """
    #    subprocess.run()
    def parseArgs(self, inputFile):
        arg_list = []
        args = self.run_args

        arg_list.append(self.exec)
        if args['headless'] is True: arg_list.append('--headless')
        # possible that this line depends on the version of ilastik.
        if args['readonly'] is True: arg_list.append('--readonly=True')

        arg_list.append('--project={}'.format(self.project))
        arg_list.append('--export_source={}'.format(args['export_source']))
        arg_list.append('--output_format={}'.format(args['output_format']))
        arg_list.append('--output_filename_format={}'.format(args['output_filename_format']))
        arg_list.append(inputFile)
        return arg_list

    def run(self, argList):
        # def run(self, argList, stdOut: str):
        """
        Run pixel classifier
        This method has access to namespace of iliastikClassifier...ie exec, project and run_args
        It just needs to take that information and create a list of strings to pass to subprocess.
        There should be another method that parses run_args and returns a list of strings to be
        passed to run. Maybe this should also be requried as an abstract method within mother class
        of ilastikClassifier

        I should have this return the path to the pixel prediction map...and call subprocess
        """
        subprocess.call(argList)
        # with open(stdOut as f,'w'):
        #    subprocess.call(argList, stdout=f)
        return argList[-1]


def test_PxClassifier():
    exec = '/Applications/ilastik-1.4.0b13-OSX.app/Contents/ilastik-release/run_ilastik.sh'
    project = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/interfaceLocating/pixelClassification_multiLaneTraining/tfrGel09052019a_shearRun10292018f_pxClassifier.ilp'
    yamlMetaPath = '/Users/zsolt/Colloid_git/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'

    input = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/tfrGel10212018A_shearRun10292018f_decon_hv00050.tif'

    with open(yamlMetaPath, 'r') as f:
        run_args = yaml.load(f)['ilastik']['pxClassifier']['run_args']
    irunner = pixelClassifier(run_args, exec, project)
    irunner.run(irunner.parseArgs(input))


class ilastikThreshold(dpl.dplHash, pixelClassifier):
    def __init__(self, metaDataPath, computer='ODSY'):
        self.dpl = dpl.dplHash(metaDataPath)
        self.computer = computer
        self.ilastik_meta = self.dpl.metaData['ilastik']
        print("Initializing ilastik threshold")

    def _sethv(self, hv: int):
        """ This is a hacked together solution because I didnt originally seperate setting hv
        and reading pxprob and decon for needing to use threshold class for getIlastikPath...sigh"""
        self.hv = hv

    def setHashValue(self, hv: int):
        self.hv = hv
        # load decon and px probblity from file
        self._readDecon(self.getPathIlastik('decon'))
        self._readPxProbability(self.getPathIlastik('pxProb'))
        print("Setting hashValue for ilastik threshold")
        return 'HashValue is {} and decon and pxProbabilities are loaded into class'.format(self.hv)

    @staticmethod
    def ilastik_h5_numpy(h5file: str):
        with h5py.File(h5file, mode='r') as f:
            tmp = f['exported_data'][:]
        return np.array(tmp)

    def getPathIlastik(self, kwrd: str):
        """
        kwrd:
            [+] exec, path to run_ilastik.sh
            [+] project, path to *.ilp file
            [+] decon, path to decon input tif file
            [+] pxProb, path to hdf5 output px probability file
        """
        if kwrd == 'exec':
            return self.dpl.metaData['filePaths']['ilastik_{}'.format(self.computer)]
        elif kwrd == 'project':
            _dir = self.dpl.metaData['filePaths']['calibrationDirectory_{}'.format(self.computer)]
            dir_frmt = _dir + '/ilastik/{}'
            ilp = self.ilastik_meta['pxClassifier']['classifier_fName']
            return dir_frmt.format(ilp)

        elif kwrd == 'decon' or kwrd == 'input':
            return self.dpl.getPath2File(self.hv, kwrd='decon', computer=self.computer)

        elif kwrd == 'pxProb':
            """ path to pixel probability will depend on the path to the input"""
            _fName = self.dpl.getPath2File(self.hv, kwrd='decon', computer=self.computer)
            fName_frmt = _fName.split('.')[0] + '_{}'
            _ext = self.ilastik_meta['pxClassifier']['run_args']['output_filename_format'].split('_')[-1]
            return fName_frmt.format(_ext)
        else:
            raise KeyError('{} is not recognized as file path option!'.format(kwrd))

    def _runClassifier(self):
        input = self.getPathIlastik('input')
        run_args = self.ilastik_meta['pxClassifier']['run_args']
        exec = self.getPathIlastik('exec')
        project = self.getPathIlastik('project')

        irunner = pixelClassifier(run_args, exec, project)
        irunner.run(irunner.parseArgs(input))
        return True

    def _readPxProbability(self, hdf5Path: str):
        """
        load pixel probabilities from h5path
        assign to namespace as numpy array or dictionary of numpy arrays

        default path:
        /Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/tfrGel10212018A_shearRun10292018f_decon_hv00050_probabilities.h5
        """
        self.pxProb = self.ilastik_h5_numpy(hdf5Path)
        self.pxLabel = self.ilastik_meta['pxClassifier']['channels']

    def _readDecon(self, deconPath: str):
        """
        load decon image that is paired with px classifier

        debug
        /Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/tfrGel10212018A_shearRun10292018f_decon_hv00050.tif
        """
        self.decon = tifffile.imread(deconPath)

    def threshold(self, mat: str):
        """
        sed_prob = prob[:,:,:,2] + prob[:,:,:,3] # sum up probabilities on sedColloid and sed_background channels
        decon_threshold = np.where(sed_prob <0.7, 0, decon)
            # if sed prob is less than 0.7 threshold,
            # send the pixel values of decon to zero. Otherwise accept decon pixel values
        """

        tmp = self.ilastik_meta['pxClassifier']['sedGelSplit']
        w = tmp['weights'][mat]
        cutoff = tmp['cutoff'][mat]

        prob = np.zeros_like(self.pxProb[:, :, :, 0])
        for i in range(len(w)): prob = prob + w[i] * self.pxProb[:, :, :, i]
        print("ilastik thresholded hv {} for material {}".format(self.hv, mat))

        return np.where(prob < cutoff, np.nan, self.decon)


class ilastikIntegrate(dpl.dplHash):
    def __init__(self, metaDataPath, computer='ODSY'):
        self.dpl = dpl.dplHash(metaDataPath)
        self.computer = computer
        self.meta = self.dpl.metaData['ilastik']

    def setHashValue(self, hashValue):
        self.hv = hashValue
        self.mat = self.dpl.sedOrGel(self.hv)

    def getPathIlastik(self, kwrd: str):
        """
        kwrd:
            [+] exec, path to run_ilastik.sh
            [+] project, path to *.ilp file
            [+] decon, path to decon input tif file
            [+] pxProb, path to hdf5 output px probability file
        """
        if kwrd == 'exec':
            return self.dpl.metaData['filePaths']['ilastik_{}'.format(self.computer)]
        elif kwrd == 'project':
            _dir = self.dpl.metaData['filePaths']['calibrationDirectory_{}'.format(self.computer)]
            dir_frmt = _dir + '/ilastik/{}'
            ilp = self.meta['pxClassifier']['classifier_fName']
            return dir_frmt.format(ilp)

        elif kwrd == 'decon' or kwrd == 'input':
            return self.dpl.getPath2File(self.hv, kwrd='decon', computer=self.computer)

        elif kwrd == 'pxProb':
            """ path to pixel probability will depend on the path to the input"""
            _fName = self.dpl.getPath2File(self.hv, kwrd='decon', computer=self.computer)
            fName_frmt = _fName.split('.')[0] + '_{}'
            _ext = self.meta['pxClassifier']['run_args']['output_filename_format'].split('_')[-1]
            return fName_frmt.format(_ext)
        else:
            raise KeyError('{} is not recognized as file path option!'.format(kwrd))

    def _readPxProb(self, hdfPath: str = None) -> None:
        """This needs to be cropped in the same way locations will be cropped
           Which should be just fft bool.
        """
        if hdfPath is None: hdfPath = self.getPathIlastik('pxProb')
        smartCropMeta = self.dpl.metaData['smartCrop']

        if smartCropMeta['fftCrop']['bool']:
            print("Warning, assuming all cropping on between ilastik px prob and location input img is just fft crop")

            pxProb_uncropped = ilastikThreshold.ilastik_h5_numpy(hdfPath)
            cz, cy, cx = [smartCropMeta['fftCrop'][x] for x in ['Z', 'Y', 'X']]
            nz, ny, nx, nc = pxProb_uncropped.shape

            self.pxProb = pxProb_uncropped[cz:nz - cz, cy: ny - cy, cx:nx - cx, :]

        else:
            self.pxProb = ilastikThreshold.ilastik_h5_numpy(hdfPath)

    @staticmethod
    @jit(nopython=True, cache=False)
    def _intPxProb_numba(loc: np.array, img: np.array, \
                         Nz: Tuple, Ny: Tuple, Nx: Tuple) -> np.array:

        out = np.zeros(loc.shape[0])
        # loop over locations
        for N in range(loc.shape[0]):
            pos = loc[N]
            prob_total = 0
            cz, cy, cx = [int(coord) for coord in np.rint(pos[0:3])]
            for n in range(len(Nx)):
                prob_total += img[cz + Nz[n], \
                                  cy + Ny[n], \
                                  cx + Nx[n]]
            out[N] = prob_total
        return out

    def integratePxProb(self, pos_df: pd.DataFrame, \
                        pos_keys: Dict = {'z': 0, 'y': 1, 'x': 2}) -> pd.DataFrame:
        """
        For a dataFrame of particle positions and sizes, integrate each of the pxProb channels separately
        and return a dataFrame with same index as loc, but additional columns corresponding to the integrated intensity
        of ilastik pxProbability maps. (should be N x 6 dataFrame for N particles with 6 ilastik channels.
        """

        # form a mask using trackpy masks.binary
        def _prepMask(subType: str) -> np.array:
            if subType == 'gel':
                r = tuple(self.dpl.metaData['locating']['pxClassifier']['integrate_mask']['gel'])
            elif subType == 'sed_core':
                r = tuple(self.dpl.metaData['locating']['pxClassifier']['integrate_mask']['sed']['core'])
            elif subType == 'sed_shell':
                r = tuple(self.dpl.metaData['locating']['pxClassifier']['integrate_mask']['sed']['shell'])
            return masks.binary_mask(r, 3)

        pos_np = pos_df[pos_keys.keys()].to_numpy()
        N_channels = self.pxProb.shape[-1]  # Caution: indices on h5 is z,y,x,channels, which is not slow -> fast

        if self.mat == 'gel':
            prob_np = np.zeros((pos_np.shape[0], N_channels))
            mask = _prepMask('gel')
            Nz, Ny, Nx = mask.nonzero()

            for cName, index in self.meta['pxClassifier']['channels'].items():
                img = self.pxProb[:, :, :, index]
                prob_np[:, index] = self._intPxProb_numba(pos_np, img, Nz, Ny, Nx)

            return pd.DataFrame(data=prob_np, index=pos_df.index, columns=self.meta['pxClassifier']['channels'].keys())

        elif self.mat == 'sed':
            prob_np = np.zeros((pos_np.shape[0], 2 * N_channels))

            # sed_core
            mask = _prepMask('sed_core')
            Nz, Ny, Nx = mask.nonzero()
            for cName, index in self.meta['pxClassifier']['channels'].items():
                img = self.pxProb[:, :, :, index]
                prob_np[:, index] = self._intPxProb_numba(pos_np, img, Nz, Ny, Nx)

            # sed_shell
            mask = _prepMask('sed_shell')
            Nz, Ny, Nx = mask.nonzero()
            for cName, index in self.meta['pxClassifier']['channels'].items():
                img = self.pxProb[:, :, :, index]
                prob_np[:, index + N_channels] = self._intPxProb_numba(pos_np, img, Nz, Ny, Nx)

            _keys = self.meta['pxClassifier']['channels'].keys()
            col_names = ['{}_{}'.format(stem, subtype) for subtype in ['core', 'shell'] for stem in _keys]

            return pd.DataFrame(data=prob_np, index=pos_df.index, columns=col_names)


def test_ilastikIntegrate():
    metaPath = '/Users/zsolt/Colloid_git/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'
    probPath = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/tfrGel10212018A_shearRun10292018f_decon_hv00050_probabilities.h5'
    hv = 50
    postDeconOutputPath = ''

    # inialize instance
    integrateInst = ilastikIntegrate(metaPath, computer='IMAC')

    # set hashValue
    integrateInst.setHashValue(50)

    # read px proobabilty
    integrateInst._readPxProb(probPath)

    # run integratePxProb on on locMicro that has been passed to the function.
    return integrateInst
    # create indices of nonzero elts using i,j,k = mask.nonzero()
    # set up some things, like indices and labels for numba implementation
    # and convert dataFrame to numpy
    #
    # in numba:
    # loop over center of features
    # round the position to nearest integer
    # index into the image and loop (1D!) over len(i) and use:
    # mass += img[cz + i[n], cy + j[n], cz + k[n]]
    # do this for each of the px probability channels.
    #
    # convert to dataFrame and return


class interfaceLocate(dpl.dplHash):
    def __init__(self, metaDataPath, computer='IMAC'):
        self.dpl = dpl.dplHash(metaDataPath)
        self.computer = computer
        self.gelSurface = None

    def interfaceTrainingHV(self, hvList=None):
        if hvList is None:
            """
           # contains some hard coded values specific to hashing used currently
           #   -> 5x5 hashing (x and y selection or corners and center)
           #   -> location of interface in z (z index of 1 or 2)
           #   -> time range 0 to 89 for frame number
           # currently opting to just pash the results as fixed values
           # zsolt, March 13 2021
           
           for x in [0,2,4]:
             for y in [0,2,4]:
               for z in [1,2]:
                 for t in [0,89]:
                   if x == y: hvList.append((x,y,z,t))
                   if x ==4 and y == 0: hvList.append((x,y,z,t))
                   if y== 4 and x == 0: hvList.append((x,y,z,t)) 
           """
            hvList = [25, 11150, 50, 11175, 45, 11170, 70, 11195, 37, 11162, 62, 11187, 29, 11154, 54, 11179, 49, 11174,
                      74, 11199]
        out = []
        for hv in hvList:
            tmp = self.dpl.queryHash(hv)
            x, y, z, t = tmp['index']
            out.append([hv, tmp['material'], x, y, z, t, self.dpl.sedGelInterface(hv)[1]])

        return pd.DataFrame(data=out, columns=['hv', 'material', 'x', 'y', 'z', 't', 'interfaceBool']).sort_values(
            'hv').reset_index(drop=True)

    def readGelSurface(self, path):
        self.gelSurface = pd.read_csv(path)

    def maskGel(self, hv, topSurfaceEqn):
        """
        For a given hashValue hv, this returns a boolean mask with zero for everything
        that is not the material type specified by hv. That is, hv is a gel, then everything
        *above* the topSurfaceEqn will be masked to zero, while if it sed, then everything below
        topSurfaceEqn will be masked.
        """

        # from hv, get image coordinates and material type by querying dplHash

        # get equation for the plane
        m = lambda p0, p1: (p0[0] - p1[0]) / (p0[1] - p1[1])
        b = lambda p0, p1: p1[1] - m(p0, p1) * p1[0]

        """
        SCRATCH
        
        # %%
        m = lambda p0, p1: (p0[0] - p1[0]) / (p0[1] - p1[1])
        b = lambda p0, p1: p1[1] - m(p0, p1) * p1[0]

        # %%
        m_val = m(line_dict['p0'], line_dict['p1'])
        b_val = b(line_dict['p0'], line_dict['p1'])

        # %%
        out = np.ones((6, 6), dtype=bool)
        for ny in range(6):
            for nx in range(6):
                if (ny + 1) < (m_val * (nx + 1) + b_val): out[ny, nx] = False
        # print(out)
        print(a * out)
        """

        # for each z-slice, evaluate the plane equation at z-value
        # Evaluate eqn on range of full coordinates in the hashValue.
        # eg if hash value has xdim of 100, the image coords might be (682,782)
        # and evaluate on (682,782), not (0,100)

        # translate coordinates equation to hash dimensions in xy

        # init mask to True and reassign False using floor and ceiling
        # applied to z coordinate of given xy value.
        # This is like making a level set on z slices.
        # repeat for all the z-slice

        # this should be sped up for jit

        return True

    def maskGelParticle(self, gelPos_stitched_df, rawImage):
        """
        This function masks all the particle in the gelPos_stitched dataFrame
        It is very similar to the iterative part of particle locating, or paint by locations
        but with a more generous mask.
        All fixed parameters are read from the metaDataYaml file associated with the class.

        Param:

        return: rawImage array with masked array positions.
        """
        # convert gelPos_stitched_df to np

        #

        return True
