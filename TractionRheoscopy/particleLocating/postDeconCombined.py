import dplHash_v2 as dpl
import locating
import paintByLocations
import flatField
import yaml
import pyFiji
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

    def visualize(self,computer='ODSY'):
        self.overlay = paintByLocations.locationOverlay(self.locations[0],self.postDecon[0], locatingprogram = 'trackpy')
        #testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
        #              'OddysseyHashScripting/pyFiji/testImages/'
        #visualizePath = self.dpl.getPath2File(self.hashValue,kwrd='visualize',computer=computer,pathOnlyBool=True)
        #testImgPath = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/visualize'
        #if computer != 'ODSY' or computer != 'AWS':
        #    print(pyFiji.send2Fiji([self.overlay.glyphImage, self.overlay.inputPadded],
        #                           wdir=visualizePath,
        #                           metaData=yaml.dump(self.dpl.metaData, sort_keys = False)))
        #fName_glyph = self.dpl.metaData['fileNamePrefix']['global'] + 'vGlyph_' + 'hv{}'.format(str(self.hashValue).zfill(4)) + '.tif'
        fName_glyph = self.dpl.getPath2File(self.hashValue,kwrd='visualize', computer=computer, extension='visGlyph.tif')
        fName_locInput = self.dpl.getPath2File(self.hashValue,kwrd='visualize', computer=computer, extension='visLocInput.tif')
        #fName_locInput = self.dpl.metaData['fileNamePrefix']['global'] + 'vLocInput' + 'hv{}'.format(str(self.hashValue).zfill(4)) + '.tif'
        flatField.array2tif(self.overlay.inputPadded,
                            fName_locInput,
                            metaData = yaml.dump(self.dpl.metaData, sort_keys = False))
        flatField.array2tif(self.overlay.glyphImage,
                            fName_glyph,
                            metaData = yaml.dump(self.dpl.metaData, sort_keys = False))

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
