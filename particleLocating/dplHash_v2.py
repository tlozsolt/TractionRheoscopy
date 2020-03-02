import yaml, os, math, glob, flatField, re
from datetime import datetime
"""
This file is used to create a deconvolution particle locating (dpl) Hash table
for a given set of colloid microscopy xyzt stacks and complete metaData yaml file.
The class generates a hashtable that when given a positive number could yield:
    key: [+] complete and debugged, [-] partially complete, [ ] not yet started, [X] plan changed, not necessary
    
    [+] script to preoprocess and crop a given xyzt chunk (imageJ)
    [+] script to deconvolve the images with the parameters listed 
        in metaData (imageJ or deconvolution Lab 2)
    [+] script to further process decon output to format matlab input tiff (ie 8 bit tiff, thresholded background) (imageJ)
    [+] particle locate using iterative locating in matlab 
    [+] bash script to execute these scripts in order on oddyysey. 
    [+] submission scripts (yes plural if more than 10k chunks are needed) that handles
          resource request, error and output, module loading and submit bash script to execute 
          preprocessing, deconvolution, postDecon processing, and particle location
    [+] some kind of makeAll function that will create all the necessary scripts in the correct directories and optionally submit them
          and record job submission numbers etc.
    [+] migrate git repo to github and away from code.harvard.edu
    [+] write change on odsy and push to local branch 
"""

class dplHash:
  """ a class for deconvolution and particle locating 
      large xyzt confocal data in spatial chunks 
  """
  def __init__(self, metaDataPath):
    """ 
      initialize the instance with a  hashTable dictionary and metaData dictionaryf keys
      self.hash(str(hashValue)) returns a dictionary with keys "xyztCropTuples" and "material" 
      self.metaData gives a dictionary representation of the associated yaml file
    """
    with open(metaDataPath,'r') as stream: self.metaData=yaml.load(stream,Loader=yaml.SafeLoader)
    # compute the total hash size
    timeSteps = self.metaData['imageParam']['timeSteps']
    # xy chunks are directly computed given chunk size, minimum overlap, and full xy dimensions
    xyDim_full = [self.metaData['imageParam']['xDim'],self.metaData['imageParam']['yDim']]
    #xyDim_crop = [self.metaData['deconvolution']['deconCropXDim'], self.metaData['deconvolution']['deconCropYDim']]
    xyDim_gel = self.metaData['hash']['dimensions']['gel']['xyz'][:2]
    xyDim_sed = self.metaData['hash']['dimensions']['sed']['xyz'][:2]
    if xyDim_gel == xyDim_sed: xyDim_crop = xyDim_sed
    else:
      print("Material dependent xy dimensions for gel vs sed are not supported yet")
      raise KeyError

    #xyDim_minOverlap = [self.metaData['deconvolution']['minXOverlap'],self.metaData['deconvolution']['minYOverlap']]
    xyzDim_minOverlap_gel = self.metaData['hash']['dimensions']['gel']['minOverlap']
    xyzDim_minOverlap_sed = self.metaData['hash']['dimensions']['sed']['minOverlap']
    if xyzDim_minOverlap_gel == xyzDim_minOverlap_sed:
      xyDim_minOverlap = xyzDim_minOverlap_gel[:2]
      zDim_minOverlap = xyzDim_minOverlap_gel[2]
    else:
      print("Material dependent xyz overlaps for gel vs sed are not supported yet (but xy may be different from z)")
      raise KeyError
    xyChunks = math.ceil((xyDim_full[0] - xyDim_minOverlap[0])/(xyDim_crop[0]-xyDim_minOverlap[0]))
    # z chunks are computed separately for gel and sediment and based on min overlap and z dimension in gel and sed portions
    zDim_full = self.metaData['imageParam']['zDim']
    zDim_gel = self.metaData['imageParam']['gelSedimentLocation']
    zDim_sed = zDim_full - zDim_gel
    #zDimGel_crop = self.metaData['deconvolution']['deconCropZDimGel']
    zDimGel_crop = self.metaData['hash']['dimensions']['gel']['xyz'][2]
    zDimSed_crop = self.metaData['hash']['dimensions']['sed']['xyz'][2]
    #zDimSed_crop = self.metaData['deconvolution']['deconCropZDimSediment']
    #zDim_minOverlap = self.metaData['deconvolution']['minZOverlap']
    zChunks_sed = math.ceil((zDim_sed - zDim_minOverlap)/(zDimSed_crop - zDim_minOverlap))
    zChunks_gel = math.ceil((zDim_gel - zDim_minOverlap)/(zDimGel_crop - zDim_minOverlap))
    # create an empty dictionary with keys spanning the hash size
    hashSize = timeSteps*(zChunks_sed + zChunks_gel)*xyChunks*xyChunks
    self.metaData['hashDimensions'] = {'hashSize':hashSize,\
                                       'xyChunks':xyChunks,\
                                       'zChunks_sed':zChunks_sed,\
                                       'zChunks_gel':zChunks_gel,\
                                       'timeSteps':timeSteps}
    #print(hashSize, timeSteps, zChunks_sed, zChunks_gel,xyChunks)
    # Generate, for each hashValue, the xyz cutoffs...permutations on 3 choices, each, on x and y
    # to generate x points, we start at left (ie 0), \
    # add the cropped dimension until we exceed the full dimension, \
    # and then shift the rightmost point to end at rightmost extent. Same for y
    xCrop = []
    for n in range(xyChunks):
      if n==0: leftPt=0
      elif n>0 and n<(len(range(xyChunks))-1): leftPt=n*xyDim_crop[0] - xyDim_minOverlap[0]
      else: leftPt = xyDim_full[0] - xyDim_crop[0]
      xCrop.append((leftPt,leftPt+xyDim_crop[0]))
    zCrop = [] # list of all the z-positions for gel and sediment
    material = [] # list of str specifiying whether material is sed or gel
    for n in range(zChunks_gel):
      if n==0: bottomPt=0
      else: bottomPt=n*(zDimGel_crop - zDim_minOverlap)
      zCrop.append((bottomPt,bottomPt+zDimGel_crop))
      material.append('gel')
    for n in range(zChunks_sed):
      if n==0: topPt = zDim_full
      else: topPt = zDim_full - n*(zDimSed_crop - zDim_minOverlap)
      zCrop.append((topPt-zDimSed_crop,topPt))
      material.append('sed')
    hashTable={}
    hashInverse = {}
    # Do we need a hashInverse table? Yes because the job flow requires numbers to be passed to the
    # job queing. We could in principle make one hash with
    # key:values like '9,3,16':(9,3,16)
    # as opposed to:
    #      hash: '68':(9,3,16) and
    #      inverseHash: '9,3,16':68
    # but we cannot pass '9,3,16' to the job queuing as a hashValue
    # additionally, converting '68' to 68 is simplest possible type conversion \
    # that will most probably work as the default
    # on any level of the job control whether in python, bash or whatever the SLURM is written in.
    # This type conversion is also handled in this class by using the method queryHash
    keyCount = 0
    for t in range(timeSteps):
      for z in range(len(zCrop)):
        for y in range(len(xCrop)):
          for x in range(len(xCrop)):
            hashTable[str(keyCount)] = dict([('xyztCropTuples',[xCrop[x],xCrop[y],zCrop[z],t])])
            hashTable[str(keyCount)]['material'] = material[z]
            hashTable[str(keyCount)]['index'] = (x,y,z,t)
            hashInverse[str(x)+ ',' + str(y)+ ',' +str(z)+ ','+ str(t)]=keyCount
            # are the keys unique? Yes but commas are required to avoid '1110' <- 1,11,0 pr 11,1,0 or 1,1,10
            keyCount += 1
    """
    I have to check that 'the bounds are working the way I think they are. \
    In particular the overlaps need to added/subtracted somewhere and this, \
    I think, depends on the direction and start point from which I am counting.
    To generate z, we split case into sediment and gel, propagating from gel bottom \
    up until we past gel/sediment and propagate down from TEM grid until we get past gel/sediment.\
    No shifting at the end necessary since this is just going to influence \
    how much gel/sediment overlap is in each chunk. \
    We should make these into functions along with wrapping the dimension counting into functions as well.\
    Additionally, there is likely a better yaml implementation to put dimensions used into \
    an easier to use data structure. \
    Can we view line comments in yaml after importing? Definately yes if you make the comments \
    part of the data structure. Then everything is a value and text string...
    I feel like the rest of this is a nested for loop over 
    indices:  time, no. of gel and  no of sed chunks, xy chunks, 
    """
    self.hash = hashTable
    self.invHash = hashInverse

  def queryHash(self, hashValue):
    """ query the dplHash table with given hashValue and return a dictionary
        containing all the query-able values: crop params,
        Note this this function makes sure that you always pass the correct type to the hashTable
        by forcing a conversion to string
    """
    return self.hash[str(hashValue)]

  def index2key(self,index):
    """ convert tuple to list of strings and then concatenate the list with comma seperators. """
    if type(index)!= list and type(index) != tuple: raise TypeError
    elif len(index) != 4: print("Querying inverse hash with tuple with len != 4. Problem?")
    return ','.join([str(elt) for elt in index])

  def queryInvHash(self,index):
    """
    query the inverse hash table with xyzt index of 4 tuple
    :return: hashValue
    """
    return self.invHash[self.index2key(index)]

  def time2HVList(self,time,material):
    """
    returns a list of all hashValues for a specified time and material
    """
    out = []
    for hv in [int(elt) for elt in self.hash.keys()]:
      if self.queryHash(hv)['xyztCropTuples'][3] == time:
        if material not in ['sed','gel', 'all']:
          print("time2HVList called with unsupported material type: ", material)
          raise KeyError
        elif material == 'all': out.append(hv)
        elif self.queryHash(hv)['material'] == material:
          out.append(hv)
        else: pass
      else: pass
    return out

  def getNNBHashValues(self,hashValue):
    """
    get hashValues of all nearest neighbor cells of the same material to input hashValue

    basic approach is to get all possible nnb indices and then remove any that:
      - are not of the same material
      - are no present
    :param hashValue:
    :return:
    """
    hashNNB = {} # this is a dict with two keys 'sed' and 'gel' depending on whether the nnb hashValue material type
    hashNNB['sed'] = []
    hashNNB['gel'] = []
    # look up the associated xyzt index value by querying hash table
    index = self.queryHash(hashValue)['index']
    print("index is: ", index)
    # generate all possible nnb index keys adding/subtracting to the index values
    indexList = []
    for dz in [-1,0,1]:
      for dy in [-1,0,1]:
        for dx in [-1,0,1]:
          if dz == 0 and dy == 0 and dx == 0: pass
          else:
            indexList.append((index[0] + dx,index[1] + dy, index[2] + dz, index[3]))
            #print(dx,dy,dz)
    # attempt to query inverse hash with candidate keys and if not present, just move on or maybe note boundary case
    for index in indexList:
      try:
        nnb = self.queryInvHash(index)
        mat = self.sedOrGel(nnb)
        hashNNB[mat].append(nnb)
      except KeyError:
        print(index)
        pass
    # return a list of hashValues.
    return hashNNB

  def getOverlap(self, hv1,hv2):
      """
      get the , possibly null, spatial overlap region between a pair of hashValues (hv1, hv2)
      this function is material agnostic. It will report the overlap for different material flags
      :param (hv1,hv2): pair of hashValues
      :return: False if no overlap, otherwise the overlap region following the cropIndex format.
      """
      def segOverlap(seg1,seg2):
        """ returns overlap between seg1 and seg2 or returns False"""
        # decide which segment has a larger extent and label it "top"
        if seg1[1]> seg2[1]:
          top = seg1
          bottom = seg2
        else:
          top = seg2
          bottom = seg1
        if top[0] > bottom[1]: return False
        else: return (top[0],bottom[1])

      # query the hashtable and get the crop parameters for each hashvalue
      crop1, crop2 = self.getCropIndex(hv1),self.getCropIndex(hv2)
      output = []
      for i in range(3):
        overlap = segOverlap(crop1[i],crop2[i])
        if overlap != False: output.append(overlap)
        else: pass
      if len(output) == 3: return output
      else: return False

  def sedOrGel(self,hashValue):
    """ returns either 'sed' or 'gel' as a string specifying whether the chunk corresopondnign to hashvalue is a
        colloid sediment or a gel
    """
    return self.queryHash(hashValue)['material']

  def sedGelInterface(self,hashValue):
    """
       returns tuple of (material type, interfaceBool, cushionSlices )

       interfaceBool is True if the chunk contains the interface
       cushionSlices is an int giving the maximum number of slices to crop before interface is cropped out
          - from the interface to the stack bottom for sediment chunks where interfaceBool is True
          - from interface to stack top for gel whre interfaceBool is True
          - some nengative number is interfaceBool is False.
    """
    # get the z-range of the specified hashValue
    zRange = self.getCropIndex(hashValue)[2]
    interface = self.metaData['imageParam']['gelSedimentLocation']
    # is self.metaData['imageParam']['gelSedimentLocation'] in the range?
    mat = self.sedOrGel(hashValue)
    if mat == 'sed': cushion = interface - zRange[0]
    elif mat == 'gel': cushion = zRange[1] - interface
    if interface < zRange[1] and interface > zRange[0]:
      return (self.sedOrGel(hashValue),True, cushion)
    else:
      return (self.sedOrGel(hashValue),False, cushion)

  def getTimeStep(self, hashValue):
    """
       query the dplHash table with given hashValue and returns the timeStep
       corresponding to the input hashValue.
    """
    return self.queryHash(hashValue)['xyztCropTuples'][3]

  def getCropIndex(self,hashValue):
    """
        query dplHash table with given hashValue and return the cropIndex
        cropIndex:intArray ((xmin,xmax),(ymin,max),(zmin,zmax))
    """
    return self.queryHash(hashValue)['xyztCropTuples'][0:3]

  def stepBool(self,kwrd):
    """
    Simple function that returns True or False if kwrd is included in pipeline step
    :param kwrd:
    :return:
    """
    pipelineList = self.metaData['pipeline']
    pairList = [(list(elt.keys())[0],list(elt.values())[0]) for elt in pipelineList]
    outBool = False
    for pair in pairList:
      # delete all the false entries
      if pair[0] == kwrd: outBool = pair[1]
    return outBool

  def getPipelineStep(self,kwrd,stream='up',out='kwrd'):
    """
    simple function that will return a keyword specifying the upsteam or down stream analysis relative to the step
    specified with kwrd.
    Note this is read off the pipeline list of dict entries in yaml and will naturally exclude any steps that are skipped
    Additionally, no keyword is hardcoded into the function itself; all the keywords and pipeline steps are read from the yaml file. (?)
    :param hashValue:
    :param kwrd: what step in the pipeline is queried?
    :param stream: do you want the step up or down relative to the kwrd step? Note this will automatically exclude any boolean flags that are False.
    :return: another keyword specifying the step
    """
    # figure out what exactly is the data structure contained in the yaml file...its a list of dictionary key? Really?
    pipelineList = self.metaData['pipeline']
    pairList = [(list(elt.keys())[0],list(elt.values())[0]) for elt in pipelineList]
    for pair in pairList:
      # delete all the false entries
      if pair[1] == False:
        pairList.remove(pair)
      else: pass
    keyList = [elt[0] for elt in pairList]
    # find where the kwrd is in the list
    # find nearest up- and down- stream keywords that is True.
    outputKey = 'initial'
    for i in range(len(keyList)):
      if keyList[i] == kwrd:
        if stream=='up':
          try: outputKey = keyList[i-1]
          except IndexError:
            print("getPipelineStep requested something upstream of the start of the pipeline?")
            raise IndexError
        elif stream == 'down':
          try: outputKey = keyList[i+1]
          except IndexError:
            print("getPipelineStep requested something downstream of the end of the pipeline?")
            raise IndexError
      else: i+=1
    # return the required keyword
    if outputKey == 'initial':
      print("getPipelineStep failed due to invalid input keyword: ",kwrd)
      raise KeyError
    else: return outputKey

  def writeLog(self,hashValue,pipelineStep,*returnIndexDict, computer='ODSY'):
    """
    This function creates and updates a file that records any changes to the reference position and
    stack dimensions of a given hashValue chunk as it passes through the image analysis pipeline.
    It should be called whenever a script is called.
    file formatting: yaml (tentatively)
    possible pipelineStep values:
    rawTiff, hash, preprocessing,flatField, decon, smartCrop, postDecon, locations, tracking

    <processing step name>:
      cropBool: boolean flag on whether this step introduced cropping/hashing
      origin: [vector pointing from the origin in the outout of the step realtive to the origin of the previous step]
      dimensions: [dimensions of the cropped image/array]
      time: # time this step was completed

    :param hashValue:
    :return:
    """
    # open the file and if not present create it in a permanent (not scratch) directory.
    logFilePath = self.getPath2File(hashValue,kwrd='log',computer=computer)
    f = open(logFilePath,'a')
    # append the file with the following yaml text with maybe an additional case for the initalization
    if pipelineStep == 'hash':
      # from the metaData hashTable, get the origin, dimensions and for good measure some other details like ref pos etc
      metaData = self.queryHash(hashValue)
      origin = [metaData['xyztCropTuples'][n][0] for n in range(3)]
      dim = [metaData['xyztCropTuples'][n][1] - metaData['xyztCropTuples'][n][0] for n in range(3)]
      index = metaData['index']
      log  = "hashValue: " + str(hashValue)+"\n"
      log += "metaDataYAML: " + self.getPath2File(hashValue,kwrd='metaDataYAML',computer=computer) +"\n"
      log += "hash: \n"
      log += "  origin: [" + self.index2key(origin) + "]\n"
      log += "  dim: [" + self.index2key(dim) + "]\n"
      log += "  index: [" + self.index2key(index) + "]\n"
      log += "  material: " + self.sedOrGel(hashValue) +"\n"
      f.write(log)
    elif pipelineStep == 'rawTiff': print("called writelog for rawTiff step, but there is nothing to record")
    else: # we are not initializing now
      log = pipelineStep + ":\n"
      log += "  crop: " + str(self.metaData[pipelineStep]['crop']) + " # Any cropping on this step?\n"
      log += "  time: " + str(datetime.now()) + "# current time, YYYY-MM-DD HH:MM:SS \n"
      # test whether there was cropping
      if self.metaData[pipelineStep]['crop'] == False:
        # This should be true for preprocessing, flatField, decon, postDecon, locations, and tracking
        log += "  origin: [0,0,0] # vector pointing from origin of previous step to output of this step\n"
        log += "  dim: [0,0,0] # change in the xyz dimensions from previous step\n"
      else:
        # This is either hashParam or a function that is going to crop (ie smartCrop and maybe something else in the future)
        if pipelineStep == 'smartCrop':
        # format the return data of the function
          for key in returnIndexDict[0].keys(): # dont know why returnIndexDict is a tuple...
            log += "  " + key + ": [" + self.index2key(returnIndexDict[0][key])+"]\n"
        else: print("There is some missing or extra key that you are trying to log..")
      f.write(log)
    f.close()

  def loadStack_imageJ(self,hashValue, imgType='sequence',computer='ODSY'):
    """
       output the lines required to open the stack in imageJ
       two cases: imageStack and psf. These should be handled differently and are labeld by str variable 'imgType'
    """
    cropTuples=self.hash[str(hashValue)]['xyztCropTuples']
    if imgType=='sequence':
      #path = '/path/to/folder/containing/image/sequence/to/be/read/from/self.metaData/note/no/trailing/backslash'
      path = self.getPath2File(hashValue,computer=computer,kwrd='rawTiff',pathOnlyBool=True)
      fileSelection = 't'+ '%04d'%cropTuples[-1]  # typcially a string corresponding to 't' + zero padded integer for time point
      start, number = str(cropTuples[2][0]),str(cropTuples[2][1]- cropTuples[2][0]) # begin at 'start' and load 'number' of images. Note type conversion to str
      text = 'run("Image Sequence...", "open='+path+' number='+number+' starting='+start+' file='+fileSelection+' sort use");\n'
    elif imgType=='stack':
      path = self.getPath2File(hashValue,kwrd='rawTiff',computer=computer)
      text = 'open("'+path+'");\n'
    else:
      print('variable imgType must be either "sequence" or "stack" when calling loadStack_imageJ')
      raise TypeError
    return text

  def getPath2File(self,hashValue, kwrd ='rawTiff' ,computer='ODSY',extension='default',pathOnlyBool=False,fileNameOnlyBool=False):
    """
    kwrd options
      - particleLocating
      - metaDataYAML
      - hash (?)
      - log
      - darkTiff
      - raw: scratchdir/prefix
      - psf
      - preprocessingOut
      - deconOut
      - postDeconOut
      - flatField

      fileName = [path or scratch] + [globalPrefix] + [type specific prefix] + [file extension]
      example prior to creating this function:
               if computer=='ODSY': outPath = self.metaData['filePaths']['preprocessOutPath_ODSY']
               fName = self.metaData['filePaths']['fName_prefix']+str(hashValue).zfill(5)+'.tif'

      return: full path to file as a string
    """
    # Given kwrd, decide on scratch or project
    scratchSubDirList = self.metaData['filePaths']['scratchSubDirList']
    projectSubDirList = self.metaData['filePaths']['projectSubDirList']
    gitSubDirList = self.metaData['filePaths']['gitDirList']
    if kwrd in scratchSubDirList:
      path = self.metaData['filePaths']['scratchDirectory_'+str(computer)]
      path += '/' + kwrd # immediately form the directory
    elif kwrd in projectSubDirList:
      path = self.metaData['filePaths']['projectDirectory_'+str(computer)]
      path += '/' + kwrd # immediately form the directory
    elif kwrd in gitSubDirList:
      #print('tractionRheoscopyGit_'+str(computer))
      path = self.metaData['filePaths']['tractionRheoscopyGit_'+str(computer)]
      if kwrd == 'kilfoil_matlab': path += '/particleLocating'
      path += '/' + kwrd # immediately form the directory
    else:
      print("possible kwrd in getPath2File must be listed in subDirectories in \
            either project scratch or git in YAML")
      print("or you have to code in some exception")
      raise KeyError
    # given kwrd, decide on file extension (ie tif or text or yaml)
    if extension == 'default': fileExt = self.metaData['fileExtensions'][str(kwrd)]
    else: fileExt=extension

    # now build the fileName keeping in mind that I will need exceptions for some keywords
    if kwrd in self.metaData['fileNamePrefix'] and kwrd != 'rawTiff' and kwrd != 'darkTiff' and kwrd != 'metaDataYAML':
      fName = self.metaData['fileNamePrefix']['global']
      fName += self.metaData['fileNamePrefix'][str(kwrd)]
      fName += 'hv'+str(hashValue).zfill(5)+fileExt
    elif kwrd == 'rawTiff':
      # choose the right time slice to open the rawTiff file given the hashValue.
      timeStep = self.getTimeStep(hashValue)
      # assemble the right fileName for the given timeStep
      fName = self.metaData['fileNamePrefix']['rawTiff'] + 't' + str(timeStep).zfill(4) + fileExt
    elif kwrd =='darkTiff':
      fName = self.metaData['fileNamePrefix'][str(kwrd)]+fileExt
    elif kwrd == 'metaDataYAML':
      fName = self.metaData['fileNamePrefix']['global']
      fName += self.metaData['fileNamePrefix'][str(kwrd)]
      fName += fileExt

    elif (kwrd in scratchSubDirList or kwrd in projectSubDirList or kwrd in gitSubDirList) and kwrd not in self.metaData['fileNamePrefix']:
      # I think this case should only occur in cases where I really want to return a path to a directory and not
      # a path to a specific file. For the kwrds currnently listed this includes:
      # - psfPath
      # - fullStackPath
      # -calibrationPath
      fName=''
    else:
      print("possible kwrd in getPath2File must be listed in subDirectories in either project or scratch in YAML")
      raise KeyError
    if pathOnlyBool == False and fileNameOnlyBool==False: return path + '/' + fName
    elif pathOnlyBool == True and fileNameOnlyBool==False: return path
    elif pathOnlyBool == False and fileNameOnlyBool==True: return fName
    else:
      print("getPath2File was called with an invalid combination of pathOnly and fileNameOnly boolean flags")
      raise ValueError

  def loadStack_py(self,hashValue,kwrd,imgType='stack',computer='ODSY'):
    """
      output the lines require to load a stack with a specific hashValue in python
      using scikit. This is mostly selecting the right timestep and also detemrining which
      part of the processing pipeline we need to open
        -anything, just specify full path to tiff stack
        -raw
        -preprocessing
        -flatfield
        -postDecon
    :param hashValue:
    :param imgType:
    :param computer:
    :return:
    """
    # now for the right fullpath to the tiff stack given the timeStep specified by hashValue
    if imgType == 'stack':
      path = self.getPath2File(hashValue,kwrd=kwrd, computer=computer)
      with ski.external.tifffile.TiffFile(path) as tif:
        data = tif.asarray()
      return data
    else:
      print("loadStack_py does not currently support loading something other than a stack")
      raise KeyError

  def xyCrop_imageJ(self, hashValue):
    """ output lines to crop image corresponding to hashValue integer provided """
    cropTuples=self.hash[str(hashValue)]['xyztCropTuples']
    x0,y0 = cropTuples[0][0],cropTuples[1][0]
    w,h = cropTuples[0][1]-cropTuples[0][0],cropTuples[1][1]-cropTuples[1][0]
    text = 'makeRectangle('+str(x0)+', '+str(y0)+', '+str(w)+', '+str(h)+');\n'
    text += 'run("Crop");\n'
    text += 'cropTitle=getTitle();\n'
    return text

  def bkSubtract_imageJ(self, bitOutput=16):
    """ output the lines used for ImageJ background subtraction"""
    sigma = self.metaData['preprocessing']['params']['width']
    text  = 'run("Duplicate...", "duplicate");\n'
    text += 'run("Gaussian Blur...", "sigma='+str(sigma)+' stack");\n'
    text += 'blurTitle = getTitle();\n'
    text += 'imageCalculator("Subtract create 32-bit stack", cropTitle,blurTitle);\n'
    text += 'bkSubtractTitle = getTitle();\n'
    if bitOutput==16:
      text +='selectWindow(bkSubtractTitle);\n'
      text += 'run("16-bit");\n' 
      text += 'run("Enhance Contrast","saturated=0.1");\n'
    elif bitOutput==8:
      text += 'selectWindow(bkSubtractTitle);\n'
      text += 'run("8-bit");\n'
      text += 'run("Enhance Contrast","saturated=0.1");\n'
    else: 
      print('variable "bitOutput" in bkSubtract_imageJ() must type int and equal to 8 or 16') 
      raise TypeError
    return text 

  def savePreprocessing_imageJ(self,hashValue,computer='ODSY'):
    """ this function saves the preprocessing image to a tiff file with name given by hashvalue
        and path specfied in metaData
    """
    fName = self.getPath2File(hashValue,kwrd='preprocessing',computer=computer)
    text = 'saveAs("Tiff", "'+ fName +'");\n'
    text += 'run("Close All");\n'
    return text

  def decon_DL2_imageJ(self,hashValue,computer='ODSY'):
    """ Run a headless deonconvolution run as an imageJ macro
       
       image = "-image file /Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/preprocessingOut/deconScriptHashing_00058.tif"
       psf = " -psf file /Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/psf_stacks/psf_hashIndex_parameters/psf_730_730_167_z125_25.tif"
       algorithm = " -algorithm RLTV 40 0.1000"
       parameters = ""
       parameters += "-path /Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/deconOut"
       run("DeconvolutionLab2 Run", image + psf + algorithm + parameters)
    """
    # choose the right bksubtracted input image for the input hashValue
    if self.metaData['preprocessing']['bkSubtractBool']==True:
      inputPath = self.metaData['filePaths']['preprocessOutPath_'+computer]
    else: inputPath=self.metaData['filePaths']['rawTiff_'+computer]
    ##
    outputText = 'image = " -image file '+inputPath+'/'
    outputText += self.metaData['filePaths']['fName_prefix']+str(hashValue).zfill(5)+'.tif"\n' # This should be added to hash object and then resaved to yaml file? 
    ##
    ## choose the right height dependent psf for the input hashValue. The type (sediment or gel) also comes into play.
    ##
    materialStr = self.hash[str(hashValue)]['material']
    # compute absolute height (in um) above the coverslip for the hashValue
    zCropTuple_px = self.hash[str(hashValue)]['xyztCropTuples'][2]  
    zOffset_um = self.metaData['imageParam']['piezoPos']['imageStackBottom']-self.metaData['imageParam']['piezoPos']['coverslip']
    zHeight_um = self.metaData['imageParam']['px2Micron'][2]*zCropTuple_px[0] + zOffset_um # int specifying the real height of the bottom of the chunk relative to the coverslip...aboslute position in um
    # parse the psf filenames to extract material type and absolute z-position abvove coverslip. Filename tags should be typeSED or typeGEL, and z*
    psfPath = self.metaData['filePaths']['psfPath_'+computer]
    psfList = glob.glob(psfPath+'/'+'*type'+materialStr.upper()+'*.tif') # returns a list of strings of full paths of all *.tif files in psf directory that contain typeSED or typeGEL in file name
    zHeightList=[] # list of ints containing heights
    for fPath_str in psfList:
      # get a list of ints containing zHeights
      indexStart = fPath_str.find('absZ')+4 # add the length of the search string 
      indexEnd = fPath_str.find('.tif')
      #print(fPath_str[indexStart:indexEnd])
      zHeightList.append((int(fPath_str[indexStart:indexEnd]),fPath_str))
    zHeightList.sort()
    psfIndex = 0
    for n in range(len(zHeightList)):
      height = zHeightList[n][0]
      if height<zHeight_um: psfIndex +=1
    psfFilePath_str = zHeightList[psfIndex][1]
    ##
    outputText += 'psf = " -psf file '+psfFilePath_str+'"\n'
    ##
    ## set algorithm and parameters    
    ##
    deconMethod_str = self.metaData['decon']['method']
    if deconMethod_str == 'RLTV':
      regParam,n_iteration = self.metaData['decon']['lambda'], self.metaData['decon']['iterations']
    else:
      print("You are using a decon method that is not programmed in yet. Shouldnt be hard to do though!")
      raise TypeError
    outputText += 'algorithm = " -algorithm '+deconMethod_str+' '+str(n_iteration)+' '+str(regParam)+'"\n'    
    outputText += 'parameters = " -path '+self.metaData['filePaths']['deconOutPath_'+computer]
    outputText += ' -out stack '+self.metaData['filePaths']['fName_prefix']+'decon_hv'+str(hashValue).zfill(5)\
                  + ' -monitor no -stats save"\n'
    outputText += 'run("DeconvolutionLab2 Run", image + psf + algorithm + parameters);\n'
    """
    convert output to 8 bit and get rid of the background and garbage slices and save output
    
    The following code is flawed for subtle implementations reasons. \
    Fiji proceeds to the next line of a macro prior to the previous command finishing
    For example, it will attempt to assign the title *prior* to deconvoltuion finishing. \
    There is no way around this without using a full scripting language. \
    At this point, it might be better to just save the 32 bit output, not additional scaling, \
    and complete the image analysis in matlab during particle locating.\ 
    
    outputText += 'deconResultTitle = getTitle();\n' # select the 32 bit output \
                                                     # convert to 8 bit, \
                                                     set the lookup table but dont alter image, \
                                                     and save
    outputText += 'selectWindow(deconResultTitle);\n'
    outputText += 'run("8-bit");\n'
    fName = self.metaData['filePaths']['fName_prefix']+'decon_hv'+str(hashValue).zfill(5)+'.tif'
    deconOutPath = self.metaData['filePaths']['deconOutPath_'+computer]
    outputText += 'saveAs("Tiff", "'+deconOutPath+'/'+fName+'");\n'
    outputText += 'run("Close All");\n'
    """
    return outputText

  def makeDeconDL2_javaCommandLine(self,hashValue,computer='ODSY'):
    """ Run a headless deonconvolution run from the command line using java and deconvolution lab 2
       
       java -jar DeconvolutionLab_2.jar Run 
      -image synthetic Cube 10.0 1.0 size 200 100 100 
      -psf synthetic Double-Helix 3.0 30.0 10.0 size 200 100 100 intensity 255.0 
      -algorithm RIF 0.1000 -out mip MI1 -path home
    """
    if computer == 'ODSY': outputText = self.metaData['filePaths']['javaPath_'+computer] + '\n'
    else: outputText = ""
    outputText += 'java '
    outputText += ' -Xms1024m -Xmx16g -jar '+self.metaData['filePaths']['DL2Path_'+computer]+'/'+ 'DeconvolutionLab_2.jar Run '
    # decide on where the input file is to be read from given the pipeLineBool section in metData
    # choose the right bksubtracted input image for the input hashValue
    if self.stepBool('decon')==False: return print("YAML data says deconBool == False")
    inputKwrd = self.getPipelineStep('decon',stream='up')
    inputPath = self.getPath2File(hashValue,kwrd = inputKwrd, computer=computer,pathOnlyBool=True)
    outputText += ' -image file '+inputPath+'/'
    #outputText += self.metaData['filePaths']['fName_prefix']+str(hashValue).zfill(5)+'.tif' # This should be added to hash object and then resaved to yaml file?
    inputKey = self.getPipelineStep('decon',stream='up')
    outputText += self.getPath2File(hashValue,kwrd=inputKey,computer=computer,fileNameOnlyBool=True)
    ##
    ## choose the right height dependent psf for the input hashValue. The type (sediment or gel) also comes into play.
    ##
    materialStr = self.hash[str(hashValue)]['material']
    # compute absolute height (in um) above the coverslip for the hashValue
    zCropTuple_px = self.hash[str(hashValue)]['xyztCropTuples'][2]  
    zOffset_um = self.metaData['imageParam']['piezoPos']['imageStackBottom']-self.metaData['imageParam']['piezoPos']['coverslip']
    zHeight_um = self.metaData['imageParam']['px2Micron'][2]*zCropTuple_px[0] + zOffset_um # int specifying the real height of the bottom of the chunk relative to the coverslip...aboslute position in um
    # parse the psf filenames to extract material type and absolute z-position abvove coverslip. Filename tags should be typeSED or typeGEL, and z*
    #psfPath = self.metaData['filePaths']['psfPath_'+computer]
    psfPath = self.getPath2File(hashValue,kwrd='psfPath',computer=computer,pathOnlyBool=True)
    print(psfPath)
    psfList = glob.glob(psfPath+'/'+'*type'+materialStr.upper()+'*.tif') # returns a list of strings of full paths of all *.tif files in psf directory that contain typeSED or typeGEL in file name
    zHeightList=[] # list of ints containing heights
    for fPath_str in psfList:
      # get a list of ints containing zHeights
      indexStart = fPath_str.find('absZ')+4 # add the length of the search string 
      indexEnd = fPath_str.find('.tif')
      #print(fPath_str[indexStart:indexEnd])
      zHeightList.append((int(fPath_str[indexStart:indexEnd]),fPath_str))
    zHeightList.sort()
    psfIndex = 0
    for n in range(len(zHeightList)):
      height = zHeightList[n][0]
      #print(height,zHeight_um)
      if height<zHeight_um: psfIndex +=1
    psfFilePath_str = zHeightList[psfIndex][1]
    ##
    outputText += ' -psf file '+psfFilePath_str
    ##
    ## set algorithm and parameters    
    ##
    deconMethod_str = self.metaData['decon']['method']
    if deconMethod_str == 'RLTV':
      regParam,n_iteration = self.metaData['decon']['lambda'], self.metaData['decon']['iterations']
    else:
      print("You are using a decon method that is not programmed in yet. Shouldnt be hard to do though!")
      raise TypeError
    outputText += ' -algorithm '+deconMethod_str+' '+str(n_iteration)+' '+str(regParam)    
    #outputText += ' -path '+self.metaData['filePaths']['deconOutPath_'+computer]
    outputText += ' -path '+ self.getPath2File(hashValue,kwrd='decon',computer=computer,pathOnlyBool=True)
    #outputText += ' -out stack noshow ' + self.metaData['filePaths']['fName_prefix']+'decon_hv'+str(hashValue).zfill(5)
    outputText += ' -out stack noshow ' + self.getPath2File(hashValue,kwrd='decon',computer=computer,fileNameOnlyBool=True)
    outputText = outputText[0:-4] # special case, clip the .tif extension since DL2 adds an extension automatically
    outputText += ' -monitor no'
    if self.metaData['decon']['apodization']['bool']==True:
      apoDict=self.metaData['decon']['apodization']
      lateral,axial = apoDict['lateral'],apoDict['axial']
      outputText += ' -apo ' + lateral + ' ' + axial 
    if self.metaData['decon']['padding']['bool']==True:
      paddingDict = self.metaData['decon']['padding']
      lateral,axial = paddingDict['lateral'],paddingDict['axial']
      outputText += ' -pad ' + lateral + ' ' + axial 
      #NO TUKEY'# No lateral windowing, Tukey aka truncated cosine window on axial direction. This should be implemented on the YAML file as well
    #output_fName = self.metaData['filePaths']['deconOutPath_'+computer]+'/'
    #output_fName += self.metaData['fileNames']['global_prefix']+self.metaData['fileNames']['decon_prefix']+'hv'+str(hashValue).zfill(5)+'.x'
    output_fName = self.getPath2File(hashValue,kwrd='dplPath', computer=computer,extension='_decon.x')
    with open(output_fName,'w') as f: f.write(outputText)
    os.chmod(output_fName,509) # google "chmod +x on python". No idea wtf octal literal is and what 509 means here
    return "executable decon script for hashValue " + str(hashValue) +" created at :"+output_fName

  def makePreprocessing_imageJ(self,hashValue,computer='ODSY'):
    """
    based on the metaData including boolean flags for preprocessing etc create
    an imageJ script that will execute on command line and headless
    """
    if self.stepBool('preprocessing')==False: return print("YAML says there is no preprocessing step")
    output = self.loadStack_imageJ(hashValue,computer=computer)
    if self.metaData['preprocessing']['cropBool'] == True:
      output += self.xyCrop_imageJ(hashValue)
    if self.metaData['preprocessing']['bkSubtractBool'] == True:
      output += self.bkSubtract_imageJ()
    output += self.savePreprocessing_imageJ(hashValue,computer=computer)
    output_fName = self.getPath2File(hashValue,kwrd='dplPath', computer=computer,extension='_preprocessing.ijm')
    with open(output_fName,'w') as f: f.write(output)
    return 'ImageJ preprocessing macro created for hashValue ' + str(hashValue) +' created at :'+output_fName

  def makeFlatFielding_python(self,hashValue,computer='ODSY'):
    """
    Will create a python script to carry out flat field correction following
        https://en.wikipedia.org/wiki/Flat-field_correction
      and implemented in flatField.py
    The darkFrame is measured and is an input from the yaml metadata file,
      while the flatField image is approximated by longpass 2D gaussian filter
      carried out on each slice

    param hashValue: int
    param computer: str
    return: None, but will write a python script for the specified hashValue
    """
    # import flatField and other necessary modules
    flatFieldScript = 'import sys\n'
    gitDir = self.getPath2File(0,kwrd='particleLocating',computer = computer,pathOnlyBool = True,extension='')
    #flatFieldScript += 'sys.path.insert(0,\"' + str(self.metaData['filePaths']['particleLocatingSCRIPTS_'+computer]) +'\")\n'
    flatFieldScript += 'sys.path.insert(0,\"' + gitDir + '\")\n'
    flatFieldScript +='import flatField\n\n'
    flatFieldScript += 'import dplHash_v2 as dplHash\n'
    # open the raw file
    #rawPath = self.metaData['filePaths']['rawTiff_'+computer]
    # By using getPath2File and specifying the hashValue,  only the relevant timestep is loaded
    #rawPath = self.getPath2File(hashValue,kwrd='rawTiff',computer=computer)
    inputkwrd = self.getPipelineStep('flatField',stream='up')
    print(inputkwrd)
    rawPath = self.getPath2File(hashValue,kwrd = inputkwrd ,computer=computer)
    #darkPath = self.metaData['filePaths']['darkTiff_'+computer]
    darkPath = self.getPath2File(hashValue,kwrd='darkTiff',computer=computer)
    #outPath = self.metaData['filePaths']['preprocessOutPath_'+computer]
    outPath = self.getPath2File(hashValue,kwrd='flatField',computer=computer)
    flatFieldScript += 'rawStack = flatField.zStack2Mem(\''+str(rawPath)+'\')\n'
    # open the dark file and create master dark
    flatFieldScript += 'darkStack = flatField.zStack2Mem(\''+str(darkPath)+'\')\n'
    flatFieldScript += 'masterDark = flatField.zProject(darkStack)\n'
    # crop according to the yaml file and read off the crop parameters specific to hashValue from metaData
    if self.metaData['flatField']['crop2Hash'] == True:
      cropIndex = self.getCropIndex(hashValue)
      flatFieldScript += 'rawStack = flatField.cropStack(rawStack,'+ str(cropIndex) + ')\n'
      flatFieldScript += 'masterDark = flatField.cropStack(masterDark,'+ str(cropIndex[0:2]) + ')\n'
    flatFieldScript += 'flatStack = flatField.gaussBlurStack(rawStack,sigma='+str(self.metaData['flatField']['sigma'])+')\n'
    # carry out flatFielding
    flatFieldScript += 'corrStack = flatField.correctImageStack(rawStack,masterDark,flatStack)\n'
    # save corrected image to the right outPath...proably preprocessing subdirectory
    flatFieldScript += 'flatField.array2tif(corrStack,\''+str(outPath)+'\')\n'
    flatFieldScript += 'hashObject = dplHash.dplHash(\'' \
                       + self.getPath2File(hashValue,kwrd='metaDataYAML',computer=computer) \
                       + '\')\n'
    flatFieldScript += 'hashObject.writeLog(' + str(hashValue) + ', \'flatField\', computer=\''+str(computer) + '\')\n'
    #output_fName = self.metaData['filePaths']['flatField_'+computer]+'/'
    #output_fName += self.metaData['fileNames']['global_prefix']+self.metaData['fileNames']['flatField_prefix']+'hv'+str(hashValue).zfill(5)+'.py'
    output_fName = self.getPath2File(hashValue,kwrd='dplPath',extension='_flatField.py',computer=computer)
    with open(output_fName,'w') as f: f.write(flatFieldScript)
    return "Flatfielding python script for hashValue "+str(hashValue) +" created at: "+output_fName

  def makeParticleLocating_matlab(self,hashValue, computer='ODSY'):
    """ output the lines to run iteratuve particle location in matlab
        given the hashvalue provided. Check that the deconvolved file
        has been outputted successfully
        This is mother of a file. Its likely that the original file should be Stripped of
        All Unnecessary Bullshit (SAUBS) before this script is written. Note the output
        of this function will never:
          -crop
          -run anywhere except oddyssey
          -need any comments
    image_filename = 't0000_rb150px_gelStackZ113.tif';
    # also some addpath commands from the filePaths metaData section
    folder = '/n/regal/spaepen_lab/zsolt/DATA/tfrGel09102018b_shearRun09232018b/Colloid_z205_829_deconSNR12_SingleStacks/Colloid_z205_829_singleStacks';
    display(['Image file ' image_filename ' will be used.'])
    output_filename = strcat(image_filename(1:length(image_filename)-4),'_coordinates.xyz');
    lnoise = [0.9 0.9 0.7];
    lobject = [27 27 27];

    diameters = [35 35 35];
    mask_size = [20 20 20];
    min_separation = [19 19 19];
    masscut_initial = 7e5;
    masscut_residuals = 1e5;
    false_particle_size = [29 29 29];
    bridgeThreshold = 0.5;
    zeroPxThreshold =200;
    run('/n/regal/spaepen_lab/zsolt/SCRIPTS/locating_mat.m');
    # I dont think anything in particleLocating.m needs to be changed for a given hashVlaue
      although the function should call a version of particle locating that has:
      -bandpass filtering remove
      -iterative enabled
      -accepts 8 bit images
      -has the zeroPx Threshold and compression in bit range
      -maybe natively takes a 32 bit image, converts to 8 bit with some thresholding.
         I dont know the best thing to do with threholding at the moment
    """
    #locatingScriptDirectory = self.metaData['filePaths']['particleLocatingSCRIPTS_'+computer]
    locatingScriptDirectory = self.getPath2File(0,kwrd='kilfoil_matlab',computer = computer,pathOnlyBool = True,extension='')
    #inputFilePath = self.metaData['filePaths']['postDeconOutPath_'+computer]
    inputFilePath = self.getPath2File(hashValue,kwrd='postDecon',computer=computer,pathOnlyBool=True)
    #inputFilePath = self.getPath2File(hashValue,kwrd='postDecon',computer=computer,pathOnlyBool=True)
    # inputFileName_str = self.metaData['fileNames']['global_prefix'] \
    #                     + self.metaData['fileNames']['postDecon_prefix'] \
    #                     + 'hv'+str(hashValue).zfill(5)+'.tif'
    inputFileName_str = self.getPath2File(hashValue,kwrd='postDecon',computer=computer,fileNameOnlyBool=True)
    output  = "% Insert some comments here \n" # initialize the output string
    output += "image_filename = '"+inputFileName_str +"';\n"
    output += "folder = '" + inputFilePath+"'\n" # Careful that this isnt adding the path twice...ie the full path is not folder/inputfileName but rather just inputFileName
    output += "display(['Image file ' image_filename ' will be used.'])\n"
    #output += "output_filename = '" + self.metaData['filePaths']['locationsHashed_'+computer]\
            #+ '/'+self.metaData['fileNames']['global_prefix']\
            #+ self.metaData['fileNames']['location_prefix'] \
            #+ 'hv'+str(hashValue).zfill(5)+'_'\
            #+ self.sedOrGel(hashValue)+"_pxLocations.text';\n"

    pxLocationExtension = '_'+self.sedOrGel(hashValue)+"_pxLocations.text'"
    output += "output_filename = '"
    output += self.getPath2File(hashValue,kwrd='locations',extension=pxLocationExtension,computer=computer)
    output += ";\n"
    # separate the output streams into sed or gel distinctions.
    if self.sedOrGel(hashValue) == 'sed':
      for var, value in self.metaData['locating']['sedimentParam'].items():
        if type(value) == list: output += var +' = [' +' '.join([str(n) for n in value]) +'];\n' # List comprehension along with some formatting brackets
        elif (type(value) == int or type(value) == float): output += var+' = ' + str(value)+';\n'
        elif type(value) == str: pass # only string value should be prefix
        else:
          raise TypeError
    elif self.sedOrGel(hashValue) == 'gel':
      for var, value in self.metaData['locating']['gelParam'].items():
        if type(value) == list: output += var +' = [' +' '.join([str(n) for n in value]) +'];\n' # List comprehension along with some formatting brackets
        elif (type(value) == int or type(value)==float): output += var+' = ' + str(value)+';\n'
        elif type(value) == str: pass # only string value should be prefix
        else: raise TypeError
    #output_fName = self.metaData['filePaths']['deconOutPath_'+computer]+'/'+'deconScript_hv'+str(hashValue).zfill(5)+'.x'
    output += "addpath '"+locatingScriptDirectory+"'\n"
    output += "run('"+locatingScriptDirectory\
                     +"/" + "iterative_residual_locating_SAUBS.m')\n"
    output += "exit"
    #output_fName = self.metaData['filePaths']['locationsHashed_'+computer]+'/'
    #output_fName += self.metaData['fileNames']['global_prefix']+self.metaData['fileNames']['location_prefix']+'hv'+str(hashValue).zfill(5)+'.m'
    output_fName = self.getPath2File(hashValue,kwrd='dplPath',computer=computer,extension='_locating.m')
    # Note, that fileName and function names in MatLab can be no more than 63 characters.
    with open(output_fName,'w') as f: f.write(output)
    return "Particle location matlab *.m file for hashValue "+str(hashValue) +" created at: "+output_fName

  def makePostDecon_imageJ(self,hashValue,computer='ODSY'):
    """
    Make postprocessing imageJ script to convert 32 bit deconvolved chunk to 8 bit with thresholding for particle locating
    Looks something like this:

    open the rleevant tiff image (not a series since decon output is single stack.)
    setMinAndMax("28538.91", "89282.28");//These min and max should be read from yaml file assuming 32 bit image
    run("8-bit");
    'save and close all'
    """
    metaData = self.metaData['postDecon']
    # Load the deconvolved stack chunk for given hashValue
    inputKwrd = self.getPipelineStep('postDecon')
    inputPath = self.getPath2File(hashValue, kwrd = inputKwrd, computer=computer)
    outputScript = 'open("'+ inputPath +'");\n'

    # crop out garbage slices and border from deconvolved result
    # somehow update the reference pixel values in hash table or write a function and metaData parameter
    # to update the reference pixel positions to be used for stitching.
    if metaData['crop'] == True:
      outputScript += '// Crop the image according to parameters in yaml file \n'
      outputScript += '// to throw away perimeter artifacts due to deconvolution \n'
      # compute the crop dimensions keeping in mind that Fiji requires the corners of the cropped region as input
      # and the yaml file lists a width in pixel to throw away.
      imgDim = self.getCropIndex(hashValue) # triple of ints: ((xmin,xmax),(ymin,ymax), (zmin,zmax))
      postDeconImgDim = ((imgDim[0][0] + metaData['crop']['X'],imgDim[0][1] - metaData['crop']['X']),\
                         (imgDim[1][0] + metaData['crop']['Y'],imgDim[1][1] - metaData['crop']['Y']), \
                         (imgDim[2][0] + metaData['crop']['Z'],imgDim[2][1] - metaData['crop']['Z'])
                         )
      self.hash[str(hashValue)]['postDeconImgDecon'] = postDeconImgDim # I think this should work, however the syntax for adding a new dict maybve wrong.
      xyCropPt1 = (metaData['crop']['X'],metaData['crop']['Y'])
      xyCropWH = (imgDim[0][1] - imgDim[0][0] - 2*metaData['crop']['X'],imgDim[1][1] - imgDim[1][0] - 2*metaData['crop']['Y'])
      zSubstack = (metaData['crop']['Z'],imgDim[2][1] - imgDim[2][0]-metaData['crop']['Z'])

      # Add the information to the output script
      # makeRectangle(5, 5, 118, 118);
      outputScript += "makeRectangle(" + str(xyCropPt1[0]) + ", " + str(xyCropPt1[1]) + ", "
      outputScript += str(xyCropWH[0]) + ", " + str(xyCropWH[1]) + ");\n"
      # run("Crop");
      outputScript += 'run(\"Crop\");\n'
      # run("Make Substack...", "delete slices=10-180");
      outputScript += 'run(\"Make Substack...\",\"delete slices=' + str(zSubstack[0])+'-'+str(zSubstack[1]) +'\");\n'

    # Apply a threshold depending on information in the yaml file
    # Currently tow options are available:
    # -manual threshold parameters
    # -autothreshold based on MaxEntropy

    thresholdMeth = self.metaData['postDecon']['threshold']['method'][str(self.sedOrGel(hashValue))]
    if thresholdMeth == 'Manual':
      if self.sedOrGel(hashValue) == 'sed': thresholdMin,thresholdMax = self.metaData['postDecon']['threshold32Bit_SED']
      elif self.sedOrGel(hashValue) == 'gel': thresholdMin,thresholdMax = self.metaData['postDecon']['threshold32Bit_SED']
      # add the relevant commands to the script to carry out manual global thresholding
    elif thresholdMeth == 'MaxEnt':
      outputScript += 'setAutoThreshold(\"MaxEntropy dark no-reset stack\");\n'
      outputScript += 'run(\"Threshold...\");\n'
      outputScript += 'run(\"NaN Background\",\"stack\");\n'
      outputScript += 'run(\"Enhance Contrast\", \"saturated=0.0\");\n'
    elif thresholdMeth == 'Default':
      outputScript += 'setAutoThreshold(\"Deafult dark no-reset stack\");\n'
      outputScript += 'run(\"Threshold...\");\n'
      outputScript += 'run(\"NaN Background\",\"stack\");\n'
      outputScript += 'run(\"Enhance Contrast\", \"saturated=0.0\");\n'


    outputScript += 'run("8-bit");\n' # Convert to 8bit
    mosaicYAML = self.metaData['postDecon']['mosaic']
    if mosaicYAML['bool'] == True:
       """
       run("Gausian Curvature 3D", "number_of_iterations=50");
       run("Curvature Filters 2D", "filter=[MC (Mean Curvature)] method=[No split] number=1");
       """
       if mosaicYAML['gaussianCurvature']['bool'] == True:
         gc = mosaicYAML['gaussianCurvature']
         outputScript += 'run(\"' + gc['filter_name'] + '\",'
         outputScript += '\"number_of_iterations=' + str(gc['iter'])+'\");\n'
       if mosaicYAML['meanCurvature2D']['bool'] == True:
         mc = mosaicYAML['meanCurvature2D']
         outputScript += 'run(\"Curvature Filters 2D\", \"filter=' + mc['filter_name'] +' method='
         outputScript += mc['method'] +' number=' + str(mc['iter']) +'\");\n'
    # save 8 bit tiff result to path scpecified in yaml file.
    #fName = self.metaData['fileNames']['global_prefix']+self.metaData['fileNames']['postDecon_prefix']+'hv'+str(hashValue).zfill(5)+'.tif'
    #outPath = self.metaData['filePaths']['postDeconOutPath_'+computer]
    #outputScript += 'saveAs("Tiff", "'+outPath+'/'+fName+'");\n'
    outPath = self.getPath2File(hashValue,kwrd='postDecon',computer=computer)
    outputScript += 'saveAs("Tiff", "'+outPath+'");\n'
    outputScript += 'run("Close All");\n'
    # consider deleting 32 bit image.? Or alternatively start archiving the 8 bit files as we are working in SCRATCH anyway
    #output_fName = self.metaData['filePaths']['postDeconOutPath_'+computer]+'/'
    #output_fName += self.metaData['fileNames']['global_prefix']+self.metaData['fileNames']['postDecon_prefix']+'hv'+str(hashValue).zfill(5)+'.ijm'
    output_fName = self.getPath2File(hashValue,kwrd='dplPath',computer=computer,extension='_postDecon.ijm')
    with open(output_fName,'w') as f: f.write(outputScript)
    return "The postDecon imageJ macro for hashValue" + str(hashValue) + " was created at: " + output_fName

  def smartCrop(self,hashValue, computer='ODSY',output='log'):
    """
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
    """

    # read in the correct upstream input data
    metaData = self.metaData['smartCrop']
    inputKwrd = self.getPipelineStep('smartCrop')
    inputPath = self.getPath2File(hashValue,kwrd=inputKwrd,computer = computer)
    fullStack = flatField.zStack2Mem(inputPath) # this is just reading in the "full stack" before cropping but after hashing etc
    refPos = self.getCropIndex(hashValue)
    originLog = [0,0,0] # relative changes in origin and dimensions to be updated and recorded in writeLog()
    dimLog = [0,0,0]

    # if appropriate, crop out the uniform decon FFT artifacts in XY and Z
    if metaData['fftCrop']['bool'] == True:
      # get the stack dimensions
      dim = fullStack.shape
      crop = (metaData['fftCrop']['X'], metaData['fftCrop']['Y'],metaData['fftCrop']['Z'])
      #cropTriple = [(crop[i],dim[i]-crop[i]) for i in range(len(crop))]
      # crop the stack using stack[crop : dim - crop]
      #fullStack = flatField.cropStack(fullStack,cropTriple)
      fullStack = fullStack[crop[2]:dim[0]-crop[2], \
                  crop[1]:dim[1]-crop[1],\
                  crop[0]:dim[2]-crop[0]] # crop is xyz while dim is zyx
      for i in range(len(refPos)):
        refPos[i] = (refPos[i][0] + crop[i], refPos[i][1] - crop[i])
      originLog = [originLog[n] + crop[n] for n in range(len(crop))]
      dimLog = [dimLog[n] + -2*crop[n] for n in range(len(crop))]

    # is this hashvalue contain a sed/gel interface? Is it mostly sed or gel?
    sedGel = self.sedGelInterface(hashValue)
    if sedGel[1] == True and metaData['sedGelCrop']['bool'] == True:
      # Crop and make sure that it is within the cushion.
      # compute the additional cropping in z by computing the gradient in avg pixel intensity.
      avgIntensity_z = flatField.avgXYPixel(fullStack)
      pixelZGrad = flatField.zGradAvgXY(fullStack)
      maxValue = max(pixelZGrad)
      maxIndex = list(pixelZGrad).index(maxValue) # this is the z index of the max grad (?) I think
      # Now do some quality control on this max value:
      # is the max Value large enough?
      if maxValue < metaData['sedGelCrop']['minValue']:
        print("maximum gradient for sedGelCrop is below the minValue listed in metaData")
        print(maxIndex,maxValue,metaData['sedGelCrop']['minValue'])
        raise KeyError
      # Is the index close to where the purported sed/gel interface is?
      sedGelDeviation = abs((maxIndex + refPos[2][0] - metaData['sedGelCrop']['offset']) \
                            - self.metaData['imageParam']['gelSedimentLocation'])
      if sedGelDeviation > metaData['sedGelCrop']['maxDev']:
        print("The purported gel/sediment location is further than expected given the approx value listed in metaData")
        raise KeyError
      # crop the stack using the maxindex and uniform offset
      offset = metaData['sedGelCrop']['offset']
      zSlices = fullStack.shape[0]
      if sedGel[0] == 'sed': # crop from the bottom
        fullStack = fullStack[maxIndex - offset : zSlices ,:,:]
        refPos[2] = (refPos[2][0] + maxIndex - offset,refPos[2][1])
        originLog[2] = originLog[2] + maxIndex - offset
        dimLog[2] = dimLog[2] - maxIndex - offset
        print("Warning, on smartCrop, check to make sure you have enough slices to crop given offset")
      elif sedGel[0] == 'gel':
        fullStack = fullStack[0:maxIndex - offset,:,:]
        refPos[2] = (refPos[2][0], maxIndex - offset)
        dimLog[2] = dimLog[2] - (len(pixelZGrad) - maxIndex + offset)
        print("Warning, on smartCrop, check to make sure you have enough slices to crop given offset")

    # save the fully cropped file.
    output_fName = self.getPath2File(hashValue,kwrd='smartCrop',extension='.tif',computer=computer)
    flatField.array2tif(fullStack,output_fName)
    #refPos_fName = self.getPath2File(hashValue,kwrd='smartCrop', extension='.yaml', computer = computer)
    #with open(refPos_fName,'w') as f:
    #  yaml.dump(self.hash[str(hashValue)],f)
    #  #f.write(str(refPos))
    #  f.close()
    ## return a total cropping done with final reference pixel locations given.
    if output == 'log':
      #yamlLog  = "smartCrop:\n"\
      #yamlLog += "  cropBool: True\n"
      #yamlLog += "  origin: [" + self.index2key(originLog) + "]\n"
      #yamlLog += "  dim: [" + self.index2key(dimLog) + "]\n"
      #yamlLog += "  time: " + str(time)
      #return yamlLog
      return {'origin' : originLog, 'dim' : dimLog, 'refPos' : refPos}
    else: return refPos

  def makeSmartCrop(self,hashValue,computer='ODSY', output = 'log'):
    """
    make a smart crop python script. No heavy lifting. Just call the function.
    :param hashValue:
    :param computer:
    :return:
    """
    outScript = 'import sys\n'
    gitDir = self.getPath2File(0,kwrd='particleLocating',computer = computer,pathOnlyBool = True,extension='')
    #outScript += 'sys.path.insert(0,\"' + str(self.metaData['filePaths']['particleLocatingSCRIPTS_'+computer]) +'\")\n'
    outScript += 'sys.path.insert(0,\"' + gitDir +'\")\n'
    outScript +='import flatField\n'
    outScript += 'import dplHash_v2 as dplHash\n'
    outScript += 'dplInst = dplHash.dplHash(\'' + \
                 self.getPath2File(hashValue,kwrd='metaDataYAML',computer=computer) +'\')\n'
    if output == 'log':
      outScript += 'yamlLog = dplInst.smartCrop(' + str(hashValue) + ', computer = \'' + computer + '\')\n'
      outScript += 'dplInst.writeLog('\
              + str(hashValue)\
              + ',\'smartCrop\',yamlLog,'\
              + 'computer=\'' + str(computer)\
              +'\')\n' # this still needs to be written and output/input matching
    else: outScript += 'print(dplInst.smartCrop(' + str(hashValue) + ', computer = \'' + computer + '\'))\n'

    output_fName = self.getPath2File(hashValue,kwrd='dplPath',computer=computer,extension='_smartCrop.py')
    with open(output_fName,'w') as f: f.write(outScript)
    return print('makeSmartCrop created for hashValue: ' + str(hashValue))

  def makeScratchDir(self,computer='ODSY'):
    """
    This function directly makes the scratch directory folder structure given the scratch dictory listed in yaml metaData
    It does not retutrn anything. It checks if the directories already exist
    It is lightweight enough that it can be called on the odsy login nodes provided that you can import python
    """
    parentDir = self.metaData['filePaths']['scratchDirectory_'+computer]
    subDirList = self.metaData['filePaths']['scratchSubDirList']
    for dir in subDirList:
      d = os.path.join(parentDir,dir)
      #print("This is a scratch dir: ", d)
      if not os.path.exists(d): os.makedirs(d)


  def makeAllScripts(self,hashValue,computer='ODSY'):
    """
    single python method to create all the scripts for a given hashValue.
    Currently 4 scripts are created:
      [+] postprocessing and back ground subtraction in imageJ 
      [+] deconvolution in deconvolution lab 2 using java
      [+] postDecon preocessing to convert output 32 bit image to 8 bit and threshold in imageJ
      [+] particle locating in matlab
    """
    self.makeScratchDir(computer)
    # insert boolean flag based on yaml metaData file to choose preprocessing or flatfielding.
    self.makePreprocessing_imageJ(hashValue,computer)
    self.makeFlatFielding_python(hashValue,computer)
    self.makeDeconDL2_javaCommandLine(hashValue,computer)
    self.makeSmartCrop(hashValue,computer)
    self.makePostDecon_imageJ(hashValue,computer)
    self.makeParticleLocating_matlab(hashValue,computer)
    return None

  def makeDPL_bashScript(self,computer='ODSY'):
    """
    make a bash script to call and excure decon particle location. \
    The bash file is created once and called for each hashValue
    [ ] takes as input a hashValue as input ie /dpl.x 250
    [ ] automatically cycles through the pipeline steps in the corresponding yaml file and adds those steps with \
        true boolean flags.
    :param computer:
    :return:
    """
    # path to software
    fijiPath = self.metaData['filePaths']['fijiPath_'+computer]
    matlabPath = self.metaData['filePaths']['matlabPath_'+computer]
    # get the pipeline from the yaml file
    pipeline = self.metaData['pipeline']
    # now cycle over the pipeline keys and get the relevant filepaths and add the relevant pieces to bash exec script
    # I think the lines to execute bash commands should be added a nested function like exec_smartCrop or exec_postDecon

    def exec_hash():
      if computer == 'ODSY':
        output = self.metaData['filePaths']['loadPython_ODSY'] + '\n'
        output += "source activate tractionRheoscopy\n"
      elif computer == 'MBP' : output = ""
      output += '$(python -c \"'
      output += 'import sys;\n'
      gitDir = self.getPath2File(0, kwrd='particleLocating', computer=computer, pathOnlyBool=True, extension='')
      output += 'sys.path.append(\'' + gitDir + '\');\n'
      #output += 'sys.path.append(\'' + self.metaData['filePaths']['particleLocatingSCRIPTS_'+computer] + '\');\n'
      output += 'import dplHash_v2 as dplHash;\n'
      output += 'hashObject = dplHash.dplHash(\'${yamlPath}\');\n'
      output += 'hashObject.makeAllScripts($hashValue,computer=\'${computer}\');\n'
      output += 'hashObject.writeLog($hashValue,\'hash\',computer=\'${computer}\')\n'
      output += '\")\n'
      output += 'echo \"scripts created!\"\n'
      return output

    def logPython(pipeLine):
      #if computer == 'ODSY': output = self.metaData['filePaths']['loadPython_ODSY'] + '\n\n'
      #if computer == 'MBP' : output = ""
      output = ""
      output += '$(python -c \"'
      output += 'import sys;\n'
      #output += 'sys.path.append(\'' + self.metaData['filePaths']['particleLocatingSCRIPTS_'+computer] + '\');\n'
      gitDir = self.getPath2File(0, kwrd='particleLocating', computer=computer, pathOnlyBool=True, extension='')
      output += 'sys.path.append(\'' + gitDir + '\');\n'
      output += 'import dplHash_v2 as dplHash;\n'
      output += 'hashObject = dplHash.dplHash(\'${yamlPath}\');\n'
      output += 'hashObject.writeLog($hashValue,\''+ str(pipeLine) + '\',computer=\'${computer}\')\n'
      output += '\")\n'
      return output

    def exec_flatField():
      flatFieldScript_explicitHash = self.getPath2File(0,kwrd='dplPath',computer=computer,extension='_flatField.py')
      flatFieldScript = re.sub('_hv[0-9]*_','_hv${hvZeroPadded}_', flatFieldScript_explicitHash)
      extension = '.py' # note that extension as a kwarg in getPath2File() is outside the scope...
      #if computer == 'ODSY': output = self.metaData['filePaths']['loadPython_ODSY'] + '\n\n'
      output="" # initialize to empty string
      output += 'python '
      output += flatFieldScript + ' 1>' + flatFieldScript[0:-1*len(extension)] \
                + '.log 2> ' + flatFieldScript[0:-1*len(extension)] + '.err \n'
      output += 'wait \n'
      output += 'echo \"flatFielding is done!\"\n'
      output += ""
      return output

    def exec_decon():
      deconScript_explicitHash = self.getPath2File(0,kwrd='dplPath',computer=computer,extension='_decon.x')
      deconScript = re.sub('_hv[0-9]*_','_hv${hvZeroPadded}_', deconScript_explicitHash)
      extension = '.x'
      output = deconScript + ' 1>' + deconScript[0:-1*len(extension)] \
                + '.log 2> ' + deconScript[0:-1*len(extension)] + '.err \n'
      output += 'wait \n'
      output += 'echo \"deconvolution done!\"\n'
      output += ""
      return output

    def exec_smartCrop():
      smartCropScript_explicitHash = self.getPath2File(0,kwrd='dplPath',computer=computer,extension='_smartCrop.py')
      smartCropScript = re.sub('_hv[0-9]*_','_hv${hvZeroPadded}_', smartCropScript_explicitHash)
      extension = '.py'
      #if computer == 'ODSY': output = self.metaData['filePaths']['loadPython_ODSY'] + '\n\n'
      output="" # initialize to empty string
      output += 'python '
      output += smartCropScript + ' 1>' + smartCropScript[0:-1*len(extension)] \
                + '.log 2> ' + smartCropScript[0:-1*len(extension)] + '.err \n'
      output += 'wait \n'
      output += 'echo \"smartCrop done!\"\n'
      output += ""
      return output

    def exec_postDecon():
      postDeconScript_explicitHash = self.getPath2File(0,kwrd='dplPath',computer=computer,extension='_postDecon.ijm')
      postDeconScript = re.sub('_hv[0-9]*_','_hv${hvZeroPadded}_', postDeconScript_explicitHash)
      extension = '.ijm'
      if computer == 'MBP':
        output = "export DISPLAY=:123 \n"
        output += "Xvfb $DISPLAY -auth /dev/null & (\n"
        output += self.metaData['filePaths']['fijiPath_MBP']
      elif computer == 'ODSY':
        output = "/usr/bin/xvfb-run "
        output += self.metaData['filePaths']['fijiPath_ODSY']
      # Mosaic plugin do no work with --headless flag in fiji. Googled problem.\
      # solution on mosaic suite headless issue on forum.image.sc
      # This solution involve bakgrounding Xvfb, which will prevent matlab from quitting on subsequent locating step
      # xvfb-run is a utility that is available on ODSY and can be installed which creates a random Xvfb display
      # on a random channel (equivalent to DISPLAY:123") and then closes the instances after completion.
      output += " -batch " + postDeconScript + ' 1>' + postDeconScript[0:-1*len(extension)] \
                + '.log 2> ' + postDeconScript[0:-1*len(extension)] + '.err \n'
      # if computer == "MBP": output += ')\n' # dont forget the closing parenthesis started with \
                                              # hack around --headless not working
      #output += " --headless -macro " + postDeconScript + ' 1>' + postDeconScript[0:-1*len(extension)] \
      #         + '.log 2> ' + postDeconScript[0:-1*len(extension)] + '.err \n'
      output += 'wait \n' # for whatever reason this causes it to hang after closing FIJI
      output += 'echo \"postDecon is done!\"\n'
      output += ""
      return output

    def exec_locating():
      particleLocatingScript_explicitHash = self.getPath2File(0,kwrd='dplPath', \
                                                              computer=computer, extension = '_locating.m')
      particleLocatingScript = re.sub('_hv[0-9]*_','_hv${hvZeroPadded}_', particleLocatingScript_explicitHash)
      extension = '.m'
      if computer == 'ODSY': output = self.metaData['filePaths']['loadMatlab_ODSY'] +'\n' +'matlab '
      elif computer == 'MBP': output = self.metaData['filePaths']['matlabPath_'+ computer]
      output += " -nodisplay -nosplash -r "\
              "\"run(\'" + particleLocatingScript +"\'); exit\""
      output += " 1> " + particleLocatingScript[0:-1*len(extension)] + '.log' \
                + "2> " + particleLocatingScript[0:-1*len(extension)] + ".err \n"
      output += 'wait \n'
      output += 'echo \"particleLocating done!\"\n'
      output += ""
      return output

    # now loop over the pipeline and call the relevant exec functions
    pairList = [(list(elt.keys())[0],list(elt.values())[0]) for elt in pipeline]
    masterScript = "#!/usr/bin/env bash\n"
    masterScript += "hashValue=$1\n"
    #masterScript += "yamlPath=" + self.metaData['filePaths']['metaDataYaml_' + computer] + "\n"
    masterScript += "yamlPath=" + self.getPath2File(1,kwrd='metaDataYAML',computer=computer) +"\n"
    masterScript += "computer=" + str(computer) + "\n"
    masterScript += 'printf -v hvZeroPadded \"%05d\" ${hashValue}\n'  # create the zeroPadded bash variable
    masterScript += ""
    for elt in pairList:
      if elt[1] == True:
        try:
          masterScript += eval("exec_" + elt[0] + "()")
          if elt[0] in ['decon','postDecon','locating','tracking']: masterScript += logPython(elt[0])
        # This is a miserable design because any error in exec functions will be suppressed as likely NameError
        except NameError:
          print("exec_" + elt[0] +"(), probably has a bug. \
                                  Try explicitly calling the function without eval() and try again")

    output_fName_explicitHash = self.getPath2File(0,kwrd='dplPath', computer=computer,extension = '_exec_pipeline.x')
    output_fName = re.sub('_hv[0-9]*_', '_', output_fName_explicitHash)
    # we want to save this without a hashValue string as the file takes in a hashValue as an input arguement.
    with open(output_fName, 'w') as f:
      f.write(masterScript)
    os.chmod(output_fName, 509)  # google "chmod +x on python". No idea wtf octal literal is and what 509 means here
    return output_fName

  def makeDPL_bashScript_legacy(self,computer='ODSY',dplBool=[1,1,1,1,1]):
   """
   make a bash script to call and execute Decon Particle Locating. This bash files thats created here \
   is created once and call for each hashvalue
     [+] takes as input $HASHVALUE parameter from submit script. (ie set HASHVALUE = $1 or something to that effect)
         and file is executed for hashvalue 34298 as "./dpl.x 34298"
     [+] Call this python class and generate the directories and necessary files specific for the passed hashvalue
         -> This requires single python call to write the files to be executed below
     [+] call fiji and execute the preprocessing script (open, crop, gaussian bksubtract) whose filename \
         will be something containing $hashValue five zeropadded
         looks like $fijiPth --headless -macro someFileNamePrefix$i{hashvalue} > hv${hashValue}.log 
     [+] call java and execute DL2 with command line commands. 
     [+] call fiji and execute postDecon script to convert 32 bit to 8 bit and some basic thresholding
     [+] call matlab and execute particle locating input parameter script.
   In principle, the submission script for a given hash value just needs to execute this script for the given hash value. 
   RETURN: output path to saved file. Note saved filename has hashvalue modulo division baked in 
           ie hashValue=3 ->> leaf00_hv00003 and hashValue=10003 ->> leaf01_hv00003
   """
   # path to software
   fijiPath = self.metaData['filePaths']['fijiPath_'+computer]
   DL2Path = self.metaData['filePaths']['DL2Path_'+computer]
   matlabPath = self.metaData['filePaths']['matlabPath_'+computer]
   # filenames including full path but excluding extensions and bash variable dependece on hashValuee 
   #   ie "/full/path/globalPrefix_scriptPrefix_hv${hvZeroPadded}"
   filePathDict=self.metaData['filePaths']
   fileNameDict=self.metaData['fileNames']
   preprocessScript =filePathDict['preprocessOutPath_'+computer] +'/'\
                     +fileNameDict['global_prefix']+fileNameDict['preprocess_prefix']\
                     +'hv${hvZeroPadded}'
   deconScript =filePathDict['deconOutPath_'+computer] +'/'\
                +fileNameDict['global_prefix']+fileNameDict['decon_prefix']\
                +'hv${hvZeroPadded}'
   postDeconScript =filePathDict['postDeconOutPath_'+computer] +'/'\
                   +fileNameDict['global_prefix']+fileNameDict['postDecon_prefix']\
                   +'hv${hvZeroPadded}'
   particleLocationScript =filePathDict['locationsHashed_'+computer] +'/'\
                          +fileNameDict['global_prefix']+fileNameDict['location_prefix']\
                          +'hv${hvZeroPadded}'
   #
   output  = "#!/usr/bin/env bash\n"
   output += "hashValue=$1\n"
   output += "yamlPath="+self.metaData['filePaths']['metaDataYaml_'+computer]+"\n"
   output += "computer=" +str(computer)+"\n"
   output += 'printf -v hvZeroPadded \"%05d\" ${hashValue}\n' # create the zeroPadded bash variable
   output += ""
   # call python to create scripts
   if dplBool[0]==True:
     if computer=='ODSY': output +=self.metaData['filePaths']['loadPython_ODSY']+'\n\n'
     output += '$(python -c \"'
     output += 'import sys;'
     output += '  sys.path.append(\''+ self.metaData['filePaths']['dplPythonClass_ODSY'] +'\');'
     output += '  import dplHash;'
     output += '  hashObject = dplHash.dplHash(\'${yamlPath}\');'
     output += '  hashObject.makeAllScripts($hashValue,computer=\'${computer}\');\")\n'
     output += 'echo \"scripts created! Moving onto preprocessing\"\n'
   # call fiji headless for preprocessing
   if dplBool[1]==True:
     output += fijiPath + ' --headless -macro '+ preprocessScript+'.ijm 1> '\
                        + preprocessScript+'.log 2> '+ preprocessScript+'.err\n'
     output += 'wait \n'
     output += 'echo \"preprocessing done! Moving onto deconvolution\"\n'
   # call DL2 java to deconvolve
   if dplBool[2]==True:
     output += deconScript +'.x 1>'+ deconScript \
                           +'.log 2> '+ deconScript +'.err \n' 
     output += 'wait \n'
     output += 'echo \"deconvolution done! Moving onto postDecon\"\n'
   # call fiji for postDecon
   if dplBool[3]==True:
     output += fijiPath + ' --headless -macro '+ postDeconScript +'.ijm 1> '\
                        + postDeconScript+'.log 2> ' + postDeconScript +'.err \n'
     output += 'wait \n'
   # call matlab for particle locating
   if dplBool[4]==True:
     output += 'echo \"postDecon done! Moving onto particle locating\"\n'
     output += matlabPath + ' -nodisplay -nosplash -r \"run(\''+particleLocationScript+'.m\'); exit\" '
     output += '> ' + particleLocationScript+'.log 2> ' + particleLocationScript +'.err\n' 
   #output += self.decon_DL2_javaCommandLine(hashValue,computer=computer)
   # Must include some modulo loading functionality. 
   output_fName = self.metaData['filePaths']['particleLocatingSCRIPTS_'+computer]+'/'+'dplScript.x'
   with open(output_fName,'w') as f: f.write(output)
   os.chmod(output_fName,509) # google "chmod +x on python". No idea wtf octal literal is and what 509 means here
   return output_fName

  def makeSubmitScripts(self,computer='ODSY',resubmitBool=False):
    """
    Create a submission script for the given yaml file. Including *ALL* relevant chunks etc. \
    Dynamically create the correct number of submission scripts for case where the number of chunks exceeds 10k.\
    All jobs should be submitted to serial requue
    This submit script has to manipulate bash variables ${SLURM_ARRAY_TASK_ID} and run the correct\
    bash submission file for the hashValue
    [-] Most boneheaded way of doing this is to create a different bash file for each hashValue and a \
    few submission scripts with bash variables giving the leaf value (ie 00 01 02 would allow for 30k subjobs.)
    [-] Other option is that each subjob creates the scripts required by taking as input \
        the hashValue and the python hashTable. Do we really need 50k files?
        In what instance can't 50k files be replaced by a function that takes 50k input? \
        Isnt the hashTable class I've created literally the function I need?
        I could just call python with a variable passed by bash and hard coded leaf values and \
        then this program generates potentially unique scripts?
        or passes values downstream? I dont know how to pass values to fiji for example, \
        I pass values to DL2, I could pass values to matlab if I
        rewrote locating_input_parameters to take in SED or GEL and a unique path. \
        Does this really matter? How long will it take to make all the files
        for 50k chunks? And if file gneeration *also* happens in parallel then this is \
        negliblge computing cost. File generation is milliseconds on an hour job
        We also may not need two indices for jobs longer than 10k...just use job_array values 9999-19999 \
        and the job array *length* is less than 10k and the jobarray value naturally corresponds to the hashValue. \
        FALSE: we do need two indices for jobarray longer than 10k as the max value is \
        one less than the max array size, which for ODSY is 10k.
     [+] OK, so the submit script should call this python function and yaml file,
        write the relevant fiji and scripts, run them, and save the output
 
    COPY most recent submission script from Odyssey here 
    
    #!/usr/bin/env bash
    
    #SBATCH -J JobArray			# JobArray scehduling type used for submitting a group of similar serial jobs all at once.
    #SBATCH -o JobArray_locating_all.out	# Output file
    #SBATCH -e JobArray_locating_all.err	# Error file
    #SBATCH -p serial_requeue		# Requested queue, serial_requeue has lowest wait \
                                    # but maybe killed/restarted without warning
    #SBATCH --array=0-134			# Probably the size of the job array. \
                                    #For locating this would be set to the number of time steps \
                                    # with indexing starting at 1.
    #SBATCH -n 1				# Number of requested cores
    #SBATCH -t 360			# Requested runtime in Days-HH:MM, 240 is equiv to 0-02:00
    #SBATCH --mem=128000			# Memory pool for all cores in Mb
    #SBATCH --requeue			# Unclear what the difference between --requeue and -p serial_requeue
    
    module load matlab/R2016b-fasrc02
    
    matlab -nodisplay -nosplash -r "locating_input_parameters(${SLURM_ARRAY_TASK_ID});exit" \
      
    > myjob.${SLURM_ARRAY_TASK_ID}.out \
      
    2> myjob.${SLURM_ARRAY_TASK_ID}.err
    """
    header  = "#!/usr/bin/env bash\n\n"
    name = self.metaData['fileNamePrefix']['global']
    header += "#SBATCH -J " + name + "\n"
    header += "#SBATCH -o " + name + "dplLocating_%A_%a.out\n"
    header += "#SBATCH -e " + name + "dplLocating_%A_%a.err\n"
    #header += "#SBATCH -e JobArray_locating_all.err\n"
    header += "#SBATCH -p "+str(self.metaData['ODSY_Resources']['queue'])+"\n"
    footer  = "#SBATCH -n "+str(self.metaData['ODSY_Resources']['cores'])+"\n"
    footer += "#SBATCH -t "+str(self.metaData['ODSY_Resources']['time'])+"\n"
    footer += "#SBATCH --mem="+str(self.metaData['ODSY_Resources']['mem'])+"\n"
    if self.metaData['ODSY_Resources']['queue']=='serial_requeue': footer += "#SBATCH --requeue\n"
    
    if resubmitBool == False:
      # compute the number of jobArrays and set up second index if necessary
      hashSize = self.metaData['hashDimensions']['hashSize'] # 'hashDimension' is created during initialization \
                                                             # and is not on the yaml file
      leaf = math.floor(hashSize/10000)
      # now figure out what array index you need given the hashSize and number of leaves
      nTail_jobArray = hashSize - 10000*leaf
    elif resubmitBool ==True:
      # compute the maximum single entry need to be resubmitted
      # partition the actual values into leaves
      # basically this part needs to recreate the leaf and arrayIndex value corresponding to the missing hashValue
      missinghv_intList = self.resubmit_MissingHV()
      leaf = max(missinghv_intList)[0]
    for jobs in range(leaf+1):
      output = header
      if jobs==0: # you are in the remainders
        if resubmitBool==False: output += "#SBATCH --array=0-"+str(nTail_jobArray)+"\n"
        elif resubmitBool==True:
          output += "#SBATCH --array="
          tmp=''
          for hv in missinghv_intList:
            if hv[0]==jobs: tmp += str(hv[1])+','
          output += tmp.rstrip(',')
      else:
        if resubmitBool== False: output += "#SBATCH --array=0-9999\n"
        elif resubmitBool == True:
         output += "#SBATCH --array="
         tmp=''
         for hv in missinghv_intList:
           if hv[0]==jobs: tmp += str(hv[1])+','
         output += tmp.rstrip(',')
      output += footer
      output +="\n"
      output += "HASHVALUE=$((SLURM_ARRAY_TASK_ID+"+str((jobs)*10000)+"))\n" 
      # Create a bash variable for the true hashValue for this lead that we can pass to python. 
      # Note the bash expression $(( )) means "arithemtic interpretation" 
      # and that bash is sensitive to white space. ie 'HASHVALUE=122' will work, but 'HASHVALUE = 122' will not work
      # now $HASHVALUE can be passed instead of $SLURM_ARRAY_TASK_ID
      # call bash  with the bash variable $HASHVALUE which will execute a bash file with the correct hashValue as \
      # each serial_requeue cycles through jobArray
      #fName_dpl = self.metaData['filePaths']['particleLocatingSCRIPTS_'+computer]+'/'+'dplScript.x'
      fName_tmp = self.getPath2File(0, kwrd='dplPath', computer=computer, extension='_exec_pipeline.x')
      fName_dpl = re.sub('_hv[0-9]*_', '_', fName_tmp)
      output += "chmod +x "+fName_dpl+"\n"
      output += fName_dpl + " $HASHVALUE\n"
      #fName_SBATCH = self.metaData['fileNames']['global_prefix']+"dpl_leaf"+str(jobs).zfill(3)+'.sbatch'
      fName_SBATCH = self.getPath2File(0,kwrd='dplPath',computer=computer,\
                                       extension="dpl_leaf"+str(jobs).zfill(3)+'.sbatch')
      with open(fName_SBATCH,'w') as f: f.write(output)
    return "submission script *.sbatch file(s) written to: "\
           + self.getPath2File(0,kwrd='dplPath',computer=computer,pathOnlyBool=True)

  def getMissingValues(partial,complete):
    """
    This function takes two lists, partial and complete, and returns a list of the missing entries
    """
    return list(set(complete) - set(partial)).sort()

  def resubmit_MissingHV(self,computer='ODSY'):
    """
    [+] list all files that succeeded in forming particle locations 
    [+] parse on tag hv and type convert to list
    [ ] run the helper function getMissingValues()
    [ ] return a formatted string that should be added to submitScripts with boolean flag modification. 
    """
    complete = [x for x in range(self.metaData['hashDimensions']['hashSize'])]
    #fName_list=os.listdir(self.metaData['filePaths']['locationsHashed_'+computer])
    fName_list=os.listdir(self.metaData['filePaths']['locationsHashed_'+computer])
    # where does the phrase 'hv' start?
    hvStart = fName_list[1].find('hv')
    start,stop = hvStart + 2, hvStart + 4
    partial = []
    for fName in fName_list:
      partial.append(int(fName[start:stop]))
    missing_int = self.getMissingValues(partial,complete)
    return missing_int

  def hv2leafArray(hv):
    # take an integer hashvalue and return a pair (leaf,array) used for job submission
    # hv = leaf*10000+array
    leaf,array = math.floor(hv/10000),hv%10000
    return (leaf,array)

if __name__ == "__main__":
  # Tests to run
  # -load yaml file and call some simple hashValue entries
  yamlTestingPath = '/Users/zsolt/Colloid/SCRIPTS/tractionForceRheology_git/TractionRheoscopy\
                     /metaDataYAML/tfrGel09052019b_shearRun05062019i_metaData_scriptTesting.yaml'
  print("Loading yaml metaData file: ", yamlTestingPath)
  dplInst = dplHash(yamlTestingPath)
  print(len(dplInst.hash.keys()))
  dplInst.makeSubmitScripts()
  #print(dplInst.queryHash(350))
  #print(dplInst.getNNBHashValues(160))
  #print(dplInst.getOverlap(161,160))
  #print(dplInst.getCropIndex(160))
  #print(dplInst.getCropIndex(161))
  #pipeline = dplInst.getPipelineStep('postDecon')

  # Test loading filePaths
  #for k in dplInst.metaData['filePaths']['scratchSubDirList']:
    #print("filePath for keyworkd",str(k),": ",dplInst.getPath2File(0,kwrd=k, computer='MBP'))
  #for elt in dplInst.metaData: print(elt,':',dplInst.metaData[elt])
  # -print scripts for each stage of dplHash

  #   -flatField,
  #print(dplInst.makeFlatFielding_python(0))

  # writeLog
  # for key in ['hash','flatField','decon','postDecon']:
  #   print('writing log file for step: ' + key)
  #   dplInst.writeLog(249,key,computer = 'MBP')

  # smartCropOutput = dplInst.smartCrop(249,computer='MBP')
  # print(smartCropOutput)
  # print(type(smartCropOutput))
  # dplInst.writeLog(249,'smartCrop',smartCropOutput, computer = 'MBP')

  #   -decon,
  #print(dplInst.metaData['filePaths']['psfPath_MBP'])
  #print(dplInst.makeDeconDL2_javaCommandLine(1, computer='MBP'))

  #   -postprocess,
  #print(dplInst.metaData["hashDimensions"])
  print(dplInst.makeAllScripts(0,computer='MBP'))
  print(dplInst.makeDPL_bashScript(computer = 'MBP'))

  #   -particle locate

  #   -stitch

  #print(dplInst.smartCrop(250,computer='MBP'))
  # TODO:
  #  [ ] create stitching functions, both particle locations and deconvolved images.
  #  [+] update filepaths to be operational on MBP, not just placeholders.
  #      -flatfield
  #      -deconvolution
  #      -postDecon
  #      -particle Location
  #  [+] move files around on MBP to allow for a single hashValue to be processed
  #  [ ] FIJI upscaling? Not sure if this is necessary
  #  [ ] create visualization fucntions to automatically slice and project samples from the deconvolved data.
