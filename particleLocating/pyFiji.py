from particleLocating import flatField
import pyperclip
import numpy as np
import datetime
import skimage as ski
from scipy import ndimage
import tifffile

def send2Fiji(arrayList, wdir = None, metaData=None):
  """ This function takes a nparray or a list of np.array
  [+] writes each array to tif in wdir using a tmp.tif
  [+] creates an text object that contains a fiji macro
  [+] copies the macro to the system keyboard.
  After it runs, if you go to fiji and press fn-F1, fiji will take the text on system clipboard and run it as macro
  This last step is accomplished by creating a simple macro and mapping it it to F1 using fiji shortcut
  I also created a macro to close all windows in fiji and mapped that to F2.
  """
  if wdir is None: wdir = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/pyFiji/'
  def send2FijiSingleArray(nparray,index):
    dataStr = str(datetime.date.today())
    #path = flatField.array2tif(nparray,wdir+'/tmp_'+ dataStr + str(index)+'.tif',metaData=metaData)
    path = wdir + 'tmp_{}.tif'.format(index)
    tifffile.imwrite(path, nparray)
    return 'open("'+ path +'");\n'
  macroText = ''
  if type(arrayList) == np.ndarray:
    macroText += send2FijiSingleArray(arrayList,0)
  elif type(arrayList) == list:
    for n in range(len(arrayList)):
      macroText += send2FijiSingleArray(arrayList[n],n)
  pyperclip.copy(macroText)
  print("Images saved to tif and copied to system clipboard.")
  return macroText

def recastImage(imgArray, dtypeOut):
  """
  output an array where each value has been recast to a new data type without any other change
  The entire dynamic range of the image is remapped to the output bit depth. There is no clipping.
  :param imgArray: np.array of image data
  :param dtypeOut: str or dict specifying output data type.
      'uint16': rescale to array max and array min and convert to 16bit integer
      'uint8': rescale to array max and array min and convert to 8bit integer
      'uint16_corr': rescale to **fixed** range of -1 to 1 and convery to 16bit integer
      'uint16_dict': dictionary with custom max and min values and 16 bit output

  :return:
  """
  if dtypeOut == 'uint16':
    min, max = 0.99 * np.nanmin(imgArray), 1.01 * np.nanmax(imgArray)
    m = 2 ** 16 / (max - min)
    b = 2 ** 16 - m * max
    mArray = np.full(imgArray.shape, m)
    bArray = np.full(imgArray.shape, b)
    return np.array(np.multiply(mArray, imgArray) + bArray).astype('uint16')
  elif dtypeOut == 'uint8':
    min, max = 0.99 * np.nanmin(imgArray), 1.01 * np.nanmax(imgArray)
    m = 2 ** 8 / (max - min)
    b = 2 ** 8 - m * max
    mArray = np.full(imgArray.shape, m)
    bArray = np.full(imgArray.shape, b)
    return np.array(np.multiply(mArray, imgArray) + bArray).astype('uint8')
  elif dtypeOut == 'uint16_corr':
    min, max = -1.00001,1.00001
    m = 2 ** 16 / (max - min)
    b = 2 ** 16 - m * max
    mArray = np.full(imgArray.shape, m)
    bArray = np.full(imgArray.shape, b)
    return np.array(np.multiply(mArray, imgArray) + bArray).astype('uint16')
  elif type(dtypeOut) == dict:
    min, max = dtypeOut['min'], dtypeOut['max']
    m = 2 ** 16 / (max - min)
    b = 2 ** 16 - m * max
    mArray = np.full(imgArray.shape, m)
    bArray = np.full(imgArray.shape, b)
    return np.array(np.multiply(mArray, imgArray) + bArray).astype('uint16')
  else:
    raise ValueError('recasting is only availabe to uint8 and uint16, not dtypeOut=', dtypeOut)


if __name__ == "__main__":
  # open an array and apply a filter
  testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/'\
                'OddysseyHashScripting/pyFiji/testImages'
  img = flatField.zStack2Mem(testImgPath + '/stack8bit.tif')
  blurImg = flatField.gaussBlurStack(img,sigma=2)
  print(send2Fiji([blurImg.astype('uint16'),img],wdir=testImgPath))

