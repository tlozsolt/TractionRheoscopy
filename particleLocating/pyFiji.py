import flatField
import pyperclip
import numpy as np
import skimage as ski
from scipy import ndimage

def send2Fiji(arrayList,\
              wdir ='/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
                    'OddysseyHashScripting/pyFiji/testImages/'):
  """ This function takes a nparray or a list of np.array
  [+] writes each array to tif in wdir using a tmp.tif
  [+] creates an text object that contains a fiji macro
  [+] copies the macro to the system keyboard.
  After it runs, if you go to fiji and press fn-F1, fiji will take the text on system clipboard and run it as macro
  This last step is accomplished by creating a simple macro and mapping it it to F1 using fiji shortcut
  I also created a macro to close all windows in fiji and mapped that to F2.
  """
  def send2FijiSingleArray(nparray,index):
    path = flatField.array2tif(nparray,wdir+'/tmp'+str(index)+'.tif')
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

if __name__ == "__main__":
  # open an array and apply a filter
  testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/'\
                'OddysseyHashScripting/pyFiji/testImages'
  img = flatField.zStack2Mem(testImgPath + '/stack8bit.tif')
  blurImg = flatField.gaussBlurStack(img,sigma=2)
  print(send2Fiji([blurImg.astype('uint16'),img],wdir=testImgPath))

