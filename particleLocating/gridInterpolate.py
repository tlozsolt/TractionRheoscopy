from scipy.interpolate import griddata
from particleLocating import pyFiji, flatField
import numpy as np

def sampleImage(array,points):
    """
    Take an image array and samples the values at specified points
    :param array: image array
    :param points: tuple specifying how many samples to take in each dim (z,y,x)
    :return: values array compatible with scipy.interpolate.griddata
    """
    dim = array.shape
    samplePoints = np.mgrid[\
                   0:dim[0]:int(dim[0]/points[0]), \
                   0:dim[1]:int(dim[1]/points[1]),\
                   0:dim[2]:int(dim[2]/points[2])\
                   ]
    values = array[samplePoints] # no idea if this is going to work
    return values

#%%
testImgPath = '/Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/' \
              'OddysseyHashScripting/pyFiji/testImages'
microStack8bit = flatField.zStack2Mem(testImgPath+'/microStack8bit.tif')
sampleImage(microStack8bit,(4,4,4))

#%%
from joblib import Parallel, delayed
from math import sqrt
import time
[sqrt(i ** 2) for i in range(10)]
def makeTuple(i,j,k): return (i,j,k)
def sumTuple(i,j):
    print(i,j)
    time.sleep(3)
    print(i,j)
    return 10*i+0.1*j
def addN(i,j,N=5,m=3):
    print(i,j,N,m)
    return i+j+N

def addN_Curry(N): return addN(i,j,N=N)
#print([(i,j,k) for i in range(1) for j in range(2) for k in range(2)])
#print([sumTuple(i,j) for i in range(10) for j in range(10)])
#Parallel(n_jobs=2)(delayed(addN_Curry(10))(i,j,m=[1,2,3]) for i in range(3) for j in range(2))
tmp = Parallel(n_jobs=8)(delayed(sumTuple)(i,j) for i in range(10) for j in range(10))
print(tmp)
#Parallel(n_jobs=2)(delayed(str)(i +j) for i in range(10) for j in range(10))

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Plot histogram and ridgeline of image intensities with optional scaling:
#   -for absolute bit depth with labels
#   - max/min range
#   - probability densities or absolute counts.
h = sns.distplot(microStack8bit.ravel(),axlabel="Pixel Value (uint8)")
#h.axes.set_yscale('log')
#h.set_axis_labels("Pixel Value (uint8)","Normalized Density")
plt.show()


#%%
sns.set()
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
plt.plot(X, S, color="red", linewidth=2.5, linestyle="-")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.ylim(C.min()*1.1,C.max()*1.1)
plt.yticks([-1, 0, +1],
[r'$-1$', r'$0$', r'$+1$'])

plt.show()
