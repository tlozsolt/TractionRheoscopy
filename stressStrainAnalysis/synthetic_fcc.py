import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import trackpy as tp
import seaborn as sns
from matplotlib import pyplot as plt

"""
This code generates a synthetic fcc lattice with three coaxial stacking faults bounded by 
partial dislocation loops with the stacking sequence (left fcc, stacking faults, right fcc):

   f f f f f f f f f f f f f 
 A B C A B C A B C A B C A B C
 A B C A B A B C B C A C A B C 
   f f f h h f h h f h h f f
   

A -> (0,0)
B -> (a/2, a/(2*sqrt(3))
C -> (a, sqrt(3)/3)  

and the spacing between the planes d_{111} is a/sqrt(3)
"""

def makeClosePackedLayer(a, dim=10, z=0,xy_offset=[0,0]):
    """
    makes a single close packed layer with lattice spacing a and
    :param dim: number of atoms along 110 direction
    :param a:
    :return: DataFrame of positions with columns x, y, z
    """
    out = []
    for n_101 in range(-5*dim,5*dim,1):
        for n_110 in range(-5*dim, 5*dim+1, 1):
            pt = (a*n_110 + a/2*n_101+xy_offset[0],np.sqrt(3)/2*a*n_101+xy_offset[1],z)
            #pt = (n_101,n_110)
            out.append(pt)
    out_dataFrame = pd.DataFrame(out,columns=['x','y','z'])
    # crop
    return out_dataFrame[(out_dataFrame['x']<dim) &\
                         (out_dataFrame['x']>-1*dim) &\
                         (out_dataFrame['y']<dim) &\
                         (out_dataFrame['y']>-1*dim)]

def stackSequence(seq,a=1.55,dim=15):
    """
    :param seq:
    :return: 
    """
    stack = []
    for n in range(len(seq)):
        layer = seq[n]
        if layer =='A': stack.append(makeClosePackedLayer(a,dim=dim, z=n*a/np.sqrt(3),xy_offset=[0,0]))
        elif layer =='B': stack.append(makeClosePackedLayer(a,dim=dim, z=n*a/np.sqrt(3),xy_offset=[a/2,a/(2*np.sqrt(3))]))
        elif layer =='C': stack.append(makeClosePackedLayer(a,dim=dim, z=n*a/np.sqrt(3),xy_offset=[a,a*np.sqrt(3)/3]))
        else: raise ValueError
    return pd.concat(stack)

def weld(df1, df2,diameter):
    merge = pd.concat([df1,df2])
    bTree = BallTree(merge.values)
    # this will return a nested array in which each entry is an array of incides
    # of all pts within diameter of the index in merge. If the len of entry is one then it is not close to anything
    pairList = bTree.query_radius(merge.values,r=0.6*diameter)
    noDoubles = pd.DataFrame([merge.values[n[0]]for n in pairList if len(n)==1],columns=['x','y','z'])
    doubleHits = [np.sort(double) for double in pairList if len(double)>1]
    noDoubleSet = set([elt[0] for elt in doubleHits])
    noDoublesDF = pd.DataFrame([merge.values[n] for n in noDoubleSet],columns=['x','y','z'])
    return pd.concat([noDoubles,noDoublesDF])

if __name__ == "__main__":
    fcc_background = stackSequence('ABC'*5,dim=25)
    fcc_no_core = fcc_background[(fcc_background['x'] >  14) |
                                 (fcc_background['x'] < -14) |
                                 (fcc_background['y'] >  14) |
                                 (fcc_background['y'] < -14)]
    stackingFaultCore = stackSequence('ABCABABCBCACABC')
    merge_noDoubles = weld(fcc_no_core, stackingFaultCore,1.55)
    #sns.scatterplot(layer['x'], layer['y'])
    #plt.show()
    merge_noDoubles.to_csv('/home/zsolt/buf/stackingFault.csv',sep=' ')
