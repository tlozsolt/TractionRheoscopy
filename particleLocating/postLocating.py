# functions used to assess whether pariticle locating runs failed.
# Some example of failures:
# - dramatically fewer particles located due to misidentified gel/sediment phase
#   during ilastik pxClassifier.
import pyperclip

from particleLocating import dplHash_v2 as dpl
import pandas as pd
import numpy as np
import os
import yaml
import seaborn as sns
from matplotlib import pyplot as plt

def binHashValue(hash_df):
    """
    takes in a hashTable from dplInst and returns a pandas or numpy array with
    output[t] is a list of hashValues for a given time t in the hashTable.
    """
    items = hash_df.groupby(['x','y','z']).groups.items()

    # now just restructure the dataframe. I am sure there is a better way than this...
    hvList = []
    keyList = []
    for elt in items:
        hvList.append([int(x) for x in elt[1]])
        keyList.append(elt[0])

    # combine lists to dataFrame
    _binHash = pd.DataFrame(data=hvList, index=keyList)

    # return the dataFrame with some renaming of the columns and reset the index to just index the hashValue
    # over a single time step
    return _binHash.reset_index().rename(columns = {'index':'xyzt'})

def particleCountFromLog(path):
    """
    read a folder of log files and return a dataFrame of number of particles found and the hashValue.
    basically parse the yaml log files into a dataFrame
    """
    hvList = []
    particleCount = []
    for f in os.listdir(path):
        if f.endswith('.yaml'): log = yaml.safe_load(open(path + '/' + f))
        else: continue
        #try: log = yaml.safe_load(open(path + f))
        #except IsADirectoryError: pass
        try:
            hvList.append(log['hashValue'])
            particleCount.append(log['locating']['particles'])
        except KeyError:
            print("hashvalue {} did not get to locating step or some other problem occurred".format(f))
            particleCount.append(-1)
    return pd.DataFrame(data=particleCount, index=pd.Index(hvList, name='hv')).rename(columns={0:'particle count'})

def _tukeyFence(k, df, col_key):
    # set up the fence
    q25, q75 = df.describe().loc[['25%','75%']][col_key].values
    w = k*(q75 - q25)
    (l,u) = (q25 - w, q75 + w)
    return (l,u)


def detectOutliersTukey(particleCount, k = 3, col_key = 'particle count'):
    """
    Detects outliers in dataframe using Tukey's fences, a simple interquartile test with one adjustable
    parameter k.
    (lower, upper) = ( (Q25 - k(Q75 - Q25), Q75+ k(Q75 - Q25) )
    Most of this function carries out formatting of dataframes

    particleCount has this format:
    hv particle count
    75       16030
    200      15975

    Basic call:
    for elt in particleCount.loc[binHash.drop('xyzt',axis=1).loc[3].values].sort_index()['particle count']:
        l,u = 16004 - 3*(16040-16004), 16040 + 3*(16040-16004)
        if elt <l or elt > u: print(elt)
    """
    # set up the fence
    #q25, q75 = particleCount.describe().loc[['25%','75%']][col_key].values
    #w = k*(q75 - q25)
    #(l,u) = (q25 - w, q75 + w)
    (l,u) = _tukeyFence(k, particleCount,col_key)

    # now index and select with boolean expression based on fence parameters
    outlier_df = particleCount.loc[(particleCount[col_key] < l ) | (particleCount[col_key] > u)]

    return outlier_df

def compileOutliers(particleCount, binHash, k=1.5):

    # compute hash size, I found a bug that assumed a constant hash size of 125
    # and hence did not detect outlier for hv % t above 125
    hashSize = binHash.index.max() + 1

    out = []
    for hv_modt in range(hashSize):
        particleCount_slice =particleCount.loc[binHash.drop('xyzt', axis=1).loc[hv_modt].values].sort_index()
        outlier = detectOutliersTukey(particleCount_slice,k=k).copy()
        l,u = _tukeyFence(k, particleCount_slice, 'particle count')
        if len(outlier.index > 0 ):
            outlier['hv % t'] = hv_modt
            outlier['Tukey lower'] = l
            outlier['Tukey upper'] = u
            out.append(outlier)
            #print(hv_modt)
            #print(detectOutliersTukey(particleCount_slice))
    try: out = pd.concat(out)
    except:
        print('No outliers to concatenate')
        pass
    return out

def test(logPath, metaPath):
    hash_df = dpl.dplHash(metaPath).hash_df
    particleCount = particleCountFromLog(logPath)
    binHash = binHashValue(hash_df)
    outliers = compileOutliers(particleCount,binHash)
    return outliers, binHash, particleCount

#%%
def plotParticleCount(outliers, binHash, particleCount, hashDim=None, path=None, fName_frmt = None):
    if hashDim is None: hashDim = 125
    if path is None: path = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/scratch/particleCount_plot'
    if fName_frmt is None: fName_frmt = 'particleCounts_{mat}_{xyz}.png'

    sns.set(rc={'figure.figsize': (16, 9)})
    sns.set_context("talk")
    # g = sns.lineplot(y=par, x=range(90),label='Shear displacement')
    #g = sns.lineplot(y=perp, x=range(90), label='Displacement perpendicular to shear')
    #g.legend()

    for hv in range(hashDim):
        plt.clf()
        # what is the xyz value?
        xyz = binHash.loc[hv].pop('xyzt')
        if xyz[-1] < 2: mat = 'gel'
        else: mat = 'sed'

        data = particleCount.loc[binHash.drop('xyzt', axis=1).loc[hv].values]
        g = sns.lineplot(data=data, label='Particle count (x,y,z) = {}'.format(xyz))

        # what fileName?
        fName = fName_frmt.format(xyz=''.join([str(elt) for elt in xyz]), mat=mat)
        g.figure.savefig(path +'/{}'.format(fName))

def plotParticle(key, qcDict, hv, path):
    pc = qcDict[key]['particleCount']
    binHash = qcDict[key]['binHash']
    xyz = binHash.loc[hv].pop('xyzt')
    data = pc.loc[binHash.drop('xyzt',axis=1).loc[hv].values]
    data['time'] = np.arange(data.shape[0])
    #return sns.lineplot(data=data, y='particle count', x='time', label = 'Particle count (x,y,z) = {}'.format(xyz))

    g = sns.lineplot(data=data, y='particle count', x='time', label = 'Particle count (x,y,z) = {}'.format(xyz))

    # what fileName?
    fName = fName_frmt.format(xyz=''.join([str(elt) for elt in xyz]), mat=mat)
    g.figure.savefig(path + '/{}'.format(fName))
    return True


#%%
# compute derivative
#data = particleCount.loc[binHash.drop('xyzt', axis=1).loc[89].values].to_numpy().squeeze()
# resets the index
#diff = data.apply(np.diff)
# >>> data -> [14690, 3415, .., 16840]
#pd.DataFrame(data=np.concatenate([np.array(data.iloc[-1] - data.iloc[1]),data.apply(np.diff).values.squeeze()]), index=data.index)


#%%
def _hvTraj(hv, particleCount, binHash):
    return particleCount.loc[binHash.drop('xyzt', axis=1).loc[hv].values]

def _deltaParticleCount(hvTraj):
    #hvTraj = _hvTraj(hv,particleCount_df, binHash_df)
    return pd.DataFrame(data=np.concatenate([np.array(hvTraj.iloc[-1] - hvTraj.iloc[1]),
                                             hvTraj.apply(np.diff).values.squeeze()]), index=hvTraj.index,columns={'deltaCounts'})

def deltaParticleCount(hv, particleCount, binHash):
    hvTraj = _hvTraj(hv, particleCount, binHash)
    diff = _deltaParticleCount(hvTraj)
    data = hvTraj.join(diff)
    data['deltaCounts/Counts'] = data['deltaCounts']/data['particle count']
    return data

#%%
def clipIndex(pd_index):
    """
    Takes a pandas index object and copies the content to the clipboard without commas as required by bash scripts.
    """
    N = pd_index.size
    out = ' '.join(map(str,list(pd_index)))
    pyperclip.copy(out)
    return print("Copied index of length {} to clipboard".format(N))

#%%
# find the indices you need to resubmit based on results in log folder (index resub) and differnt criteria (index)
#index_resub = particleCountFromLog('/path/to/archive/')
#
#out = []
#for hv in range(125): out.append(deltaParticleCount(hv,particleCount,binHash))
#resub = pd.concat(out)
#
##apply selection criterion of 10% change up or down on particle count
#index = resub[abs(resub['deltaCounts/Counts']) > 0.1].index
#
## note the manual copying of hash values 455 and 720 which were hashValues that were submitted in index_resub but did
## not complete
#index.difference(index_resub.intersection(index)).union(pd.Index([455,720]))


#%%
if __name__ == '__main__':
    #logPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/location_2021_05_24/log/'
    #logPath = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/debug/locations_log_archive_10_06_2021/log/'
    logPath = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/Transfer_ODSY/tfrGel10212018A_shearRun10292018f/tfrGel10212018A_shearRun10292018f/log/'
    metaDataPath = '/Users/zsolt/Colloid_git/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'

    hash_df = dpl.dplHash(metaDataPath).hash_df

    particleCount = particleCountFromLog(logPath)
    binHash = binHashValue(hash_df).drop(['xyzt'],axis=1)

    #hv_mod_t = 0 # this is the hashvalue for a fixed location in the image.

    # get a list of particle counts in a fixed hashvalue region across time
    hvList = []
    for hv_mod_t in range(125):
        particleCount.loc[binHash.loc[hv_mod_t].values]

    binHash = binHashValue(hash_df)

    # index into particleCount using the aggregated index of binHash (ie hashvalue list for a fixed xyz hash chunk over
    # time.
    particleCount.loc[binHash.drop('xyzt', axis=1).loc[0].values].sort_index()


    # what is the goal? Find the hashValues that have unexepectedly **low** number of particles. Sharp dips in particle
    # count over time.
    count = 0
    for hv_modt in range(125):
        particleCount_slice =particleCount.loc[binHash.drop('xyzt', axis=1).loc[hv_modt].values].sort_index()
        outlier = detectOutliersTukey(particleCount_slice)
        if len(outlier.index > 0 ):
            count += 1
            print(hv_modt)
            print(detectOutliersTukey(particleCount_slice))
    print('Total failed hv with Tukey fence of 3 is {}'.format(count))

