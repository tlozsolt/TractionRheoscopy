# functions used to assess whether pariticle locating runs failed.
# Some example of failures:
# - dramatically fewer particles located due to misidentified gel/sediment phase
#   during ilastik pxClassifier.

from particleLocating import dplHash_v2 as dpl
import pandas as pd
import os
import yaml

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
        if f.endswith('.yaml'): log = yaml.safe_load(open(path + f))
        else: continue
        #try: log = yaml.safe_load(open(path + f))
        #except IsADirectoryError: pass
        try:
            hvList.append(log['hashValue'])
            particleCount.append(log['locating']['particles'])
        except KeyError:
            print("hashvalue {} did not get to locating step".format(log['hashValue']))
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

def compileOutliers(particleCount, hashSize=125, k=1.5):
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
    return pd.concat(out)

if __name__ == '__main__':
    #logPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/location_2021_05_24/log/'
    logPath = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/debug/locations_log_archive_10_06_2021/log/'
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

