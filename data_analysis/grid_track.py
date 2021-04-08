import pandas as pd
import numpy as np
import yaml
from data_analysis import static
from scipy.spatial import cKDTree

def loadMetaData(fPath,key):
    with open(fPath) as f:
        metaData = yaml.load(f, Loader=yaml.FullLoader)
    return metaData[key]

def samplePtsLine(lineDict, spacing = None , px2Micron = None):
    """
    Sample
    """
    # init
    if px2Micron is None: px2Micron = np.array([0.15,0.115,0.115])
    if spacing is None: spacing = {'extensionFactor' : 0.3, 'dt': 0.5, 'units': 'um'}

    # get position vector in microns
    p0_keys = ['{}0'.format(x) for x in ['z','y','x']]
    p1_keys = ['{}1'.format(x) for x in ['z','y','x']]
    p0_zyx = px2Micron * np.array([lineDict.get(key) for key in p0_keys])
    p1_zyx = px2Micron * np.array([lineDict.get(key) for key in p1_keys])

    # direction vector
    d = p1_zyx - p0_zyx
    length = np.linalg.norm(d)

    # sampling
    ext = spacing['extensionFactor']
    t_sample = np.arange(-1*ext, (1 + ext)*length, spacing['dt'])

    return np.array([p0_zyx + t*d/length for t in t_sample])


def findGridParticle(pos_df, lineDict, t0=0, tf=89, posKeys=None, cutoff_dict=None, px2Micron=None):

    # set default parameters
    if cutoff_dict is None: cutoff_dict = {'d' : 5, 'units' : 'um'}
    if px2Micron is None: px2Micron = np.array([0.15, 0.115, 0.115])
    if posKeys is None: posKeys = ['{} (um, imageStack)'.format(x) for x in ['z', 'y', 'x']]

    # get complete trajectories
    idx_complete = pos_df.xs(t0, level='frame').index.intersection(pos_df.xs(tf, level='frame').index)

    # create numpy array of complete trajectories and coordinates.
    pos_t0_np = pos_df.xs(t0, level='frame').loc[idx_complete][posKeys].to_numpy()

    # create binary search tree
    tree = cKDTree(pos_t0_np)

    # query the tree with all the points in gridSamplePts
    gridSamplePts = samplePtsLine(lineDict, px2Micron=px2Micron)
    _grid_treeId = tree.query_ball_point(gridSamplePts, cutoff_dict['d'])

    # get unique indices
    _grid_treeId_flat = np.array([item for sublist in _grid_treeId for item in sublist])
    grid_treeId = np.unique(_grid_treeId_flat.flatten())

    # convert np index to pandas index using idx_complete,
    # probably also iloc (locate by position in list, not index val)
    try: grid_id = idx_complete[grid_treeId]
    except IndexError:
        if grid_treeId.shape[0] == 0 :
            print("Warning, no particles found within cutoff distance to line {}".format(lineDict))
            return None
        else: raise IndexError('There is some other problem')

    # compute xy distance and z distance to grid line
    def parseLineDict(lineDict):
        # get position vector in microns
        p0_keys = ['{}0'.format(x) for x in ['z', 'y', 'x']]
        p1_keys = ['{}1'.format(x) for x in ['z', 'y', 'x']]
        p0_zyx = px2Micron * np.array([lineDict.get(key) for key in p0_keys])
        p1_zyx = px2Micron * np.array([lineDict.get(key) for key in p1_keys])

        d = p1_zyx - p0_zyx
        l = np.linalg.norm(d)
        n = d/l
        return [p0_zyx, p1_zyx, d, l, n]

    def dist2Line(p0, n, p):
        # shortest vector from point p to line defined by point p0 and direction vector n
        v = (p0 - p) - ((p0-p) @ n) * n
        return np.linalg.norm(v), v

    p0,p1,d,l,n = parseLineDict(lineDict)
    N = grid_id.shape[0]
    p = pos_t0_np[grid_treeId,:]
    v = (p - p0) - np.vstack(((p - p0) @ n)) * np.vstack([n] * N)
    return pd.DataFrame(data=v, index=grid_id,columns=['v_{}'.format(x) for x in posKeys])


if __name__ == '__main__':
    fPath = '/Users/zsolt/Colloid_git/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'

    meta = loadMetaData(fPath,'data_analysis')
    pos0 = static.loadParticle(0)
    cutoffDict = {'d' : 1, 'units' : 'um'}

    out = []
    for lines in meta['grid']['lines']:
        out.append(findGridParticle(pos0,lines,t0=0,tf=0,cutoff_dict=cutoffDict))
    gridParticle = pd.concat(out)
    #findGridParticle(pos0, meta['grid']['lines'][0], t0=0, tf=0, cutoff_dict=cutoffDict)
