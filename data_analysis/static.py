import pandas as pd
import trackpy as tp
import numba
import dask.dataframe as ddf
from scipy.spatial import cKDTree
import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
#from shapely.ops import polygonize, unary_union
#from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point

import yaml


def loadData2Mem(path, col_list=None):
    """
    loads data from trackpy hdfstore to memory, selecting only
    columns in col_list

    :path: str, path, including file name to hdf store
    :col_list: list of str, corresponding to column names in hdfStore

    return: pandas dataframe

    # get the first time frame
    >>> df.xs(1,level='frame')
    # or
    >>> df.loc[1]

    # get the trajectory dataframe for aprticle 12432
    >>> df.xs(12432, level='particle')

    # get trajectories with length longer than 10
    # returned with the same multidex format as the input...so it can be resliced
    >>> df.groupby('particle').filter(lambda x: len(x.index) > 10)

    # go back to single index on particle for use with dask (which does not support multi index)
    >>> df.reset_index('frame')

    I think the best way to operate on trajectories to use dask
    is to have the index be particles.
    groupby particles and have numba functions operate on the trajectory...that is pass
    numpy array of positions and frames to numba compiled function
    and use cKDTree for nnb lookup...what happens if you loose a nnb during the time over which
    the strain is calculated? Just skip it, record the failed particle id, and print a warning.

    """
    if col_list is None:
        col_list = ['x (um, imageStack)', 'y (um, imageStack)', 'z (um, imageStack)',
                    'particle', 'frame', 'x_std', 'y_std', 'z_std']
    with tp.PandasHDFStore(path) as s:
        # concat using iterable s (from trackpy tutotorial) and select col_list
        df = pd.concat(iter(s))[col_list]

        #create multiindex of frame,particle id
        df = df.set_index(['frame','particle'])

        # sort in place to get faster indexing later
        df.sort_index(inplace=True)
    return df

def bin(df, zlabel = 'z (um, imageStack)', nbins = 10,frame=None):
    """
    bin data df in z using pandas cut function
    function does not care what the other columns are
    This is essentially a groupby applied
    to column of bin labels. Any column can be computed
    on and aggregated based on the zbin value
    """
    if frame == None:
        print('binning called on all frames')
        tmp=df
    else:
        tmp = df.xs(frame,level='frame').copy()
    tmp['bin'] = pd.cut(tmp[zlabel],nbins)
    return tmp.groupby('bin')

def fitTopSurface(df, frame=None, pos_keys=None, n_bin = 15):
    """
    Fit the top surface of the gel or sediment to a linear plane and return a dictionary of fit parameters
    frame is either:
       None (all frames)
       int specfiying which frame to return
       list of int

       # Normal vector to the fitted plane (reminder or basic linear algebra or affine planes)
       >>> n = np.array(fit[0],fit[1],1)
       >>> n_unit = n/np.squrt(n.dot(n))
       # can also be obtained by evaluating fit at any 3 non-colinear (x,y) pts to get 3 linearly independent vectors.
       # and computing cross prodoct of the resulting displacement vectors
       >>> p1 = (1,1)
       >>> p2 = (35,150)
       >>> p3 = (75,10)
       >>> v1 = np.array([p1[0],p1[1],np.dot(fit,np.array([p1[0],p1[1],1]))])
       >>> v2 = np.array([p2[0],p2[1],np.dot(fit,np.array([p2[0],p2[1],1]))])
       >>> v3 = np.array([p3[0],p3[1],np.dot(fit,np.array([p3[0],p3[1],1]))])
       # now take the cross product of displacement vectors
       >>> n = np.cross(v3-v2, v3-v1)
       >>> n_unit = n/np.sqrt(n.dot(n))
    """
    if frame == None:
        frame=range(max(df.index.get_level_values('frame') +1))
    elif type(frame) == int: frame = [frame]
    else: pass

    if pos_keys == None:
        pos_key = {}
        pos_key['x'] = 'x (um, imageStack)'
        pos_key['y'] = 'y (um, imageStack)'

    out = {}
    for t in frame:
        tmp = df.xs(t,level='frame').copy()
        tmp['xbin'] = pd.cut(tmp[pos_key['x']], n_bin)
        tmp['ybin'] = pd.cut(tmp[pos_key['y']], n_bin)
        x = tmp.groupby(['xbin', 'ybin']).max()['z (um, imageStack)'].reset_index()['xbin'].apply(lambda x: 0.5 * (x.left + x.right)).to_numpy()
        y = tmp.groupby(['xbin', 'ybin']).max()['z (um, imageStack)'].reset_index()['ybin'].apply(lambda x: 0.5 * (x.left + x.right)).to_numpy()
        z = tmp.groupby(['xbin', 'ybin']).max()['z (um, imageStack)'].reset_index()['z (um, imageStack)'].to_numpy()
        A = np.vstack([x,y,np.ones(len(x))]).T
        try:
            fit, residual, rank, s = np.linalg.lstsq(A, z)
            out[t] = {'fit ax + by + c': fit, 'residual': residual, 'rank' : rank, 's': s}
        except np.linalg.LinAlgError:
            out[t] = 'fit did not converge with nbin = {} is likely too small'.format(n_bin)
            pass
    return out

def distFromPlane(df,out_key, fit_dict,pos_keys=None, frame=None, method='best_fit'):
    """
    Compute the vertical distance for particle coordinates
    and fit parameters a,b,c where the plane is defined by the euqation z_0 = ax + by + c
    Negative values mean points lie below the plane, and positive values above the plane.

    Parameters
    ~~~~~~~~~~

    : df dataframe of particle positions with hierarchical index of (frame, particle id)
    : out_key: str giving the key to save the output distance from plane
    : fit_dict: dictionary of output from fitTopSurface() function. Keys are from np.linagl.lstsq()
                - t: int giving the frame number
                - fit_dict[t] = {'fit ax + by + c': fit, 'residual': residual, 'rank' : rank, 's': s}
    : pos_keys: dictionary of strings to use as particle positions
    : frame: list of int giving the frames to compute. If frame==None, compute all frames
    : method: str, giving small modifications to procedure
               - `best_fit`: simply subtract the best fit plan
               - `max` : after best best fit, find the max deviation and subtract it from all the values

    : return None: modifies df in place to include computed out_key column
    """
    if pos_keys == None:
        pos_keys = {}
        pos_keys['x'] = 'x (um, imageStack)'
        pos_keys['y'] = 'y (um, imageStack)'
        pos_keys['z'] = 'z (um, imageStack)'

    if frame == None:
        frame = range(max(df.index.get_level_values('frame') + 1))
    elif type(frame) == int:
        frame = [frame]
    else: pass

    for t in frame:
        # extract fit paramters from fit dict at the right frame
        a,b,c = fit_dict[t]['fit ax + by + c']
        x = df.xs(t, level='frame')[pos_keys['x']].to_numpy()
        y = df.xs(t, level='frame')[pos_keys['y']].to_numpy()
        z = df.xs(t, level='frame')[pos_keys['z']].to_numpy()
        dist = z - (a*x + b*y +c)
        #pos.xs(t,level='frame')[out_key] = dist
        if method == 'best_fit': df.loc[t,out_key] = dist
        elif method =='max': df.loc[t,out_key] = dist - max(dist)
        else: raise ValueError('Unrognized option {} for method in distFromPlane'.format(method))
    return None

def gelStrain(df,h_offset, pos_keys=None, frame=None):
    """
    Computes the strain in the gel for each tracer particle.
    For particles that are not in the reference configuration, they are ignored.

    Parameters
    __________
    :df pandas dataFrame of gel particle positions with multi index (frame, particleID)
               Should be the output of loadData2Mem applied to gel hdf5 file
    :h_offset: float, height to offset (likely the imageStack locations) to get true gel height
               This is simply added to the positions currently, but something more sophisticated
               is required. Perhaps, reference height of the top most particle? Affects quantitative results
               but not qualitiative results. If input keys is rheo_sed_depth then this this parameter
               should be 0
    : pos_keys: dictionary mapping of position keys to use {'x': 'x (um, imageStack), ...} for example
    :frame: int or Nonoe. Which frame to run this analysis on. If frame==None, run all the available frames

    Return
    ______
    :gelStrain_df: dataFrame of trajectories with multiIndex (particle, values) with
                   values being quanitties like 'z (um, imageStack)', 'e_xz', 'z (um, below gel)'
                   and columns 'dt -1', 'dt 0', dt 89' giving relative displacement to ref config
                   'dt -1' is just reference configuration, no relative change
                   'dt 0' is identically zero...change in position of 0 to 0
                   'dt 76' is change in position of frame 76 relative to 0

    Usage
    ______
    select only those gelStrains that are above 20um
    >>> tmp = gelStrain_df.unstack()
    >>> tmp[tmp['dt -1']['z (um, imageStack)'] > 20].stack().dropna()
    """
    if pos_keys == None:
        pos_keys = {}
        pos_keys['x'] = 'x (um, imageStack)'
        pos_keys['y'] = 'y (um, imageStack)'
        pos_keys['z'] = 'z (um, imageStack)'
        pos_keys_list = list(pos_keys.values())

    if frame == None:
        frame = range(max(df.index.get_level_values('frame') + 1))
    elif type(frame) == int:
        frame = [frame]
    else: pass

    ref_config = df.loc[0,pos_keys_list]
    out = df.loc[0, pos_keys_list]
    out['e_xz'] = 0
    out['z (um, below gel)'] = df.loc[0,'z (um, below gel)']
    out = out.stack().rename('Ref Pos').to_frame()
    for t in frame:
        displacement = df.loc[t, pos_keys_list] - ref_config
        displacement['e_xz'] = displacement[pos_keys['x']]/(ref_config[pos_keys['z']] + h_offset)
        #displacement['e_yz'] = displacement[pos_keys['y']]/(ref_config[pos_keys['z']] + h_offset)
        #displacement['e_zz'] = displacement[pos_keys['z']]/(ref_config[pos_keys['z']] + h_offset)
        displacement['z (um, below gel)'] = df.loc[t, 'z (um, below gel)']
        displacement = displacement.stack().rename('(0,{})'.format(t))
        out = out.join(displacement)
    out.set_index(out.index.rename(['particle','value']),inplace=True)
    return out

def linearElastic(sedPos_df, G_sed=6.25, G_gel = 1.5, h_gel = 180, h_sed = 55):
    """
    returns a np array of strains of a linear elastic gel with thickness h_gel and modulus
    G_gel deforming with sediment of dimensions h_sed and modulus G_sed
    """
    posKey = ['x (um, imageStack)', 'y (um, imageStack)', 'z (um, imageStack)']
    sed_disp = sedPos_df[posKey] - sedPos_df[posKey].xs(0,level='frame')
    sed_disp['zbin']= pd.cut(sedPos_df['z (um, imageStack)'],60)
    d_top = sed_disp[sed_disp['zbin'] == pd.Interval(left=84.2, right=85.239)].dropna()['x (um, imageStack)'].unstack().mean(axis=1)
    strain = (d_top)*((G_sed)/(G_gel*h_sed + G_sed*h_gel))
    return strain

def plotSpatialGelStress(gelStrain_df, t,sedPos_df, pos_keys=None, type='mean', outSuffix=''):
    """
    Generates a plot of the local strain integrated upto time t
    I think I could add a z selection after the fact by selecting on the
    input gelStrain_df


    Follows the code example in
    https://stackoverflow.com/questions/41244322/how-to-color-voronoi-according-to-a-color-scale-and-the-area-of-each-cell

    """

    if pos_keys == None:
        pos_keys = {}
        pos_keys['x'] = 'x (um, imageStack)'
        pos_keys['y'] = 'y (um, imageStack)'
        pos_keys['z'] = 'z (um, imageStack)'
        pos_keys_list = list(pos_keys.values())

    # points are xy coordinates of all particles with complete trajectories
    points = gelStrain_df.dropna()['Ref Pos'].unstack()[[pos_keys['x'],pos_keys['y']]].to_numpy()
    strain = gelStrain_df.dropna()['(0,{})'.format(t)].unstack()['e_xz']

    # generate Voronoi tessellation
    vor = Voronoi(points)

    # now do some stuff to get only include cells within convex hull
    # https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells/34969162
    lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

    convex_hull = MultiPoint([Point(i) for i in points]).convex_hull.buffer(2)
    result = MultiPolygon([poly.intersection(convex_hull) for poly in polygonize(lines)])
    result = MultiPolygon([p for p in result] + [p for p in convex_hull.difference(unary_union(result))])

    # find min/max values for normalization
    #minima = min(strain)
    #maxima = max(strain)
    #gelStrain_df.rename_axis(index=['particle','values'],inplace=True)
    minStrain_list = gelStrain_df.xs('e_xz',level='values').apply(np.amin)
    minStrain = minStrain_list.min()
    maxStrain_list = gelStrain_df.xs('e_xz',level='values').apply(np.amax).max()
    maxStrain = maxStrain_list.max()
    avgStrain_list = gelStrain_df.xs('e_xz', level ='values').apply(np.mean)

    if type == 'mean':
        # set global bounds on color map
        # and leave strain unchanged
        minima = minStrain
        maxima = maxStrain
        statistics = strain.describe().to_dict()
        strain = strain.to_numpy()
    elif type == 'fluct':
        strain_fluct = gelStrain_df.xs('e_xz', level='values') - gelStrain_df.xs('e_xz', level='values').apply(np.mean)
        statistics = strain_fluct['(0,{})'.format(t)].describe().to_dict()
        minima = strain_fluct.apply(np.amin).min()
        maxima = strain_fluct.apply(np.amax).max()
        # strain is the array holding values to be plotted
        strain = strain_fluct['dt {}'.format(t)].to_numpy()
    elif type == 'linearElastic_fluct':
        strain_fluct = gelStrain_df.dropna().xs('e_xz',level='values').transpose().reset_index().drop(0).drop(columns='index').transpose().subtract(linearElastic(sedPos_df).transpose(),axis=1)
        #statistics = strain_fluct['(0,{})'.format(t)].describe().to_dict()
        statistics = strain_fluct[t].describe().to_dict()
        minima = strain_fluct.apply(np.amin).min()
        maxima = strain_fluct.apply(np.amax).max()
        # strain is the array holding values to be plotted
        strain = strain_fluct[t].to_numpy()

    # normalize chosen colormap
    # This should be normalized across **all** strains, not just the given time
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    if type == 'mean': mapper = cm.ScalarMappable(norm=norm, cmap=cm.YlGn)
    elif type == 'fluct': mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn)
    elif type == 'linearElastic_fluct': mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn)

    # plot Voronoi diagram, and fill finite regions with color mapped from speed value
    #voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_width=1)
    plt.plot()
    #for r in range(len(vor.point_region)):
    #    region = vor.regions[vor.point_region[r]]
    #    if not -1 in region:
    #        polygon = [vor.vertices[i] for i in region]
    #        # I dont understand what this is doing...*zip(*polygon(...? unpacking a zip?
    #        plt.fill(*zip(*polygon), color=mapper.to_rgba(strain[r]))
    for n in range(len(result)):
        r = result[n]
        plt.fill(*zip(*np.array(list(
            zip(r.boundary.coords.xy[0][:-1], r.boundary.coords.xy[1][:-1])))),
                 color=mapper.to_rgba(strain[n]))
    #plt.show()
    path ='/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/plots/gelStrain/'
    fName = 'strainHeatMap_{outSuffix}_{type}_t{frame:03}.png'.format(outSuffix=outSuffix, type=type,frame=t)
    statistics_fName = 'descriptive_statistics_{outSuffix}_{type}.yaml'.format(outSuffix=outSuffix,type=type)
    plt.savefig(path+fName,dpi=300)
    with open(path+statistics_fName, 'a') as f:
        yaml.dump({t: statistics},f)
    return "File saved to {}".format(path+fName)

@numba.jit(nopython=True,nogil=True,cache=False)
def computeLocalStrain(refPos, curPos, nnbArray):
    """
    Computes local strain for each particle in currentPos, relative to refPos
    following Falk and Langer.

    Parameters
    ----------
    refPos: numpy array of particle positions in the reference configuration.
            refPos[i] gives the position of ith particle [x,y,z]
    curPos: numpy array of particle positions in current cofiguration
    nnbArray: padded array of nnb indices
              eg [0,1,3,4,27,634,4,4,4,4,4,4]
              where 4 is the index of the central particle with coordinates
              refPos[4] and curPos[4]
    """
    # for every particle (or row in refPos)
    out = np.zeros((nnbArray.shape[0],10))
    for n in range(len(nnbArray)):
        nnbList = nnbArray[n]
        # get the coordinates of central particle
        # note that each elt of nnbArray is a padded array
        r0_ref = refPos[n]
        r0_cur = curPos[n]

        # initialize X and Y matrices (see Falk and Langer, PRE Eqn 2.11-2.14)
        X = np.zeros((3,3))
        Y = np.zeros((3,3))

        # now loop over the indices of nnb
        for m in range(len(nnbList)):
            # this is raising a bug when compiling with numba
            # https://github.com/numba/numba/issues/5680
            #if n == m: pass # this is just the center particle
            #if nnbList[m] == - 1: pass # this is padding
            #else:

            # get particle id
            pid = nnbList[m]

            # get the reference and current positions of the neighboring particle
            r_ref = refPos[pid]
            r_cur = curPos[pid]

            # compute X and Y matrices and add element wise result to stored matrix
            X += np.outer((r_cur - r0_cur),(r_ref - r0_ref))
            Y += np.outer((r_ref - r0_ref),(r_ref - r0_ref))

        # once X and Y have been calculated over the full nnb list, compute deformation tensor
        try: epsilon = X @ np.transpose(np.linalg.inv(Y)) - np.identity(3)
        # note, numba has **very** strict limitationon the excpetions, and you cannot pass numpy exceptions
        # but LinAlgError will match under the general Exception class matches
        except Exception:
            epsilon = np.zeros((3,3))
            epsilon[:] = np.nan

        # with deformation tensor, compute $D^2_{min}$, which caputes mean squared deviation
        # of the deomfration tensor and the actual deformation of the particles.

        # initialize to zero
        D2_min = 0.0
        # loop over all neareast neighbors like the previous for loop
        for m in range(len(nnbList)):
            #if n == 0: pass
            #if nnbList[n] == -1: pass
            #else:
            pid = nnbList[m]
            r_ref = refPos[pid]
            r_cur = curPos[pid]
            # Eqn 2.11 in F+L (except for rolling outer sum on nnb)
            D2_min += np.sum(
                ((r_cur - r0_cur) - (epsilon + np.identity(3) @ (r_ref - r0_ref)))**2)

        # get symmetric and skew symmetric parts of the matrix
        epsilon_sym = 0.5*(epsilon + np.transpose(epsilon))
        epsilon_skew = 0.5*(epsilon - np.transpose(epsilon))

        # flatten the array and select the components we care about
        sym_flat = np.array([epsilon_sym[0,0],
                             epsilon_sym[0,1],
                             epsilon_sym[0,2],
                             epsilon_sym[1,1],
                             epsilon_sym[1,2],
                             epsilon_sym[2,2]])
        skew_flat = np.array([epsilon_skew[0,1],
                              epsilon_skew[0,2],
                              epsilon_skew[1,2]])

        # add results to output array
        out[n,:] = np.concatenate((np.array([D2_min]), sym_flat, skew_flat))
    return out

def localStrain(pos_df, t0, tf, nnb_cutoff=2.2, pos_keys=None, verbose=False):
    """
    Wrapper function ofr computeLocalStrain to pair it with pandas dataFrames
    and return pandas data frames with the particle ids intact.
    """
    if pos_keys == None:
        pos_keys = {}
        pos_keys['x'] = 'x (um, imageStack)'
        pos_keys['y'] = 'y (um, imageStack)'
        pos_keys['z'] = 'z (um, imageStack)'
    pos_keys_list = list(pos_keys.values())

    # slice pos_df to get configuration at t0 and tf
    refConfig = pos_df.xs(t0, level='frame')[pos_keys_list]
    curConfig = pos_df.xs(tf, level='frame')[pos_keys_list]

    # filter to get only complete trajectories between time points
    # ensure arrays are sorted and equal length with identical particle ids
    idx = curConfig.index.intersection(refConfig.index)
    refConfig = refConfig.loc[idx].to_numpy()
    curConfig = curConfig.loc[idx].to_numpy()

    # generate search tree
    if verbose == True: print('generating search tree')
    refTree = cKDTree(refConfig)
    #curTree = cKDTree(curConfig)

    # query tree with all points to nnb upto first minima in rdf
    if verbose == True: print('querying tree for nnb')
    nnbIdx = refTree.query_ball_point(refConfig, nnb_cutoff)

    # let's keep track of the number of nnb for each particle id
    nnb_count = pd.Series(np.array([len(nnbList) for nnbList in nnbIdx]),index=idx,name='nnb count')

    # pad nnbIdx to array Nx17
    def padN(l,val,N=17): return np.pad(np.array(l),(0,N),mode='constant',constant_values=val)[0:N]
    # Caution, think about this next line of code
    #     -the index of the central particle may not be the first entry
    #     -I need to pad nnbIdx with index of the central particle to get around a bug numba
    #      https://github.com/numba/numba/issues/5680
    #      having to deal with if/else clauses in for looops
    #     -if local strain run on central particle, X and Y matrices are both zero and so no effect.
    nnbIdx_np = np.array([padN(nnbIdx[m],m) for m in range(len(nnbIdx))])

    if verbose == True: print('computing local strain')
    localStrainArray_np = computeLocalStrain(refConfig,curConfig,nnbIdx_np)
    localStrainArray_df = pd.DataFrame(localStrainArray_np,columns=['D2_min',
                                                                    'exx', 'exy', 'exz', 'eyy', 'eyz', 'ezz',
                                                                    'rxy', 'rxz','ryz'], index = idx).join(nnb_count)
    return localStrainArray_df

def makeLocalStrainTraj(pos_df, tPairList, output = 'strainTraj'):
    """
    Make LocalStrainTraj on a list of time points
    tPairs = list(zip([0 for n in range(90)],[n for n in range(2,90)]))

    This is a wrapper.  Mostly handles formatting the dataFrames.
    """
    strain_traj = localStrain(pos_df, 0, 1)
    strain_traj = strain_traj.stack().rename('(0,1)').to_frame()
    for n in range(len(tPairList)):
        tRef, tCur = tPairList[n]
        tmp = localStrain(pos_df, tRef, tCur)
        tmp = tmp.stack().rename('({},{})'.format(tRef, tCur)).to_frame()
        strain_traj = strain_traj.join(tmp)
        del tmp
    strain_traj.set_index(strain_traj.index.rename(['particle', 'values']), inplace=True)
    if output == 'hdf':
        raise KeyError('Saving strainTraj directly to hdf is not implemented yet')
        #strain_fName = 'tfrGel10212018A_shearRun10292018f_sed_strainTraj_consecutive.h5'
        #strain_traj.to_hdf(hdf_stem + strain_fName, '(0,t)', mode='a', format='table', data_columns=True)
    elif output == 'strainTraj':
        return strain_traj
    else: raise KeyError('output {} not recognized'.format(output))


if __name__ == '__main__':
    hdf_stem = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/locations_stitch/'
    hdf_fName = 'tfrGel10212018A_shearRun10292018f_sed_stitched.h5'
    pos = loadData2Mem(hdf_stem + hdf_fName)

    #%%
    pos_key_list = ['x (um, imageStack)', 'y (um, imageStack)', 'z (um, imageStack)']
    refConfig_micro = pos.xs(0,level='frame')[pos_key_list].head(10)
    curConfig_micro = pos.xs(1,level='frame')[pos_key_list].head(10)

    #%%
    idx = curConfig_micro.index.intersection(refConfig_micro.index)
    refConfig_micro_idx = refConfig_micro.loc[idx].to_numpy()
    curConfig_micro_idx = curConfig_micro.loc[idx].to_numpy()

    #%%
    refTree_micro = cKDTree(refConfig_micro_idx)
    nnbIdx = refTree_micro.query_ball_point(refConfig_micro_idx,3)

    strainOut = localStrain(pos,0,10)


