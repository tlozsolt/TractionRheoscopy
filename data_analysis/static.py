import pandas as pd
import trackpy as tp
import numba
from scipy.spatial import cKDTree
import numpy as np
import os

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from deprecated import deprecated
from multiprocessing import Pool
#from shapely.ops import polygonize, unary_union
#from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point

import yaml

def loadExperiment(hdfStem=None,
                   fNameDict = None):
    """
    hdfStem, str: path to hdf pandas archives
    fNameList list of str, fileNames of hdf paths to open

    loads the hdf files and returns a dictionary of pandas files with keywords
    that can be used as variable names if you want

    if None, initiliazed to the experiment tfrGel10212018A_shearRun10292018f
    """
    if hdfStem is None:
        hdfStem = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/locations_stitch/'
    if fNameDict is None:
        fNameDict = {'sedPos': 'tfrGel10212018A_shearRun10292018f_sed_stitched_pandas.h5',
                     'sedStrain_traj': 'tfrGel10212018A_shearRun10292018f_sed_strainTrajZeroRef.h5',
                     'gelPos': 'tfrGel10212018A_shearRun10292018f_gel_stitched_pandas.h5',
                     'eigen' : 'tfrGel10212018A_shearRun10292018f_{}.h5'.format('eigenStrain_decomposition'),
                     'sedStrain_rot' : 'tfrGel10212018A_shearRun10292018f_{}.h5'.format('sedStrain_paraX_perpY')
                     }
    out = {}
    for key,fName in fNameDict.items():
        out[key] = pd.read_hdf(hdfStem + fName)
    return out

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

def fitTopSurface(df, frame=None, pos_keys=None, n_bin = 15, method='maxProject'):
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
        if method != 'ovito':
            tmp['xbin'] = pd.cut(tmp[pos_key['x']], n_bin)
            tmp['ybin'] = pd.cut(tmp[pos_key['y']], n_bin)
            # Comment, I think this gives an array x and y bin centers of the particles in the gel that have the highest z
            # coordinate. The plane is fit to the x and y bin centers of the particle with the largest z value in the bin...
            # I think its much better to get a permanent index of particles, slice into pos_df to get locations, and then fit
            # the plane to xyz coordinates of those particle.
            # the function should also be agnostic to what plane or material you are feeding. just pass an index and
            # position dataframe and return a plane through those points. particle sslection is implicit in the index passed
            # zsolt, Aug 2021
            x = tmp.groupby(['xbin', 'ybin']).max()['z (um, imageStack)'].reset_index()['xbin'].apply(lambda x: 0.5 * (x.left + x.right)).to_numpy()
            y = tmp.groupby(['xbin', 'ybin']).max()['z (um, imageStack)'].reset_index()['ybin'].apply(lambda x: 0.5 * (x.left + x.right)).to_numpy()
            z = tmp.groupby(['xbin', 'ybin']).max()['z (um, imageStack)'].reset_index()['z (um, imageStack)'].to_numpy()
            A = np.vstack([x,y,np.ones(len(x))]).T
        elif method == 'ovito':
            print('Try using fitSurface()')
            raise NotImplemented
        try:
            fit, residual, rank, s = np.linalg.lstsq(A, z, rcond=None)
            out[t] = {'fit ax + by + c': fit, 'residual': residual, 'rank' : rank, 's': s}
        except np.linalg.LinAlgError:
            out[t] = 'fit did not converge with nbin = {} is likely too small'.format(n_bin)
            pass
    return out

def fitSurface_singleTime(pos_df, idx, coordStr):
    """For a given position dataframe, fit a plance through the particles specified by idx
    pos_df: index should match idx (not a multiindex)
    idx: index of particles, whose positions should be fit to a plane
    """
    tmp = pos_df.loc[idx]
    tmp['ones'] = 1
    A_keys = ['x {}'.format(coordStr), 'y {}'.format(coordStr), 'ones']
    fit, residual, rank, s = np.linalg.lstsq(tmp[A_keys].to_numpy(), tmp['z {}'.format(coordStr)].to_numpy())
    return {'fit function': 'ax + by + c',
            'coordinate string': coordStr,
            'number of points': idx.shape[0],
            'a':fit[0], 'b': fit[1], 'c':fit[2],
            'residual': residual.squeeze(),
            'rank': rank, 's': s}

def fitSurface(pos_df, idx, coordStr = '(um, imageStack)'):
    """loop over all the frames, fitting each surface"""
    frames = max(pos_df.index.get_level_values('frame'))
    out = {}
    for t in range(frames + 1):
        tmp = pos_df.xs(t,level='frame')
        out[t] = fitSurface_singleTime(tmp,idx.intersection(tmp.index),coordStr)
    return pd.DataFrame(out).T

def readOvitoIdx(path):
    'parses an xyz file with a single column of particle ids and returns a pandas index object'
    if not os.path.exists(path):
        print('No ovito idx file found at path {}. Returning None'.format(path))
        return None
    else:
        tmp = pd.read_csv(path, sep='\n').drop(0).set_axis(['particle id'], axis=1)
        return pd.Index(np.array(tmp['particle id']).astype(np.int))

@deprecated(version='pre-Pandas', reason='This function assumes a dictionary of fit parameters as opposed to DataFrame.\n '
                                          'Use pt2Plane and distFromPlane_df instead')
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

def pt2Plane(pt_zyx, plane_abc):
    """
    returns the distance of pt_zyx to plane defined by f(z) = ax + by + c
    Note, this is not the shortest distance to the plane, but rather the signed distance along z
    """
    try: z, y, x = pt_zyx[:, 0], pt_zyx[:, 1], pt_zyx[:, 2]
    except IndexError: z,y,x = pt_zyx
    a, b, c = plane_abc
    return z - (a * x + b * y + c)

def distFromPlane_df(pos_df, fit_df, idStr, coordStr = '(um, imageStack)'):
    """
    >>> gel_pos = gel_pos.join(da.distFromPlane_df(gel_pos, gel_fits,'dist top gel'))
    """
    #check that coordinate systems match
    if coordStr != fit_df['coordinate string'].iloc[0]: raise ValueError('Mismatching coordinate systems!')
    n_frames = fit_df.index.shape[0]
    out = []
    # loop over time, maybe make parallel?
    for t in range(n_frames):

        cols = ['{} {}'.format(x,coordStr) for x in ['z','y','x']]
        # slice the multi-index on pos_df, and keep track of the multiindex
        tmp_df = pos_df.loc[(t, slice(None)), :][cols]
        mIdx = tmp_df.index

        # select the right plane to fit
        fit_tmp = fit_df.loc[t][['a','b','c']]

        # compute the distance
        _dist = pt2Plane(tmp_df.values, fit_tmp.values)

        # reassemble the dataFrame from arrays so that you can join the output with pos_df
        dist_df = pd.DataFrame({idStr:_dist}, index=mIdx)
        out.append(dist_df)

    # concatenate and return
    return pd.concat(out)

def gelStrain(df,h_offset, R = None, pos_keys=None, frame=None):
    """
    Computes the strain in the gel for each tracer particle.
    For particles that are not in the reference configuration, they are ignored.

    Parameters
    __________
    :df pandas dataFrame of gel particle positions with multi index (frame, particleID)
               Should be the output of loadData2Mem applied to gel hdf5 file
    :h_offset: float, height to offset (likely the imageStack locations) to get true gel height
               For imageStack coordinate system, this the height above the coverslip for a hypthetical particle
               at the bottom of the imaging stack (ie z=0)
               This is simply added to the positions currently, but something more sophisticated
               is required. Perhaps, reference height of the top most particle? Affects quantitative results
               but not qualitiative results. If input keys is rheo_sed_depth then this this parameter
               should be 0
    : R : np arrray, 2x2 rotation matrix that converts image coordinates to shear coordinates by removing the small
               misalignment between the shear direction and the x direction.
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
    if pos_keys is None:
        pos_keys = {}
        pos_keys['x'] = 'x (um, imageStack)'
        pos_keys['y'] = 'y (um, imageStack)'
        pos_keys['z'] = 'z (um, imageStack)'

    pos_keys_list = list(pos_keys.values())

    if frame is None:
        frame = range(max(df.index.get_level_values('frame') + 1))
    elif type(frame) == int:
        frame = [frame]
    else: pass

    ref_config = df.loc[0,pos_keys_list]
    out = df.loc[0, pos_keys_list]
    out['e_xz'] = 0
    out['e_yz'] = 0
    out['e_zz'] = 0
    if R is not None:
        out['x (um, shearCoord para)'] = 0
        out['y (um, shearCoord perp)'] = 0
        out['e_para'] = 0
        out['e_perp'] = 0
    #try: out['z (um, below gel)'] = df.loc[0,'z (um, below gel)']
    #except KeyError('key \'(z (um, below gel)\' was not found'): pass
    out = out.stack().rename('Ref Pos').to_frame()

    def rotate(rotMat, p, paraOrPerp='para'):
        x, y = rotMat.dot(p)
        if paraOrPerp == 'para': return x
        elif paraOrPerp == 'perp': return y
        else: return x, y

    for t in frame:
        displacement = df.loc[t, pos_keys_list] - ref_config
        displacement['e_xz'] = displacement[pos_keys['x']]/(ref_config[pos_keys['z']] + h_offset)
        displacement['e_yz'] = displacement[pos_keys['y']]/(ref_config[pos_keys['z']] + h_offset)
        displacement['e_zz'] = displacement[pos_keys['z']]/(ref_config[pos_keys['z']] + h_offset)
        #displacement['z (um, below gel)'] = df.loc[t, 'z (um, below gel)']

        if R is not None:
            # rotation matrix was given, so rotate and name perp and para
            displacement['x (um, shearCoord para)'] = displacement.apply(lambda p: rotate(R,
                                                                                            (p[pos_keys['x']],
                                                                                             p[pos_keys['y']]),
                                                                                            paraOrPerp='para'),
                                                                         axis=1 )
            displacement['y (um, shearCoord perp)'] = displacement.apply(lambda p: rotate(R,
                                                                                        (p[pos_keys['x']],
                                                                                         p[pos_keys['y']]),
                                                                                        paraOrPerp='perp'),
                                                                         axis=1 )
            displacement['e_para'] = displacement['x (um, shearCoord para)']/(ref_config[pos_keys['z']] + h_offset)
            displacement['e_perp'] = displacement['y (um, shearCoord perp)']/(ref_config[pos_keys['z']] + h_offset)

        #displacement = displacement.stack().rename('(0,{})'.format(t))
        # modified to make parsing index easier, Aug 2021
        # use '0 10'.split() -> ['0','10'] -> list comprehension applying int
        displacement = displacement.stack().rename('0 {}'.format(t))
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

@numba.jit(nopython=True, nogil=True, cache=False)
def computeRotation(pos,R):
    """
    Rotates the position in pos array using the rotation matrix R

    Parameters
    __________

    pos: numpy array of particle positions
         pos[i] gives the [x,y,z] position of the particles
    R: numpy matrix, typically 2D

    returns: numpy array of same dimension of pos, with with rotated positions.
    """
    out = np.zeros(pos.shape)
    for n in range(len(pos)):
        x,y = pos[n][0], pos[n][1]
        out[n] = R.dot((x,y))
    return out

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
    out = np.zeros((nnbArray.shape[0],11))
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

        # compute von Mises strain
        vM = np.sqrt(1/2*(  (epsilon_sym[0,0] - epsilon_sym[1,1])**2
                          + (epsilon_sym[1,1] - epsilon_sym[2,2])**2
                          + (epsilon_sym[2,2] - epsilon_sym[1,1])**2)
                     + 3*(epsilon_sym[0,1]**2 + epsilon_sym[1,2]**2 + epsilon_sym[0,2]**2))

        # add results to output array
        out[n,:] = np.concatenate((np.array([D2_min, vM]), sym_flat, skew_flat))
    return out

def localStrain(pos_df, t0, tf, nnb_cutoff=2.2, pos_keys=None, verbose=False):
    """
    Wrapper function ofr computeLocalStrain to pair it with pandas dataFrames
    and return pandas data frames with the particle ids intact.

    This will compute the strain for all complete trajectoris between t0 and tf
        (I think this means that the strain for a given particle may change quite a bit if is looses one of
        its neighbors due to poor tracking)
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
    max_nnb = nnb_count.max()

    def padN(l,val,N): return np.pad(np.array(l),(0,N),mode='constant',constant_values=val)[0:N]
    # Caution, think about this next line of code
    #     -the index of the central particle may not be the first entry
    #     -but the index of the particle **is** the row number in nnb_idx
    #     -I need to pad nnbIdx with index of the central particle to get around a bug numba
    #      https://github.com/numba/numba/issues/5680
    #      having to deal with if/else clauses in for looops
    #     -if local strain run on central particle, X and Y matrices are both zero and so no effect.
    nnbIdx_np = np.array([padN(nnbIdx[m],m, N=max_nnb+1) for m in range(len(nnbIdx))])


    if verbose == True: print('computing local strain')
    localStrainArray_np = computeLocalStrain(refConfig,curConfig,nnbIdx_np)
    localStrainArray_df = pd.DataFrame(localStrainArray_np,columns=['D2_min',
                                                                    'exx', 'exy', 'exz', 'eyy', 'eyz', 'ezz',
                                                                    'rxy', 'rxz','ryz'], index = idx).join(nnb_count)
    return localStrainArray_df

@deprecated(version='pre frame particle convention', reason='This function is not parallel. Use localStrain_fp instead')
def makeLocalStrainTraj(pos_df, tPairList, nnbCutoff, output = 'strainTraj',pos_keys=None,verbose=False):
    """
    Make LocalStrainTraj on a list of time points
    tPairs = list(zip([0 for n in range(90)],[n for n in range(2,90)]))
    >> tPairs = [(n-3, n) for n in range(3,22)]
    or
    >> tPairs = [(0,n) for n in range(22)]

    This is a wrapper.  Mostly handles formatting the dataFrames.

    It would be great if this was parallel with multiprocessing to deal with each of the (ref,cur)
    pairs in parallel. I feel like I may have already done this on some random jupyter nb? zsolt, Aug 2021
    >> see da.static.localStrain_fp
    """
    if verbose: print("Starting {}, entry {} of {} strains ".format(tPairList[0], 0, len(tPairList)))

    # start with the first elt in tPairList and set up a permanent data structure
    t0_ref, t0_cur = tPairList[0]
    strain_traj = localStrain(pos_df, t0_ref, t0_cur, nnb_cutoff=nnbCutoff, pos_keys = pos_keys)
    strain_traj = strain_traj.stack().rename('({},{})'.format(t0_ref, t0_cur)).to_frame()

    # proceed to the rest of the for loop
    for n in range(1,len(tPairList)):
        tRef, tCur = tPairList[n]
        if verbose: print("Starting {}, entry {} of {} strains ".format(tPairList[n],n, len(tPairList)))
        tmp = localStrain(pos_df, tRef, tCur, nnb_cutoff=nnbCutoff, pos_keys=pos_keys)
        tmp = tmp.stack().rename('({},{})'.format(tRef, tCur)).to_frame()
        strain_traj = strain_traj.join(tmp)
        del tmp
    # some naming of the indices
    strain_traj.set_index(strain_traj.index.rename(['particle', 'values']), inplace=True)

    # what output do you want?
    if output == 'hdf':
        raise KeyError('Saving strainTraj directly to hdf is not implemented yet')
        #strain_fName = 'tfrGel10212018A_shearRun10292018f_sed_strainTraj_consecutive.h5'
        #strain_traj.to_hdf(hdf_stem + strain_fName, '(0,t)', mode='a', format='table', data_columns=True)
    elif output == 'strainTraj': return strain_traj
    elif output == 'frameParticle':
        if verbose: print('converting strainTraj to frameParticle')
        return traj2frameParticle(strain_traj)
    else: raise KeyError('output {} not recognized'.format(output))

def localStrain_fp(pos_df, tPair, nnbCutoff):
    """
    tPair = (0,1) for example

    combine with multiprocessing in jupyter notebook like:

    >> def g(tPair): return da.localStrain_fp(sed_pos, tPair, 2.6)
    >> from multiprocessing import Pool
    >> with Pool(2) as p: out = p.map(g,[(0,1),(3,4)])
    >> sed_strain_df = pd.concat(out)
    """
    ref, cur = tPair
    tmp = localStrain(pos_df, ref, cur, nnb_cutoff=nnbCutoff)
    mIdx = pd.MultiIndex.from_product([[tPair], tmp.index], names=['frame', 'particle'])
    return tmp.set_index(mIdx)

def loadParticle(t, path_partial=None, fName_frmt=None):
    if path_partial is None:
        path_partial = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/locations_stitch/partial_data_Aidan/'
    if fName_frmt is None:
        fName_frmt = 'shearRun10292018f_centerParticles_t{:02}.h5'

    pos_t = pd.read_hdf(path_partial + fName_frmt.format(t))
    idx_t = pd.MultiIndex.from_product([[t], pos_t.index], names=['frame', 'particle'])

    return pos_t.set_index(idx_t)

def df2xyz(df, fPath,fName, mode='w'):
    """
    Write a pandas dataFrame to xyz file
    """
    fPath_frmt = fPath+'{}'
    with open(fPath_frmt.format(fName),mode) as f:
        f.write(str(df.shape[0]))
        f.write('\n ')
        df.to_csv(f,mode=mode, sep=' ', na_rep='NAN')
    return fPath_frmt.format(fName)

def traj2frameParticle(sedStrain_traj):
    """
    Coverts dataFrame of sedStrain_traj with multiIndex (particle,value) and columns of time '(0,3)' to
    dataFrame with multiIndex (frame, particle) and columns of value: ('(0,3)', 5634)
    is particle 5634 on time interval '(0,3)'
    """
    sedStrain_tmp = sedStrain_traj.transpose().stack('particle')
    idx = sedStrain_tmp.index.set_names(['frame', 'particle'])
    sedStrain_frameParticle = pd.DataFrame(sedStrain_tmp, index=idx)
    return sedStrain_frameParticle

def strainDiag(strain_fp, signature=None):
    """
    compute dataFrame of eigen vectors and eigen values from strain dataFrame in frameParticle format

    This function is slow as hell. I am not sure why. Perhaps it takes a long to to diagonalize?
    Perhaps the list comprehension is not the right way to go and instead I should loop
    like rotation matrix code using jit.

    It would be ideal if this function, especially the wrapper part was rewritten with multiprocessing
    and write to file, applied automatically across time indices.

    """
    if signature is None:
        _keys = ['exx','exy','exz','eyy','eyz','ezz']
        _sig = np.array([(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)])
        _join = ['nnb count', 'D2_min']
        _out = ['u','ux','uy','uz','v','vx','vy','vz','w','wx','wy','wz']
    else:
        _keys = signature['keys']
        _sig = signature['sig']
        _join = signature['join']

    def _strainDiag(strain_1d):
        exx,exy,exz,eyy,eyz,ezz = strain_1d
        e = np.array([[exx, exy, exz],
                      [exy, eyy, eyz],
                      [exz, eyz, ezz]])
        eigen_val, eigen_vec = np.linalg.eig(e)
        # now sort by eigen value
        idx = eigen_val.argsort()[::-1]
        # return sorted val and vec
        tmp = eigen_val[idx],eigen_vec[:,idx]
        return np.array([np.concatenate((tmp[0][n],tmp[1][n]),axis=None) for n in range(3)]).flatten()
    eigen = [_strainDiag(elt) for elt in strain_fp[_keys].to_numpy()]
    m_idx = strain_fp.index
    eigen_df = pd.DataFrame(np.array(eigen), index=m_idx, columns=_out)
    return eigen_df

def getLocatingStats(particle_idx, frame, tracked_df = None, id_type='index' ):
    """
    Given a certain particle_idx (or list of IDs) in the tracked output of trackpy and a frame number,
    return all the locating statistics.

    Untested as of Jul 6, 2021
    -zsolt
    """
    #
    if tracked_df is None and id_type != 'index':
        raise KeyError('if id_type is not index, you must provide the tracked_df output from trackpy')

    # load the stitched dataframe
    mat = 'sed'
    path = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/locations'
    fName_frmt = 'tfrGel10212018A_shearRun10292018f_stitched_{}'.format(mat)+'_t{:03}.h5'
    fName = path + '/' + fName_frmt.format(frame)
    stitched = pd.read_hdf(fName, key='{}'.format(frame))

    # index into with particle_idx provided
    if id_type == 'index': return stitched[stitched['keepBool' == True]].loc[particle_idx]
    elif id_type == 'particle_id':
        _idx = tracked_df.loc[tracked_df['particle'].isin(particle_idx)].index
        return stitched[stitched['keepBool' == True]].loc[_idx]
    else: raise KeyError("id_type was {}, but must be either \'index\' or \'particle_id\' ".format(id_type))

def insersectDict(parent: dict, child: dict) -> dict:
    """
    Return  sub dictionary of parent with keys common to child
    Useful for coercing dtype is parent is master list of dtypes read form
    param file and child is dictionary of dtypes from pandas dataframe

    example
    parent = {'a':'int64', 'b':'float64', 'c':'bool', 'e': 'float64}
    child = {'a': 'object', 'b':'float64', 'c':'bool', 'd': 'float64'}
    output ->  {'a': 'int64', 'b': 'float64', 'c': 'bool'}
    """
    return {x: parent[x] for x in set(parent.keys()).intersection(child.keys())}

def loadStitched(time_list,
                 path = None,
                 fName_frmt=None,
                 posKeys= None,
                 colKeys = None,
                 dtype_dict = None):
    """
    This returns a generator (note the use of yield) and so should be called inside a for loop
    >> for data in loadStitched([1,2,3], 'gel'): print(data)
    """

    #if path is None: path = '/Users/zsolt/Colloid/DATA/tfrGel09102018b/shearRun09232018a/locations'
    #if fName_frmt is None: fName_frmt = 'tfrGel09102018b_shearRun09232018a_stitched_{}'.format(mat)+'_t{:03}.h5'
    if path is None: path = input("input path ot locations (eg \'/Users/zsolt/Colloid/DATA/tfrGel09102018b/shearRun09232018a/locations\'")
    if fName_frmt is None: fName_frmt = input("what is the fName_frmt (eg \''/tfrGel09102018b_shearRun09232018a_stitched_{}'.format(mat)+'_t{:03}.h5'\'")

    # TODO: There is bug here for loading gel columns as when I located gel, I didnt separate pxClassifer channels into
    #       core/shell suffixes -> use different posKeys or print a warning
    #       in retrospcet its probably best to specific which cols to drop (if any) as opposed to specifying which to keep.
    if posKeys is None:
        posKeys = ['{} (um, imageStack)'.format(x) for x in ['x','y','z']]
        posKeys += ['{}_std'.format(x) for x in ['x','y','z']]
        posKeys += ['size_{}'.format(x) for x in ['x','y','z']]
        posKeys += ['totalError', 'disc_size', 'mass', 'raw_mass', 'signal', 'background']
        posKeys += [ 'sed_Colloid_core', 'sed_Colloid_shell',
                     'fluorescent_chunk_core', 'fluorescent_chunk_shell',
                     'gel_Tracer_core', 'gel_Tracer_shell',
                     'sed_Background_core', 'sed_Background_shell',
                     'nonfluorescent_chunk_core', 'nonfluorescent_chunk_shell',
                     'gel_Background_core', 'gel_Background_shell']

    if colKeys is None: colKeys = ['frame']
    #else: posKeys, colKeys = params['posKeys'], params['col_keys']

    for t in time_list:
        #print('Stitched time {}'.format(t))
        fName = path + fName_frmt.format(t)
        data = pd.read_hdf(fName,key='{}'.format(t))
        # what if we just dont specfiy columns at all? Will it just give me all the columns?
        # YES...this combined with on disk queries gives new best practice:
        # loadStitched all columns and then specify index and columns during query in jupyter nb
        #yield data[data['keepBool'] == True]

        # add option to coerce datatypes
        if dtype_dict is not None:
            data = data.astype(dtype_dict)

        # but some of the columns are empty and all nan so...
        yield data[data['keepBool'] == True][posKeys + colKeys]

def stitched_h5(stitched_outPath, max_t, param_dict):
    path = param_dict['path']
    mat = param_dict['mat']
    fName_frmt = param_dict['fName_frmt']

    with tp.PandasHDFStoreBig(stitched_outPath) as s:
        for data in loadStitched(range(max_t), mat, path=path, fName_frmt=fName_frmt):
            s.put(data)

def getCompleteIdx(df1,df2, key='particle'):
    """
    Get particle indices shared between dataframes df1 and df2
    If df1 and df2 are the initial and final time points, then this will return
    the particle id (from tracking) of particles with complete trajectories.
    """
    return df1.index.intersection(df2.index)
    #return pd.Index(df1[key]).intersection(pd.Index(df2[key]))

def getDroppedIdx(df1,df2, key='particle'):
    """
    Get the indices of df1 that are not present in df2.
    Note this function is not symmetric under permutation of df1 and df2
    If df1 and df2 are the initial and final time points, then this will return
    the particle id of particle in the initial time that have incomplete trajectories.
    """
    idx1 = pd.Index(df1[key])
    idx2 = pd.Index(df2[key])
    return idx1.difference(idx2)

def xyzDropped(df1,df2, key='particle'):
    """
    Write xyz files for the particle with complete and incomplete traj between df1 and df2.
    """
    path = input("Type path where the xyz files should be saved")
    fName_frmt = input("fName_frmt eg \'/sed_{flag}_t{start:02}_t{stop:02}.xyz\'")
    comIdx = getCompleteIdx(df1,df2)
    dropIdx = getDroppedIdx(df1,df2)
    start, stop = df1['frame'].iloc[0], df2['frame'].iloc[0]
    df2xyz(df1.set_index(key).loc[comIdx],path, fName_frmt.format(flag='complete',start=start, stop=stop))
    df2xyz(df1.set_index(key).loc[dropIdx],path, fName_frmt.format(flag='incomplete',start=start, stop=stop))
    return path+'/{}'.format(fName_frmt)

def sliceFP(pos_df, idx):
    """
    slice a multiIndex position DataFrame (pos_df with multiIndex (frame, particle) to get all colummns (col) and particle indices idx
    for all the time time points in the multiIndex

    this effectively returns the trajectories of the particles in idx over time.

    wow, this a huge timesaver but there may be an even faster way with pos_df.xs(t,level='frames).loc[idx] and
    looping over frame number t

    returns a DataFrame of
    """
    col = pos_df.keys()
    out = {}
    # unstacking is the slowest part, keep it outside the for loop
    unstacked = pos_df.unstack()
    for name in col:
        out[name] = unstacked[name].T.loc[idx].T.stack()
    return pd.DataFrame(out)

def sliceTraj(pos_df, idx):
    """
    for a single column, return a trajectory format of all particles in idx
    """
    col = pos_df.keys()
    out = {}
    # unstacking is the slowest part, keep it outside the for loop
    unstacked = pos_df.unstack()
    for name in col:
        out[name] = unstacked[name].T.loc[idx]
    return out

def computeDisplacement(pos_df, pos_keys=None, coordStr='(um, imageStack)'):
    """
    compute the dipslacment of every particle relative to the first frame
    will fill in nan for particles that are either not in the first frame
    or after they are dropped.

    """
    # moving window, v2
    # something with pad and roll to compute displacement in moving window
    # >> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    xyz=['x','y','z']
    if pos_keys is None: pos_keys = ['{} {}'.format(coord,coordStr) for coord in xyz]
    unstacked = pos_df[pos_keys].unstack()

    out = {}
    #for coord in xyz:
    for key,pos in zip(['disp {}'.format(coord) for coord in xyz], pos_keys):
        #key,pos = ('disp {}'.format(coord), '{} {}'.format(coord,coordStr))
        out[key] = (unstacked[pos] - unstacked[pos].loc[0]).stack()
    return pd.DataFrame(out)

def heatMap(disp_df, out_frmt = None, interactive = False):
    """ Make a heatmap in seaborn of displacement that have been binned in xy
    # >> https://matthewmcgonagle.github.io/blog/2019/01/22/HeatmapBins
    """

    # bin in x,y
    disp_df['xbin'] = pd.cut(disp_df['x (um, imageStack)'], pd.interval_range(0, 235, 10))
    disp_df['ybin'] = pd.cut(disp_df['y (um, imageStack)'], pd.interval_range(0, 235, 10))

    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set(rc={'figure.figsize': (16, 9)})
    sns.set_context('paper', font_scale=4)

    def getMid(pair):
        """ usage: mIdx_xybins.map(getMid) returns a multiindex of bin centers. """
        def _getMid(interval): return (interval.right + interval.left) / 2
        x, y = pair
        return _getMid(x), _getMid(y)

    binned_displacement = disp_df.groupby(['xbin', 'ybin']).mean()[
        ['disp {}'.format(coord) for coord in ['x', 'y', 'z']]]
    binned_displacement['disp sqrt(x**2 + y**2)'] = np.sqrt(
        binned_displacement['disp x'] ** 2 + binned_displacement['disp y'] ** 2)
    binned_mIdx = binned_displacement.index.map(getMid)
    # plot the displacement in x and y in nanometers
    # this should be separated into separate function.
    v = {'x': (0, 330), 'y': (0, 330), 'sqrt(x**2 + y**2)': (0, 330)}
    for i, coord in enumerate(['x', 'y', 'sqrt(x**2 + y**2)']):
        plt.clf()
        tmp = binned_displacement['disp {}'.format(coord)].reindex(binned_mIdx).unstack().T
        plt.figure(i)
        if interactive: sns.heatmap(abs(1000 * tmp), cmap='viridis')
        else:
            sns.heatmap(abs(1000 * tmp), vmin=v[coord][0], vmax=v[coord][1], cmap='viridis')
            if out_frmt is None: out = input("Full path (without quotes) to output figure for coordinate {}: ".format(coord))
            else:
                if coord is 'x' or coord is 'y': out = out_frmt.format(coord=coord)
                else: out = out_frmt.format(coord='mag')
            plt.savefig(out, bbox_inches='tight')
    return True

def plot_heatMap(disp_dict, **kwargs):
    # from dict of disp df write the following figures:
    # - displacement of x,y,z and magnitude for sed and gel
    # -reisudal displacment of xyz between sed and gel on diverging cmap
    # have the following parameters passed through a file:
    #  - cmap along with bounds
    #  - number of spatial bins
    #  - figure context parameters
    #  - output paths and naming conventions
    #  - displacement scaling to nm along with figure labels (maybe this is over generlizing)
    # Note, this function should serve as the template for how to write plotting functions that
    # need to be applied across experiments.

    # define some sub functions that will be used
    #def getMid(pair):
    #    """ usage: mIdx_xybins.map(getMid) returns a multiindex of bin centers. """
    #    def _getMid(interval): return (interval.right + interval.left) / 2
    #    x, y = pair
    #    return _getMid(x), _getMid(y)
    def bindf(disp_df, key, start, stop, n): return pd.cut(disp_df[key], pd.interval_range(start, stop, n))

    def writeHeatmap(binned_disp, mat, out_frmt, vmin=None, vmax=None, cmap=None, center=None, **param):
        #v = {'x': (0, 330), 'y': (0, 330), 'sqrt(x**2 + y**2)': (0, 330)}
        binned_mIdx = binned_disp.index.map(getMid)
        for i, coord in enumerate(['x', 'y', 'mag']):
            plt.clf()
            tmp = binned_disp['disp {}'.format(coord)].reindex(binned_mIdx).unstack().T
            plt.figure(i)
            if mat is 'sed' or mat is 'gel': sns.heatmap(abs(1000 * tmp), vmin=vmin[coord], vmax=vmax[coord], cmap=cmap, center=center)
            else: sns.heatmap(1000 * tmp, vmin=vmin[coord], vmax=vmax[coord], cmap=cmap, center=center)
            if coord is 'x' or coord is 'y': out = out_frmt.format(coord=coord)
            else: out = out_frmt.format(coord='mag')
            plt.savefig(out, bbox_inches='tight')
        return True

    # unpack some of the metaData (dont know how to do this from a file)
    # for the moment, I will just create a hard coded dictionary that can be unpacked from yaml
    start,stop,n = (0,235,10)
    plot_param = {
        'sed': {'cmap': 'viridis', 'vmin':{'x':0, 'y':0, 'mag': 0}, 'vmax': {'x': 330, 'y':330, 'mag': 330}},
        'gel': {'cmap': 'viridis', 'vmin': {'x': 0, 'y': 0, 'mag': 0}, 'vmax': {'x': 330, 'y': 330, 'mag': 330}},
        'residual': {'cmap': 'vlag', 'center': 0, 'vmin':{'x':-50, 'y':-50, 'mag':-50}, 'vmax':{'x':50, 'y':-50, 'mag':50} },
        'count': {'discrete_palette': "Blues", 'title': "Number of particles per bin", 'sns_heatmap': {'vmin': 0, 'vmax':16, 'annot':True}}
    }
    figPath = './debug'
    sns_context = {'rc':{'figure.figsize':(16,12)}, 'font_scale': 4, 'style':'paper'}

    #set sns context, could also be loaded from file?
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set(rc=sns_context['rc'])
    sns.set_context(sns_context['style'], font_scale=sns_context['font_scale'])

    # now bin in x and y, followed by groupby t,xy
    binned_dict = {}
    for mat, disp_df in disp_dict.items():
        #compute magnitude
        disp_df['disp mag'] = np.sqrt( disp_df['disp x'] ** 2 + disp_df['disp y'] ** 2)
        for coord in ['x', 'y']:
            disp_df['{}bin'.format(coord)] = bindf(disp_df, '{} (um, imageStack)'.format(coord), start, stop, n)
        binned_dict[mat] = disp_df.groupby(['frame', 'xbin', 'ybin']).mean()[['disp {}'.format(coord) for coord in ['x', 'y', 'z', 'mag']]]

    #compute residual displacement across bins
    binned_dict['residual'] = binned_dict['sed'] - binned_dict['gel']

    #"""
    for mat,binned_disp in binned_dict.items():
        for frame, disp_t in binned_disp.groupby('frame'):
           out_frmt = figPath + '/'+'_'.join([mat,'{coord}','t'+f'{frame:03}'+'.png'])
           writeHeatmap(disp_t.droplevel('frame'),mat,out_frmt, **plot_param[mat])
    #"""

    # write single entry of count heatmap for gel
    plt.clf()
    _p = plot_param['count']
    c, vmax =_p['discrete_palette'], _p['sns_heatmap']['vmax']
    #vmin, vmax, annot = _p['vmin'], _p['vmax'], _p['annot']
    tmp = disp_dict['gel'].xs(0, level='frame')
    tmp_mIdx = tmp.groupby(['xbin', 'ybin']).count().index.map(getMid)

    #plot
    g = sns.heatmap(tmp.groupby(['xbin','ybin']).count().reindex(tmp_mIdx)['disp x'].unstack().T,
                cmap=sns.mpl_palette(c, vmax),  **_p['sns_heatmap'])
    #g.set_title(_p['title'])
    # save
    count_out = figPath + '/gel_particleCount.png'
    plt.savefig(count_out, bbox_inches = 'tight')

def plot_zbinDisp(disp_dict, **kwargs):
    # TODO:
    # -make params passed to fucntion
    # -load sns and set context
    # -change to iterating over groupby dict?
    sed_clean = disp_dict['sed']
    gel_clean = disp_dict['gel']

    gelBins = 6
    geltmp = gel_clean.join(da.computeDisplacement(gel_clean))
    geltmp['bin bottom sed'] = pd.cut(geltmp['dist bottom sed'], gelBins)
    geltmp['bin mid'] = geltmp['bin bottom sed'].map(lambda x: round((x.left + x.right) / 2, 1))

    sedBins = 10
    sedtmp = sed_clean.join(da.computeDisplacement(sed_clean))
    sedtmp['bin bottom sed'] = pd.cut(sedtmp['dist bottom sed'], sedBins)
    sedtmp['bin mid'] = sedtmp['bin bottom sed'].map(lambda x: round((x.left + x.right) / 2, 1))

    tMax = 22
    disp_max = 0.4 * 1000
    key = 'disp x'

    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set(rc=sns_context['rc'])
    sns.set_context(sns_context['style'], font_scale=sns_context['font_scale'])

    # make color palette magenta/cyan (or close to it)
    # have sharp transition magenta/cyan at gel/sed bin transition
    p = sns.diverging_palette(300, 185, n=20, s=100)
    b, r = p[0:10], p[10:20]
    b.reverse()
    r.reverse()
    _ = b + r
    _.reverse()

    for frame in range(tMax):
        plt.clf()

        s = sedtmp.xs(frame, level='frame')
        # tmp10['bin mid'] = tmp10['bin bottom sed'].map(lambda x: round((x.left + x.right)/2,1))
        s[key + ' (nm)'] = 1000 * s[key]

        g = geltmp.xs(frame, level='frame')
        # geltmp10['bin mid'] = geltmp10['bin bottom sed'].map(lambda x: round((x.left + x.right)/2,1))
        g[key + ' (nm)'] = 1000 * g[key]

        # concatenate sed and gel and reset index. Note, there is no way that sed and gel pareticles bins are mixed
        # on pivot as gel bin centers are far enough apart and would have to otherwise be equal
        # also bin was based on dist from sed_bottom interface, and they were binned separately.
        tmp = pd.concat([s, g]).reset_index()

        sns.boxplot(data=tmp.pivot(columns='bin mid', values=key + ' (nm)'), orient='h',
                    order=sorted(tmp['bin mid'].unique(), reverse=True),
                    showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 15},
                    whis=(5, 95),
                    palette=_[0:16],
                    showfliers=False
                    )
        plt.xlim(-100, disp_max)
        plt.xlabel('Displacement (nm)')
        plt.ylabel('Height above sed/gel interface (6 um bins)')
        plt.title('Average displacment in shear direction vs. height ')
        plt.savefig('./debug/displacement_sedGel_merged_t{t:02}.png'.format(t=frame))

def getMid(pair):
    """ usage: mIdx_xybins.map(getMid) returns a multiindex of bin centers. """
    def _getMid(interval): return (interval.right + interval.left) / 2
    x, y = pair
    return _getMid(x), _getMid(y)

def vonMises(strain_df_entry):
    exx, exy, exz, eyy, eyz, ezz = strain_df_entry
    return np.sqrt(1/2.0*((exx -eyy)**2 + (eyy-ezz)**2 + (ezz-exx)**2) + 3*(exy**2 + eyz**2 + exz**2))

def stitchGelGlobal():
    """
    Stitch and track gel tracers across experiments, including reference stacks (although this needs to be coded)

    Downstream of this I will need to deal with drift across experiments. In particular, if there is some drift
    between steps, I should separate that from strain. This could be as simple as find the best rigid body trasnslation
    between final frame of step upstream from first frame of current step (ie af and b0), and treating any
    reference mapping as rigid body translation between first frame of step after reference and reference.
    First frame of shear step should always be mapped back to reference to get:
        - absolute positional offset (rigid body translation)
        - preStress, any deviation that is not rigid body translation.
        - maybe Ovito can do this by treating the entire imageStack as unit cell?
        - Is this radically different than applying F+L to all the particles in the gel? Probably not imo
    -Zsolt Sept 21, 2021

    ----Testing-----
        + running this as a test on steps = ['a', 'b'] for experiment tfrGel10292018A_shearRun10292018

    with tp.PandasHDFStoreBig(global_stitch) as s:
        tmpa0 = s.get(0)
        tmpaf = s.get(21)
        tmpb0 = s.get(22)
        tmpbf = s.get(43)

    pd.Index(tmpa0['particle].values) -> 3994 particles at initial time
    pd.Index(tmpa0['particle'].values).intersection(pd.Index(tmpaf['particle'].values)) -> 3302 trajectories
    pd.Index(tmpb0['particle'].values).intersection(pd.Index(tmpaf['particle'].values)) -> 3840 traj
    pd.Index(tmpa0['particle'].values).intersection(pd.Index(tmpbf['particle'].values)) -> 3123 trajectories
    """

    # TODO: add these as kwrd parameters that can be passed as dictionary
    global_stitch = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10292018A_gel_global.h5'
    max_disp = 2  # probably need a large max displacement as the sample may drift betwewen steps.
    metaPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018{step}/tfrGel10212018A_shearRun10292018{step}_metaData.yaml'


    #premable
    from particleLocating import dplHash_v2 as dpl
    mIdx_tuples = []
    tMax_list = []
    # create the single large stitched h5 file to mimic locating all the gel regions all at once, across experiments
    # one file to rule them all, also not the force overwrite by calling with 'w' flag
    with tp.PandasHDFStoreBig(global_stitch, 'w') as s:
        #for step in ['ref','a']: # for the moment skip ref as that hasnt been stitched, nor has directory structure been determined.
        for step in ['ref', 'a', 'b', 'c', 'd', 'e', 'f']:  # all the steps, in order, read from yaml in future
            print('Starting step {}'.format(step))

            # open correspodning yaml file to find out time steps
            _ = dpl.dplHash(metaPath.format(step=step))
            metaData, hash_df = _.metaData, _.hash_df  # not sure if I need all the metaData or just the hash_df
            del _

            # open corresponding stitched file
            _ = {'fName_frmt': '/tfrGel10212018A_shearRun10292018{}'.format(step) + '_stitched_gel_t{:03}.h5',
                 'path': '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018{}/locations'.format(
                     step)}
            tMax = hash_df['t'].max() + 1
            offset = sum(tMax_list)  # note the edge case sum([]) = 0 works by default
            # TODO: add custom columns to loadStitched call trough dictionary expansion
            for frame, data in enumerate(loadStitched(range(tMax), **_)):
                data['frame_local'] = data['frame']
                data['frame'] = data[ 'frame'] + offset  # increment to prevent tp from overwriting the same frame number at different steps
                data['step'] = step
                if step =='ref':
                    data['z (um, refStack)'] = data['z (um, imageStack)']
                    data['z (um, imageStack)'] = data['z (um, refStack)'] - (143-29)
                else:
                    data['z (um, refStack)'] = data['z (um, imageStack)'] + (143-29)
                data['step'] = step
                s.put(data)
                mIdx_tuples.append((step, frame))
            tMax_list.append(tMax)  # increment now, after looping. Also, I explicitly checked off-by-one errors here.

    stitch_param = {'neighbor_strategy': 'KDTree',
                    'pos_columns': ['{} (um, imageStack)'.format(x) for x in ['x', 'y', 'z']],
                    't_column': 'frame', 'adaptive_stop': 0.1, 'adaptive_step': 0.95}
    with tp.PandasHDFStoreBig(global_stitch) as s:  # now track
        for linked in tp.link_df_iter(s, max_disp, **stitch_param): s.put(linked)

    return mIdx_tuples, global_stitch

def query_globalGel(global_stitch: str, mIdx_query: set, mIdx, colKeys: list):
    """
    Params
    ------
    global_stitch: str, path to globally stitched (and tracked) hdf file
    mIdx_query: set of tuples (step,frame) to query
       >> {('ref',0), ('f', 38)}
    mIdx: pandas multiIndex object. Usuall just pd.MultiIndex.from_tuples(mIdx_tuples)
          which is the output of stitchedGelGlobal
    colKeys: list of string corresponding to the columns to return.
             if colKeys is None, this will return all columns

    returns
    -------
     dataFrame of queried (step, frame_local) from a globally tracked database at global_stitch
     with multi Index of (step, frame_local, particle) and all columns specified in colKeys
     This returned object stays in memory, but only the queried keys are loaded into memory.

    Usage

     ---- get mIdx_query (example of start and stop of every step ref,a-f)----
    steps = ['ref', 'a', 'b', 'c', 'd', 'e', 'f']
    mIdx_df = pd.DataFrame(mIdx_tuples,columns=['step', 'frame_local'])
    mIdx = pd.MultiIndex.from_tuples(mIdx_tuples)

    _ = [mIdx_df[mIdx_df['step'] == step].shape[0] - 1 for step in steps]
    mIdx_query = set(
        [elt for elt in zip(steps, [mIdx_df[mIdx_df['step'] == step].shape[0] - 1 for step in steps])] + [(step, 0) for step in

    Zsolt Sept 24 2021
    """
    import trackpy as tp
    import pandas as pd

    l = [mIdx.get_loc(key) for key in mIdx_query]
    tmp = {}
    with tp.PandasHDFStoreBig(global_stitch, 'r') as steps:
        for frame in l:
            # tmp[frame] = s.get(frame).set_index(['step','frame_local','particle'])[['x (um, imageStack)', 'y (um, imageStack)']]
            if colKeys is not None: tmp[frame] = steps.get(frame).set_index(['step', 'frame_local', 'particle'])[colKeys]
            else: tmp[frame] = steps.get(frame).set_index(['step', 'frame_local', 'particle'])
    return pd.concat([tmp[frame] for frame in l]).sort_index()

def globalStressTime():
    """
    ToDo:
       - add in steps from ref/scratch where the following steps were solved.
       - Maybe this should be a separate script?

    1.) set up multiIndex and load tracked data with cont indexing
    2.) compute displacements
    3.) compute finer shifts to match a(t=0) with ref
    4.) Update reference configuration
    5.) Compute stress
    6.) Plot stress vs. time
    """
    return True

def query_pyTables(path:str, frames:list = []) -> pd.DataFrame:
    """
    lazy loading of pyTables across a list of frame queries.
       >> for frame in query_pyTables(gel_pyTables,range(3)): print(frame)
    There should be some simple mechanism to get a default behavior to loop through
    all frames...keep running s.get until it errors, for example.
    Maybe if frames is an empty list, then write a while loop and catch the error with s.get(frame,-1)
    Easy, tp.PandasHDFStoreBig has an attribute of max_frame
    """
    with tp.PandasHDFStoreBig(path) as s:
        if frames == []: frames = list(range(s.max_frame + 1))
        for frame in frames: # cycle through and yield the result
            yield s.get(frame)

def queryPairs_pyTables(path: str, framePairs: list, augment: bool = False):
    """
    given a list of pairs of frames, returns a dataFrame with a multiIndex
    of just those pairs. This should be sufficient to, for example, compute the
    strain across the same list of pairs but everything is loaded lazily.

    Ideally (this code has not be tested yet):
    # given list of pairs of time points
    framePairs = [(0,1),(0,3), (3,5)]

    # save the index
    idx_df = pd.DataFrame(framePairs, columns=['ref','cur']
    idx_df.to_csv(path+'.idx')

    # load one pair lazily
    for n, queryPair in enumerate(queryPairs_pytables(path, framePairs,augment=True)):
        # compute the strain across that pair
        ref,cur, strain_df = queryPair
        strain_df = localStrain(pair_df,framePairs[n][0], framePairs[n][1])

        # save the output a pyTables indexed by the order in which it was called
        strain_table.put(strain_df,n)
        idx_df = idx_df.append({'frame':n, 'ref':ref, 'cur':cur}, ignore_index=True)


    # append the result to a pyTables dataFrame that can likewise be queried lazily.
    # This is all done with peak memory corresponding to a single strain computation
    # across two time points.
    """
    with tp.PandasHDFStoreBig(path) as s:
        for ref, cur in framePairs:
            # query to get the pair
            ref_df = s.get(ref)
            cur_df = s.get(cur)

            # format the data into a multiIndex with pairs of frames
            cur_df = cur_df.reset_index().set_index(['frame', 'particle'])
            ref_df = ref_df.reset_index().set_index(['frame', 'particle'])

            # concat and yield the result
            if augment: yield ref, cur, pd.concat([ref_df, cur_df])
            else: yield pd.concat([ref_df, cur_df])

def makeStrain_pyTables(paths: dict, framePairs: list, params: dict, **kwargs):
    """
    Ideally this would work with parallel proocessing..with some kind of map function.
    Although I am not sure this would work at all as it would require possibly mulitple simultaneous access
    to the pandas store either to read pos_df or place to strain_df
    In principle the placing is independent as I dont really care what order the strain are placed as long
    as I can index back into (ref,cur) by cross referencing idx_df
    """
    idx_df = pd.DataFrame(None, columns=['frame', 'ref', 'cur'])
    with tp.PandasHDFStoreBig(paths['strain_df'],'a') as s: # open the output file
        for n, queryPairs in enumerate(queryPairs_pyTables(paths['locations'], framePairs,augment=True)): # loop over the generator that carries out lazy loading of pairs
            ref,cur, pos_df = queryPairs
            strain_df = localStrain(pos_df,ref,cur, **params['strain'])
            idx_df = idx_df.append({'frame': n, 'ref':ref, 'cur':cur}, ignore_index=True)
            strain_df['frame'] = n
            strain_df['ref'] = ref
            strain_df['cur'] = cur
            s.put(strain_df)
    idx_df.to_csv(paths['index'], index=False)
    return (paths['strain_df'],paths['index'])

def tukey(df: pd.DataFrame, col: list, k: float = 1.5):
    """
    apply tukey fecnes with parameter k to every column in df specified by str in col list
    Note that for default of k=1.5 applied to nnb_count gives min coordination of 9.
    For k ~2.2, you get min nnb count of 4. With that said, applying tukey to nnb is
    not the best way as I have good reason to discard all particles with fewer than
    4 nnb on that ground that the algo doesnt work.
    """

    def _tukey(df: pd.DataFrame, col: str, k: float = 1.5):
        """
        Compute a tukey fence on dataFrame df on columns in colList
        Return the dataframe with only values that lie within the tukey fence
        Also return a fraction of the data that falls within the fence
        """
        stats = df[col].describe()
        q1, q3 = stats['25%'], stats['75%']
        delta = k * (q3 - q1)

        return df[(df[col] > (q1 - k * delta)) & (df[col] < (q3 + k * delta))]

    # quick type check to add functionality for calling with string for
    if type(col) == str: return tukey(df,[col], k)
    else:
        colKey = col.pop()
        if len(col) == 0:
            return _tukey(df, colKey, k)
        else:
            df = tukey(df, colKey, k)
            return tukey(df, col, k)

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


