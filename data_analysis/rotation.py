import numpy as np
import pandas as pd
from numba import jit
from data_analysis import static as da

def parseRotYaml(rotDict):
    """ From an opened yaml file giving rotation values, return rotation matrices"""
    if rotDict['handed'] == 'right'  and rotDict['positiveSignature'] == 'clockwise' :
        if rotDict['units'] == 'radians' or rotDict['units'] == 'rad':
            theta_x, theta_y, theta_z = rotDict['theta_x'], rotDict['theta_y'], rotDict['theta_z']
        elif rotDict['units'] == 'degrees' or rotDict['units'] == 'deg':
            theta_x, theta_y, theta_z = np.pi/180*np.array([rotDict['theta_x'],
                                                            rotDict['theta_y'],
                                                            rotDict['theta_z']])
        else: raise ValueError("Could not determine units for rotation matrix. {}".format(rotDict['units']))

        r_x = np.array(((1, 0, 0),
                        (0, np.cos(theta_x), -1 * np.sin(theta_x)),
                        (0, np.sin(theta_x), np.cos(theta_x))))
        r_y = np.array(((np.cos(theta_y), 0, np.sin(theta_y)),
                        (0, 1, 0),
                        (-1 * np.sin(theta_y), 0, np.cos(theta_y))))
        r_z = np.array(((np.cos(theta_z), -1 * np.sin(theta_z), 0),
                        (np.sin(theta_z), np.cos(theta_z), 0),
                        (0, 0, 1)))
        prod_zyx = np.matmul(r_z, np.matmul(r_y, r_x))
        prod_xyz = np.matmul(r_x.T, np.matmul(r_y.T, r_z.T))
        return {'prod_zyx (left)': prod_zyx, '(prod_zyx)T (right)': prod_xyz, 'r_x': r_x, 'r_y': r_y, 'r_z': r_z}
    else:
        raise ValueError("not all options are supported in parseRotYaml yet...")

def rotationMatrix(theta_x, theta_y, theta_z):
    """
    Returns three rotation matrices that rotate a vector theta_i degree about coordinate i
    """
    r_x = np.array(((1, 0, 0),
                    (0, np.cos(theta_x), -1 * np.sin(theta_x)),
                    (0, np.sin(theta_x), np.cos(theta_x))))
    r_y = np.array(((np.cos(theta_y), 0, np.sin(theta_y)),
                    (0, 1, 0),
                    (-1 * np.sin(theta_y), 0, np.cos(theta_y))))
    r_z = np.array(((np.cos(theta_z), -1 * np.sin(theta_z), 0),
                    (np.sin(theta_z), np.cos(theta_z), 0),
                    (0, 0, 1)))
    prod_zyx = np.matmul(r_z, np.matmul(r_y, r_x))
    prod_xyz = np.matmul(r_x.T, np.matmul(r_y.T, r_z.T))
    return {'prod_zyx (left)': prod_zyx, '(prod_zyx)T (right)': prod_xyz, 'r_x': r_x, 'r_y': r_y, 'r_z': r_z}


def coordTransform(pos_df, coordStr_current, coordStr_target, z_offSet=0):
    if coordStr_current == '(um, imageStack)' and coordStr_target == '(um, rheo_sedHeight)':
        #z_offSet = kwargs['z_offSet']
        pos_df['x {}'.format(coordStr_target)] = pos_df['x {}'.format(coordStr_current)] - 235 / 2.0
        pos_df['y {}'.format(coordStr_target)] = -1 * pos_df['y {}'.format(coordStr_current)] + 235 / 2.0
        pos_df['z {}'.format(coordStr_target)] = pos_df['z {}'.format(coordStr_current)] + z_offSet
        return pos_df
    else:
        raise ValueError(
            'Coordinate transofrm between {} and {} not recognized or not yet supported'.format(coordStr_current,
                                                                                                coordStr_target))

def rotatePosition(df,rotMatrix, posKeys=None):
    # get numpy array of particle positions to rotate
    # maybe also drop na ?
    # reformat and resize
    # check that the ordering of the coordinates on the position array matches the rotation matrix
    # ...should default to zyx
    if posKeys is None:
        suffix = '(um, rheo_sedHeight)'
        posKeys = {'x': 'x {}'.format(suffix), 'y': 'y {}'.format(suffix), 'z': 'z {}'.format(suffix)}
    pos = df[posKeys.values()].to_numpy().T
    idx = df[posKeys.values()].index

    # this likely needs to be a array shape so that the matrix multiplication can be broadcast correctly
    rotPos = (rotMatrix @ pos).T

    # set index and probably restack
    # I think we'll need a join operation on the dataFrame
    # convert to dataFrame with the right index
    rotPos_df = pd.DataFrame(data=rotPos, index=idx, columns=posKeys.values())

    # should I return a modfied dataframe or just the section of the dataframe that can be joined to the
    # input dataFrame? How about just the section I computed...not the whole thing
    return rotPos_df

@jit(nopython=True)
def _rotate(strainList_np, signature_np, r):
    """
    Param
    :strainList_np: numpy array of strain components
        ->> strainList_np[10,:] gives [exx,exy,exz,eyy,eyz,ezz] of the 10th particle
    :signature_np is a flattened version of the array coordinates of the upper triangle
        -> signature_np = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    : r is the rotation matrix
    """

    # initialize output array
    out = np.zeros(strainList_np.shape)

    # loop over each particle
    for n in range(strainList_np.shape[0]):
        strain_flat = strainList_np[n]

        # init a matrix for rotation
        tmp = np.zeros((3, 3))
        # this should be computed once and passed to the function
        # signature = [(i,j) for i in range(3) for j in range(i,3)]

        # loop over the components
        for comp in range(6):
            x = int(signature_np[comp][0])
            y = int(signature_np[comp][1])
            tmp[x, y] = strain_flat[comp]
        # complete the strain matrix from just the upper triangle
        strain = tmp + tmp.T - np.diag(np.diag(tmp))
        # print(strain.shape)

        # now rotate the strain by conjugating with rot matrix
        #strain_rot = r @ strain @ r.T
        r_inv = np.linalg.inv(r)
        strain_rot = r @ strain @ r_inv
        # print(strain_rot)

        # flatten taking only the upper triangle
        strain_rot_flat = np.zeros(6)
        for comp in range(6):
            x = signature_np[comp][0]
            y = signature_np[comp][1]
            strain_rot_flat[comp] = strain_rot[x, y]

        # assign to output array
        out[n, :] = strain_rot_flat

    return out


def _rotateStrainTraj(strain_df, time_str, r, signature=None, returnIdx='particleValue'):
    """
    I dont think this wrapper should be used as I am moving onto (frame,particle) indexing
    -Zsolt March 7 2021

    Wrapper function on _rotate() that takes as input dataFrames and output dataFrames
    I think the wrapping part works as expected, however that does not mean that rotation
    is working. That still needs to be checked explicitly by comparing to strain computed
    on roated particle positions.
    Also note that this wrapper would need some smalal modifications to work on the asymetric
    part of the strain tensor...mostly the signature needs to be modified.
    Feb 6 2021
    """

    # take the strain df and take only strain components in the required order and convert to numpy array
    # do this using signature and compute sigList as above.
    if signature is None or signature == 'upper triangle':
        _sigList = ['exx', 'exy', 'exz', 'eyy', 'eyz', 'ezz']
        _sig_np = np.array([(i, j) for i in range(3) for j in range(i, 3)])
    else:
        raise ValueError('signature {} is not supported. Use \'upper triangle\' '.format(signature))

    # numpy array of the strains, and index
    strainList = strain_df.loc[(slice(None), _sigList), time_str].to_numpy()
    idx = strain_df.loc[(slice(None), _sigList), time_str].index

    # some more array handling if more than one timepoint is pased
    if type(time_str) == list: t=len(time_str)
    else: t = 1

    # split strainList, particle is going to be index on slowest axis
    Nparticle = idx.shape[0] / len(_sigList)
    Nc = len(_sigList)
    strainList = np.reshape(strainList, int(Nc*Nparticle*t), order='F')
    strainList = np.array(np.split(strainList, t*Nparticle))

    strainList_rot = _rotate(strainList, _sig_np, r)

    # reformat into dataFrame to match input

    # take every Nc (usualy 6 elements from mulitindex) to get list of particle ids.
    particle_idx = idx.get_level_values('particle')[0::Nc]

    # create multi index with particle id and time_str list
    if type(time_str) == list:
        timeParticle_multiIndex = pd.MultiIndex.from_product([time_str, particle_idx], names=('time','particle'))
    else:
        timeParticle_multiIndex = pd.MultiIndex.from_product([[time_str], particle_idx], names=('time','particle'))

    #strain_rot_df = pd.DataFrame(data=strainList_rot.flatten(), index=idx, columns=[time_str])
    strain_rot_df = pd.DataFrame(data=strainList_rot,
                                 index = timeParticle_multiIndex,
                                 columns=_sigList)
    if returnIdx == 'particleValue':
        return strain_rot_df.stack().unstack('time')

    else: return strain_rot_df

def rotateStrain(strain_df_particleValue, r, signature=None):
    if signature is None:
        _keys = ['exx','exy','exz','eyy','eyz','ezz']
        _sig = np.array([(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)])
        _join = ['nnb count', 'D2_min']
    else:
        _keys = signature['keys']
        _sig = signature['sig']
        _join = signature['join']

    tmp = _rotate(strain_df_particleValue[_keys].to_numpy(), _sig, r)
    return pd.DataFrame(data=tmp, index=strain_df_particleValue.index, columns=_keys).join(strain_df_particleValue[_join])

def rodrigues(u,k,theta):
    """
    rotate u in the plane defined by normal k about angle theta using rodrigues formula
    """
    K = np.array(((0,-k[2],k[1]),(k[2],0,-k[0]),(-k[1],k[0],0)))
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*K@K
    return R@u

@jit
def _rodrigues_jit(U,V,theta):
    """
    For np array U of vectors u, rotate them about vector v in np array V by angle theta
    return a numpy array of rotated vectors.
    """
    out = np.zeros(U.shape)
    for n in range(U.shape[0]):
        k = V[n]
        u = U[n]
        K = np.array(((0,-k[2],k[1]),(k[2],0,-k[0]),(-k[1],k[0],0)))
        R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*K@K
        out[n,:] = R@u
    return out

def maxShearStrain(eigen_df, signature=None):
    if signature is None:
        u_keys = ['u{}'.format(x) for x in ['x', 'y', 'z']]
        v_keys = ['v{}'.format(x) for x in ['x', 'y', 'z']]
        w_keys = ['w{}'.format(x) for x in ['x', 'y', 'z']]
        col = ['t{}'.format(x) for x in ['x', 'y', 'z']]
        scaleBool = True
    else:
        pass

    idx = eigen_df.index
    U = eigen_df[u_keys].to_numpy()
    V = eigen_df[v_keys].to_numpy()
    t = _rodrigues_jit(U,V,45*np.pi/180)
    if scaleBool == True:
        l = np.array(list(eigen_df['u'] - eigen_df['w'])*3).reshape((3,eigen_df.shape[0])).T
        t = l*t
    return pd.DataFrame(data=t, index=idx, columns=col)


def disp2xyz(df, multi_Idx, fPath = None, fName_frmt = None, posKeys = None):
    """
    Write displacement to sequence of xyz files.

    Param
    df: dataFrame of position with multiIndex of (time, particle Id)
    multi_Idx: multiIndex of particles to write to file
    fPath: where to write the file
    fName_frmt: fileName with format string for time
        > fName_frmt = 'topGel_z5_completeTraj_t{:02}.xyz'
    posKeys: list of posKeys to write to file and compute the displacement across
        > ['{} (um, rheo_sedHeight)'.format(x) for x in ['x','y','z']]
        Note that the function will output as am angle coord[1]/[0], which in the example
        given above is the y/x angle in plane. However if posKeys is provided in a different order,
        the angle compute will not be y/x.

    return: path to files that have been written.
    """
    if fPath is None: fPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/ovito/track_topGel'
    if fName_frmt is None: fName_frmt = 'topGel_z5_completeTraj_t{:02}.xyz'
    if posKeys is None: posKeys = ['{} (um, rheo_sedHeight)'.format(x) for x in ['x','y','z']]
    tmp = df.loc[multi_Idx]
    refPos = tmp.xs(0,level=0)[posKeys]
    for t in range(90):
        curPos = tmp.xs(t,level=0)[posKeys]
        disp = pd.DataFrame(curPos - refPos)
        theta = pd.Series(disp[posKeys[1]]/disp[posKeys[0]],name='theta').to_frame()
        df = curPos.join(disp,rsuffix=' displacement').join(theta)
        da.df2xyz(df,fPath,fName_frmt.format(t))
    return fPath + '/' + fName_frmt

if __name__ == '__main__':
    pass
    # path to rotation file, strain dataFrame, position dataFrame

    # open rotation file and load position data

    # transform positions to rheo_sedDepth

    # rotate positions

    # compute strain with rotated positions

    # rotate strain with coordinate transformation