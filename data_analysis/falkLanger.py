import pandas as pd
import numba
from scipy.spatial import cKDTree
import numpy as np
from deprecated import deprecated

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
    localStrainArray_df = pd.DataFrame(localStrainArray_np,columns=['D2_min', 'vM',
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
