import numpy as np
import numba
from numba import prange

@numba.jit(nopython=True)
def compute_eshelbyStrainField(xx,yy,zz, e_transformation = 0.01, a=2.5, v_poisson=1./3.0):
    c = 1./(4.*(4.-5.*v_poisson))
    e_yz0 = (2*e_transformation)*(4 - 5*v_poisson)/(15*(1- v_poisson))
    out = np.zeros(xx.shape[0])
    out[:] = np.nan
    rr = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    for n in range(len(xx)):
        r_sphere = rr[n]
        z = zz[n]
        x = xx[n]
        y = yy[n]
        if r_sphere < a: out[n] = e_yz0
        else:
            # wrong, strainfield is cyclindrically symmetric, not spherically symmetric.
            # Kate plotted the x=0 plane and so all x dependence drops out
            #out[n] = (e_transformation*a**3)/(4.*r**5)*(9.*a**2*c
            #                                           -1*(2.+3.*c)*r**2
            #                                           -15.*c*(7.*a**2 -5.*r**2)*np.cos(4.*np.arccos(z/r)))
            # r is now in cylindrical coordinates.
            r = np.sqrt(z**2 + y**2)
            if r!= 0: theta = z/r
            else: theta = 0
            out[n] = (a**3 * e_yz0)/(4 * (r**2 +x**2)**(9./2.)) * (
                9 * a**2 * c * r**4
                -(2 + 3*c) * r**6
                +9 * c * r**2 * (-8 * a**2 + 5 * r**2 ) * x**2
                +6 * ( r**2 + 4 * c * (a**2 + r**2) ) * x**4
                +4 * ( 1 - 6* c)*x**6
                #+15 * c * r**4 * (-7 * a**2 + 5 * ( r**2 + x**2)) * np.cos(4 * np.arccos(z / r))
                + 15 * c * r ** 4 * (-7 * a ** 2 + 5 * (r ** 2 + x ** 2)) * np.cos(4 * np.arccos(theta))
            )
    return np.column_stack((xx,yy,zz,out))

def eshelbyStrainField(minPt,maxPt,spacing, output_str='(pts,values)',**eshelbyParamDict):
    sampling = np.arange(minPt,maxPt,spacing)
    xx, yy, zz = np.meshgrid(sampling,sampling, sampling)
    out = compute_eshelbyStrainField(xx.flatten(),yy.flatten(),zz.flatten(),**eshelbyParamDict)
    l = int((maxPt-minPt)/spacing)
    if output_str == '(pts,values)': return out
    elif output_str =='array':
        tmp = np.zeros([int((maxPt-minPt)/spacing)]*3)
        # how do I slice numpy arrays?
        tmp[:] = out.reshape((l,l,l,4))[:,:,:,3]
        return tmp
    else: raise KeyError("output_str {} is not recognized".format(output_str))

#@numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True)
def spatialCorr_forLoop(A,B,k):
    """
    Computes the spatial correlation of arrays A and B over the shift vectors with magnitude less than k

    :param A: numpy array, possibly 3D
    :param B: numpy array, possibly 3D
    :param k: tuple, giving the max value of shift vectors to move B on top of A

    return: numpy array of dimension 2k with values equal to the normalize autocorrelation (ie covariance)
            of B shifted by ij, relative to A.
    """
    out = np.zeros((2*k[0]+1, 2*k[1] + 1, 2*k[2] +1))
    for kz in range(-1*k[0],k[0]+1):
        for ky in range(-1*k[1],k[1]+1):
            for kx in range(-1*k[2],k[2]+1):
                if kz == 0:
                    if ky > 0:
                        if kx > 0:    a, b = A[:, ky:,  kx: ], B[:, :-1*ky, :-1*kx  ]
                        elif kx < 0:  a, b = A[:, ky:, :kx  ], B[:, :-1*ky,  -1*kx: ]
                        else:         a, b = A[:, ky:, :    ], B[:, :-1*ky, :       ]
                    elif ky < 0:
                        if kx > 0:    a, b = A[:, :ky,  kx: ], B[:, -1*ky:, :-1 * kx  ]
                        elif kx < 0:  a, b = A[:, :ky, :kx  ], B[:, -1*ky:,  -1 * kx: ]
                        else:         a, b = A[:, :ky, :    ], B[:, -1*ky:, :         ]
                    else:
                        if kx > 0:    a, b = A[:, :  ,  kx: ], B[:, :, :-1 * kx  ]
                        elif kx < 0:  a, b = A[:, :  , :kx  ], B[:, :,  -1 * kx: ]
                        else:         a, b = A[:, :  , :    ], B[:, :, :         ]

                elif kz > 0:
                    if ky > 0:
                        if kx > 0:    a, b = A[kz:, ky:,  kx: ], B[:-1*kz, :-1*ky, :-1*kx  ]
                        elif kx < 0:  a, b = A[kz:, ky:, :kx  ], B[:-1*kz, :-1*ky,  -1*kx: ]
                        else:         a, b = A[kz:, ky:, :    ], B[:-1*kz, :-1*ky, :       ]
                    elif ky < 0:
                        if kx > 0:    a, b = A[kz:, :ky,  kx: ], B[:-1*kz, -1*ky:, :-1 * kx  ]
                        elif kx < 0:  a, b = A[kz:, :ky, :kx  ], B[:-1*kz, -1*ky:,  -1 * kx: ]
                        else:         a, b = A[kz:, :ky, :    ], B[:-1*kz, -1*ky:, :         ]
                    else:
                        if kx > 0:    a, b = A[kz:, :  ,  kx: ], B[:-1*kz, :, :-1 * kx  ]
                        elif kx < 0:  a, b = A[kz:, :  , :kx  ], B[:-1*kz, :,  -1 * kx: ]
                        else:         a, b = A[kz:, :  , :    ], B[:-1*kz, :, :         ]

                else:
                    if ky > 0:
                        if kx > 0:    a, b = A[:kz, ky:,  kx: ], B[-1*kz:, :-1*ky, :-1*kx  ]
                        elif kx < 0:  a, b = A[:kz, ky:, :kx  ], B[-1*kz:, :-1*ky,  -1*kx: ]
                        else:         a, b = A[:kz, ky:, :    ], B[-1*kz:, :-1*ky, :       ]
                    elif ky < 0:
                        if kx > 0:    a, b = A[:kz, :ky,  kx: ], B[-1*kz:, -1*ky:, :-1 * kx  ]
                        elif kx < 0:  a, b = A[:kz, :ky, :kx  ], B[-1*kz:, -1*ky:,  -1 * kx: ]
                        else:         a, b = A[:kz, :ky, :    ], B[-1*kz:, -1*ky:, :         ]
                    else:
                        if kx > 0:    a, b = A[:kz, :  ,  kx: ], B[-1*kz:, :, :-1 * kx  ]
                        elif kx < 0:  a, b = A[:kz, :  , :kx  ], B[-1*kz:, :,  -1 * kx: ]
                        else:         a, b = A[:kz, :  , :    ], B[-1*kz:, :, :         ]
                #out[kz, ky, kx] = (np.mean(a * b) - np.mean(a) * np.mean(b)) / (np.sqrt(np.var(a)) * np.sqrt(np.var(b)))
                # that line is wrong...as the shift vector k is not the same as the array index.
                out[kz + k[0], ky + k[1] ,kx + k[2]] = (np.mean(a * b) - np.mean(a) * np.mean(b)) / (np.sqrt(np.var(a)) * np.sqrt(np.var(b)))

    return out

def _pickSubArray(A,B,kz,ky,kx):
    if kz == 0:
        if ky > 0:
            if kx > 0: a, b = A[:, ky:, kx:], B[:, :-1 * ky, :-1 * kx]
            elif kx < 0: a, b = A[:, ky:, :kx], B[:, :-1 * ky, -1 * kx:]
            else: a, b = A[:, ky:, :], B[:, :-1 * ky, :]
        elif ky < 0:
            if kx > 0: a, b = A[:, :ky, kx:], B[:, -1 * ky:, :-1 * kx]
            elif kx < 0: a, b = A[:, :ky, :kx], B[:, -1 * ky:, -1 * kx:]
            else: a, b = A[:, :ky, :], B[:, -1 * ky:, :]
        else:
            if kx > 0: a, b = A[:, :, kx:], B[:, :, :-1 * kx]
            elif kx < 0: a, b = A[:, :, :kx], B[:, :, -1 * kx:]
            else: a, b = A[:, :, :], B[:, :, :]

    elif kz > 0:
        if ky > 0:
            if kx > 0: a, b = A[kz:, ky:, kx:], B[:-1 * kz, :-1 * ky, :-1 * kx]
            elif kx < 0: a, b = A[kz:, ky:, :kx], B[:-1 * kz, :-1 * ky, -1 * kx:]
            else: a, b = A[kz:, ky:, :], B[:-1 * kz, :-1 * ky, :]
        elif ky < 0:
            if kx > 0: a, b = A[kz:, :ky, kx:], B[:-1 * kz, -1 * ky:, :-1 * kx]
            elif kx < 0: a, b = A[kz:, :ky, :kx], B[:-1 * kz, -1 * ky:, -1 * kx:]
            else: a, b = A[kz:, :ky, :], B[:-1 * kz, -1 * ky:, :]
        else:
            if kx > 0: a, b = A[kz:, :, kx:], B[:-1 * kz, :, :-1 * kx]
            elif kx < 0: a, b = A[kz:, :, :kx], B[:-1 * kz, :, -1 * kx:]
            else: a, b = A[kz:, :, :], B[:-1 * kz, :, :]

    else:
        if ky > 0:
            if kx > 0: a, b = A[:kz, ky:, kx:], B[-1 * kz:, :-1 * ky, :-1 * kx]
            elif kx < 0: a, b = A[:kz, ky:, :kx], B[-1 * kz:, :-1 * ky, -1 * kx:]
            else: a, b = A[:kz, ky:, :], B[-1 * kz:, :-1 * ky, :]
        elif ky < 0:
            if kx > 0: a, b = A[:kz, :ky, kx:], B[-1 * kz:, -1 * ky:, :-1 * kx]
            elif kx < 0: a, b = A[:kz, :ky, :kx], B[-1 * kz:, -1 * ky:, -1 * kx:]
            else: a, b = A[:kz, :ky, :], B[-1 * kz:, -1 * ky:, :]
        else:
            if kx > 0: a, b = A[:kz, :, kx:], B[-1 * kz:, :, :-1 * kx]
            elif kx < 0: a, b = A[:kz, :, :kx], B[-1 * kz:, :, -1 * kx:]
            else: a, b = A[:kz, :, :], B[-1 * kz:, :, :]
    return a,b


#@numba.jit(nopython=True, cache=False)
#def _corr_roll(A, B, k):
#    if k.shape == (3,):
#        kz, ky, kx = k
#        a = A[kz:, ky:, kx:].flatten()
#        b = np.roll(np.roll(np.roll(B, kz, axis=0), ky, axis=1), kx, axis=2)[kz:, ky:, kx:].flatten()
#        return np.corrcoef(a, b)[1, 0]
#    elif k.shape == (2,):
#        ky, kx = k
#        a = A[ky:, kx:].flatten()
#        b = np.roll(np.roll(B, ky, axis=0), kx, axis=1)[ky:, kx:].flatten()
#        return np.corrcoef(a, b)[1, 0]
#    elif k.shape == 1 or k.shape == (1,):
#        a = A[k:].flatten()
#        b = np.roll(B, k)[k:].flatten()
#        return np.corrcoef(a, b)[1, 0]
#    else: raise ValueError("shift vector k must have shape (1,), (2,) or (3,) ")


#@numba.jit(nopython=True, cache=False)
#def _corr3(A,B,k):
#    out = np.zeros((2*k[0]+ 1, 2*k[1]+1, 2*k[2] + 1))
#    for kz in range(-1*k[0],k[0]+1):
#        for ky in range(-1 * k[1], k[1] + 1):
#            for kx in range(-1 * k[2], k[2] + 1):
#                a = A[kz:, ky:, kx:].flatten()
#                b = np.roll(np.roll(np.roll(B, kz, axis=0), ky, axis=1), kx, axis=2)[kz:, ky:, kx:].flatten()
#                out[kz + k[0], ky + k[1], kx + k[2]] = np.corrcoef(a, b)[1, 0]
#    return out
#
##@numba.vectorize
#@numba.jit(nopython=True)
#def roll2(B,ky,kx):
#    """ numba does not implment roll with axis arguement, so do this manually"""
#    b_ky = np.transpose(np.roll(np.transpose(np.roll(B,ky)),kx)).flatten()
#
#    return np.roll(np.roll(B, ky, axis=int(0)), kx, axis=int(1))[ky:, kx:].flatten()
#
##@numba.vectorize([numba.float64(numba.float64,numba.float64)])
#@numba.jit(nopython=True)
#def coef(a,b): return np.corrcoef(a,b)[1,0]
#
#
##@numba.jit
#def _corr2(A,B,k):
#    out = np.zeros((2 * k[0] + 1, 2 * k[1] + 1))
#    for ky in range( k[0] + 1):
#        for kx in range( k[1] + 1):
#            # crop
#            a = A[ky:, kx:].flatten()
#            b = B[ky:, kx:].flatten()
#
#            # roll B and crop to all signatures
#            b_pp = np.roll(np.roll(B, ky, axis=int(0)), kx, axis=int(1))[ky:, kx:].flatten()
#            b_pn = np.roll(np.roll(B, ky, axis=int(0)), -1*kx, axis=int(1))[ky:, kx:].flatten()
#            b_np = np.roll(np.roll(B, -1*ky, axis=int(0)), kx, axis=int(1))[ky:, kx:].flatten()
#            b_nn = np.roll(np.roll(B, -1*ky, axis=int(0)), -1*kx, axis=int(1))[ky:, kx:].flatten()
#
#            out[k[0] + ky, k[1] + kx] = np.corrcoef(a, b_pp)[1, 0]
#            out[k[0] + ky, k[1] - kx] = np.corrcoef(a, b_pn)[1, 0]
#            out[k[0] - ky, k[1] + kx] = np.corrcoef(a, b_np)[1, 0]
#            out[k[0] - ky, k[1] - kx] = np.corrcoef(a, b_nn)[1, 0]
#
#    return out
#
#@numba.jit(nopython=True, cache=False)
#def _corr1(A,B,k):
#    out = np.zeros((2*k[0]+ 1))
#    for kx in range(-1 * k[0], k[0] + 1):
#        a = A[k:].flatten()
#        b = np.roll(B, k)[k:].flatten()
#        out[ kx + k[0]] = np.corrcoef(a, b)[1, 0]
#    return out
#
#
#
#def spatialCorr_roll(A, B, k):
#    """
#    Computes the spatial correlation of A and B over all shift vectors of with lengths less than k
#    using roll and pearson correlation coeficient
#
#    Parameters
#    :param A numpy array with dimensions 1, 2 or 3
#    :param B numpy array with dimenions 1, 2, 0r 3
#    :param tuple with shape matching the dimensions of A and B giving the maximum value of shift vector component
#    :param padOut boolean, if True returns an output array of the same dimension as A and B.
#                           if False, no additional padding is done and output array is dictated by size of k
#    :return spatial correlation with k=0 (no shift) located in the center of the output array.
#
#    > A = da.arange(15**2).reshape((15,15)).rechunk(chunks=(3,3)).
#    > A.map_overlap(lambda x: spatialCorr(x,x,np.array((2,2)),padOut=True),depth=1, dtype='float32').compute()
#    """
#
#    if k.shape == (3,): out = _corr3(A,B,k)
#    elif k.shape == (2,): out = _corr2(A,B,k)
#    elif k.shape == (1,): out = _corr1(A,B,k)
#    else: raise ValueError("shift vector k must have shape (1,), (2,) or (3,) ")
#    return out

def spatialCorr_daskWrapper(A,B,k,padOut=True, corrFunc='roll'):
    """
    Wrapper for correlation function to be used in dask array with map blocks
    Two options for how to compute the correlation function flagged using corrFunc
    Also has option to pad the output array to the same size as the input for dimension
    mathching in dask map blocks

    Parameters
    : param A numpy array, probably a chunk if applied using map blocks
    : param B numpy array, probably the same as A if computing autocorrelation
    :param k, tuple of int specifying magnitude in interpolated pixels of the shift vector.
              This will scan over all shift vectors with components whose magnitude is less than
              that specifed in k. ie if k=(5,5,2) this will scan over shift vectors (0,0,0), (5,-4,2)
              but not (3,3,3)
    :param corrFunc, str specifying if spatial correlation should be calculated using forLoop function
                     of numpy roll functionality. These should be the same, although numpy roll is more
                     recently implemented and I think is conceptually cleaner, easier to maintain, and possibly faster
                     roll is also implemented for spatial dimensions 1,2,and 3, while forLoop is just dim 3.
    :param padOut boolean specifying whether the output array should be padded to have the same shape as the input.

    return: array of spatial correlation with middle value specifying the zero shift.
    """
    if corrFunc == 'roll':
        if k.shape == (3,): out = _corr3(A, B, k)
        elif k.shape == (2,): out = _corr2(A, B, k)
        elif k.shape == (1,): out = _corr1(A, B, k)
        else: raise ValueError("shift vector k must have shape (1,), (2,) or (3,) ")
        #out = spatialCorr_roll(A,B,k)
    elif corrFunc == 'forLoop':
        if k.shape == (3,): out = spatialCorr_forLoop(A,B,k)
        elif k.shape == (2,): out = spatialCorr_forLoop(np.array([A]), np.array([B]),np.array([0,k[0],k[1]])).squeeze()
        elif k.shape == (1,): out = spatialCorr_forLoop(np.array([[A]]), np.array([[B]]),np.array([0,0,k[0]])).squeeze()
        else: raise ValueError("spatialCorr_forLoop only implemented arrays of dim 1,2, or 3")
    else: raise ValueError("corrFunc must be either roll or forLoop in daskWrapper")

    if padOut == False: return out
    else:
        if k.shape==(3,):
            padded = np.zeros_like(A,dtype='float32')
            cz,cy,cx = ((np.array(padded.shape) - np.array(out.shape))/2).astype(int)
            padded[cz:cz+out.shape[0], cy:cy+out.shape[1], cx:cx + out.shape[2]] = out
            return padded
        elif k.shape==(2,):
            padded = np.zeros_like(A,dtype='float32')
            cy,cx = ((np.array(padded.shape) - np.array(out.shape))/2).astype(int)
            padded[ cy:cy+out.shape[0], cx:cx + out.shape[1]] = out
            return padded
        elif k.shape==(1,):
            padded = np.zeros_like(A,dtype='float32')
            cx = ((np.array(padded.shape) - np.array(out.shape))/2).astype(int)
            padded[ cx:cx + out.shape[0]] = out
            return padded
        else: raise ValueError("shift vector must have shape (1,) or (2,) or (3,)")

@numba.jit(nopython=True, parallel=True)
def fillArray(A,n):
    out = np.ones(n)
    a = A
    b = a
    for i in range(n):
        out[i] = (np.mean(a * b) - np.mean(a) * np.mean(b)) / ( np.sqrt(np.var(a)) * np.sqrt(np.var(b))) - 1
    return out


if __name__ =='__main__':
    eshelby_strainField = eshelbyStrainField(-10.,10.,0.1,output_str='array')

    import dask.array as da

    A_3d = da.arange(75**3).reshape((75,75,75)).rechunk(chunks=(15,15,15))
    autoCorr_3d = lambda x: spatialCorr_daskWrapper(x, x, np.array((5, 5, 5)), corrFunc='forLoop', padOut=True)

    out_overlap3D = A_3d.map_overlap(autoCorr_3d, depth=1, dtype='float32', boundary='none').compute()
    slice = out_overlap3D[:,:,7]

    _A_2d = np.random.random_sample(25**2).reshape((25,25))
    A_2d = da.from_array(_A_2d).rechunk(chunks=(5,5))
    autoCorr_2d = lambda x: spatialCorr_daskWrapper(x,x,np.array((1,1)),corrFunc='forLoop',padOut=True)
    out_overlap2D = A_2d.map_overlap( autoCorr_2d, depth=1, dtype='float32', boundary='none').compute()
    #out_overlap2D = da.arange(25 ** 2).reshape((25, 25)).rechunk(chunks=(5, 5)).map_overlap(
    #    lambda x: spatialCorr_daskWrapper(x, x, np.array((1, 1)),corrFunc='forLoop', padOut=True), depth=1, dtype='float32', boundary='none').compute()
    #out_block = da.arange(25 ** 2).reshape((25, 25)).rechunk(chunks=(5, 5)).map_blocks(
    #    lambda x: spatialCorr_daskWrapper(x, x, np.array((1, 1)), padOut=True), dtype='float32').compute()




