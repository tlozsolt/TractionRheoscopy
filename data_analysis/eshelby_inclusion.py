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
def spatialCorr(A,B,k):
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



