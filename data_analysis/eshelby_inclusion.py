import numpy as np
import numba

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
            out[n] = (a**3 * e_yz0)/(4 * (r**2 +x**2)**(9./2.)) * (
                9 * a**2 * c * r**4
                -(2 + 3*c) * r**6
                +9 * c * r**2 * (-8 * a**2 + 5 * r**2 ) * x**2
                +6 * ( r**2 + 4 * c * (a**2 + r**2) ) * x**4
                +4 * ( 1 - 6* c)*x**6
                +15 * c * r**4 * (-7 * a**2 + 5 * ( r**2 + x**2)) * np.cos(4 * np.arccos(z / r))
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



if __name__ =='__main__':
    eshelby_strainField = eshelbyStrainField(-10.,10.,0.1,output_str='array')



