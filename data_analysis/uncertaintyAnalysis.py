import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.fftpack import fftshift, fftfreq
from scipy import signal
from scipy.stats import gaussian_kde
from data_analysis import static

def resampleConfig(posDataFrame, N_samples = 10, keyList=None, px2Micron=None):
    """
    Generates a new configuration from the estimated uncertainties in the input dataframe
    posDataFrame is assumed to contain at least six columns:
        - x,y,z position (default `x (um, imageStack)`...
        - error in pixels 'xstd'
    """
    if keyList == None:
        keyList = {dim: '{} (um, imageStack)'.format(dim) for dim in ['x', 'y', 'z']}
    if px2Micron == None:
        px2Micron = {'x': 0.115, 'y': 0.115, 'z': 0.15}

    xPos = np.random.normal(posDataFrame[keyList['x']],
                            px2Micron['x']*posDataFrame['{}_std'.format('x')],
                            np.array([N_samples, len(posDataFrame.index)]))
    yPos = np.random.normal(posDataFrame[keyList['y']],
                            px2Micron['y']*posDataFrame['{}_std'.format('y')],
                            np.array([N_samples, len(posDataFrame.index)]))
    zPos = np.random.normal(posDataFrame[keyList['z']],
                            px2Micron['z']*posDataFrame['{}_std'.format('z')],
                            np.array([N_samples, len(posDataFrame.index)]))

    # then reformat to datafame
    tmp =[keyList[str(x)] for x in ['x','y','z']]
    ref = posDataFrame[tmp].set_index(pd.MultiIndex.from_product([[0],posDataFrame.index],names=['frame','particle']))

    mIdx = pd.MultiIndex.from_product([range(1, N_samples + 1), posDataFrame.index], names=['frame', 'particle'])
    resampled_df = pd.DataFrame(xPos.flatten(), index=mIdx, columns = [keyList['x']] ).join(
                   pd.DataFrame(yPos.flatten(), index=mIdx, columns = [keyList['y']] )).join(
                   pd.DataFrame(zPos.flatten(), index=mIdx, columns = [keyList['z']] ))
    return pd.concat([ref,resampled_df])

def distDecon(signal, error):
    """
    This function takes in two numpy arrays signal and error and returns a deconvolved distribution assuming
    the signal was corrupted by additive error from the distriubtion error.
    THe procedure is:
        1) Determine kernel density estimates for both signal and error
        2) return iFFT( FFT(signal_kde) / FFT(error_kde) )
    Param
    signal: 1D numpy array of values. Only the distribution will used so order here does not matter
    error: 1D numpy array of values of simulated errors, typically generated using resampleConfiguration in
           markov chain monte carlo method.

    return: kde of deconvolved distribution.
    """
    return True

def _estimateSNR(N, bw_kde=None, mu=0, sigma=2):
    """
    estimates the signal to noise ratio used in Weiner deconvolution by comparing kde from sampled gaussian
    to exact gaussian. These results are independent of mu and sigma, and depend only on the number of samples
    and likely the bandwidth used to get kde

    N: positive integer, the number of samples to draw. This should match the number of sampled from which you
       have resampled the error distribution
    bw_kde: small positive number used in estimating the kernel density from the resampled histogram. There are optimal
            choices for this, however they depend on on the distribution and in particular heavy tails. I think
            qualitatively the rule is to start from gaussian and use larger bandwidths to smooth out poorly sampled
            heavy tails, possibly replacing sigma with IQR or something to that effect. Check results by eye.

            For gaussian distribution, see Silverman's rule and Scott's rule for optimal choices of bw_kde:
            silverman's rule: ~1.06*sigma*N**(-1/5)
            scott's rule: N**(-1/5)
            If bw_kde == None: Scott's rule is used.

    mu, sigma: parameters for gaussian distribution

    return: (SNR as function of frequency, shifted frequency scale)
    """

def _wienerDecon(measurement, impulseResponse, SNR, t):
    """
    Compute a wiener regularized fourier deconvolution using the following fomrula:
    ifft( fft(measurement) / fft(impulseResponse) * (1/(1 + 1/(SNR)))
    where SNR is ration of power spectral density of pds(signal)/psd(kernel)

    meaurement: np array of probability mass function of measured singal sampled at regular intervals
    impulseResonse: np array of probability mass fucntion of impulse response sampled at regualr intervals
    SNR: np array of frequency dependent ratio of signal to noise, typically psd(idealSignal)/psd(sampling error)

    return deconvolved probability density.
    """
    impulse_psd = fft(impulseResponse)*np.conj(impulseResponse)
    delta = t[1] - t[0]
    return (ifft(fft(measurement)/fft(impulseResponse)*(1/(1+1/(impulse_psd (SNR)))))/delta,
            fftshift(t))


if __name__ == '__main__':
    # load a small subset of the data
    dataPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/locations_stitch'
    dataName = 'tfrGel10212018A_shearRun10292018f_sed_stitched_micro_t00.h5'

    sedPos = pd.read_hdf(dataPath+'/{}'.format(dataName) )
    keyList = {dim : '{} (um, imageStack)'.format(dim) for dim in ['x', 'y', 'z']}
    px2Micron = {'x':0.115, 'y': 0.115, 'z': 0.15}

    # generate another configuration assuming the loaded configuration is "noise free"
    # the entire thing is just three calls for x,y,z to the numpy function np.random.normal
    # >>> np.random.normal(sedPos['x (um, imageStack)'], 0.115*sedPos['x_std'], np.array([N_sample, N_particle]))


    xPos = np.random.normal(sedPos[keyList['x']], px2Micron['x']*sedPos['{}_std'.format('x')], np.array([10,len(sedPos.index)]))
    yPos = np.random.normal(sedPos[keyList['y']], px2Micron['y']*sedPos['{}_std'.format('y')], np.array([10,len(sedPos.index)]))
    zPos = np.random.normal(sedPos[keyList['z']], px2Micron['z']*sedPos['{}_std'.format('z')], np.array([10,len(sedPos.index)]))

    # then reformat to datafame
    mIdx = pd.MultiIndex.from_product([range(1, 11), sedPos.index], names=['sample', 'particle'])
    #resampled_df = pd.DataFrame([xPos.flatten(),yPos.flatten(), zPos.flatten()],
    #                            index=mIdx, columns=[keyList['x'], keyList['y'], keyList['z']])
    resampled_df = pd.DataFrame(xPos.flatten(), index=mIdx, columns=[keyList['x']])
    resampled_df = resampled_df.join(
        pd.DataFrame(yPos.flatten(),index=mIdx, columns = [keyList['y']] )).join(
        pd.DataFrame(zPos.flatten(),index=mIdx, columns = [keyList['z']] ))

    resampled_df = resampleConfig(sedPos, N_samples=1000)



    # compute the strain over the two configuration
    # tabulate the statistics for each particle
    # run a bunch of samples, say 100 runs.
