"""
This set of function accomplishes the following:

Input:
-trained neural network that gives pixel classification probabilities for:
    - gel particle
    - sediment particle
    - gel background
    - sediment background
    - interface background

- stack from hashValue, after deconvolution
- metaData including whether this hashvalue is an interface and sed/gel

Output
- a masked version of the tiff stack that when run through maxEnt thresholding will give either all the sediment
  particles or all the gel particles depending on sed/gel

Steps
- fft crop (same as before)
- make some decisions about what to do
- run ilastik headless and output numpy array with probability balues
- add/subtract/multiply label prbabilities from input tiff to get appriate mask
  - in demo test runs for interface (sed) the following worked well:
      raw * (1- binary(gel Particle + gel background + interface))
  - and then maxEnt threshold sent the gel portion to zero leaving only the sediment particles whose fluor intensity
    was not really modified.

"""

def loadIlastik(**ilastikDict):
    """

    follwing example with python subprocess:

        https://gist.github.com/VolkerH/06cb11218a2c4adea63e000adc8f9fce

        --- or ---

        https://forum.image.sc/t/how-to-call-and-run-ilastik-from-python/22009/26

    Call ilastik from ilastikDict which likely has the following keywords:
        - path
        - output type
        - ilp file path
        - input path
    return numpy array of probability pixel classifier and dictionary of
    what each dimension corresponds to (eg gel particle)
    """
    return None



