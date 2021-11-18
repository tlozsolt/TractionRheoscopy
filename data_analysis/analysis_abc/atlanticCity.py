import sys
import os
import yaml
import pickle as pkl
sys.path.append('/')

from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis import static as da
from particleLocating import postLocating as pl
from particleLocating import locationStitch as ls
from particleLocating import dplHash_v2 as dpl

import pandas as pd

import trackpy as tp
import ovito
import freud

class AtlanticCity(Analysis):
    """
    This class creates multiple random samples of a position dataframe called with uncertainty columns

    I think the call routine should return a generator so that I can do things like
    strain(atlanticCity(ref,10), atlanticCity(cur,10))
    and this would call strain() a total of 10 times on 10 indpendent resamples configurations
    Not sure how to really do this. I could also just save the data to an hdf file? Depends on how quick it
    is to regenerate the random samples?
    Maybe pass fixed seeds?
    I should save the output though.
    """

    def __init__(self,globalParamFile, stepParamFile):
        super().__init__(globalParamFile=globalParamFile, stepParamFile=stepParamFile)

        self.atlanticCity = self.stepParam['atlanticCity']
