import sys
import os
import yaml
import pickle as pkl
sys.path.append('/')

from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis.analysis_abc import strain
from data_analysis.analysis_abc import dataCleaning
from data_analysis.analysis_abc import stress
from data_analysis import static as da

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

class Plots(Analysis):

    def __init__(self, globalParamFile, stepParamFile):
        super().__init__(globalParamFile=globalParamFile, stepParamFile=stepParamFile)

        # add additional attributes

        # maybe these should be loaded from saved pkl instance?
        self.strain = strain.Strain(**self.abcParam) # add an instance of strain class as an attribute to access methods
        self.clean = dataCleaning.Cleaning(**self.abcParam)
        self.stress = stress.Stress(**self.abcParam)

        self.figPath = self.paths['figPath']

        #used in sed and gel method to return only locations that are flagged as keepbool in strain
        #self.keepBool = self.strain['keepBool']
        #self.posCoordinateSystem = self.strain['posCoordinateSystem']
        # ---------
        # this throws an error and I am not sure what it was meant to do.
        # self.strain is an instance of analysis_abc.Strain/...so you cant index a df by it. weird
        # I know. I meant to inherit an attribute from self.strain instance. Just go with a double dot...woah

        self.keepBool = self.strain.keepBool
        self.posCoordinateSystem = self.strain.posCoordinateSystem
        #self.keepBoolstr = 'cleanSedGel_keepBool'

        self.strainDataDir = self.paths['strainDataDir']

        self.avgStrain = pd.read_hdf(self.strainDataDir + '/frameAverage_ref0.h5')
        self.avgStrain_dt1 = pd.read_hdf(self.strainDataDir + '/frameAverage_dt1.h5')


    def __call__(self):
        print('Not yet implemented')
        pass

    def sed(self, frame, selection = None):
        if selection is None: selection = lambda df: df[self.keepBool]
        sedPos = super().sed(frame)
        #out = out[out[self.keepBool]]
        sedPos = sedPos[selection(sedPos)]
        posList = self.posList(**self.posCoordinateSystem)
        return sedPos[posList]

    def gel(self, frame: int, selection = None):
        """ get gel positions that are used for strain computation"""
        if selection is None: selection = lambda df: df[self.keepBool]

        # dont know if this is the right syntax
        # to assign output of inherited method to local variables
        out = super().gel(frame)
        #out = out[out[self.keepBool]]
        out = out[selection(out)]
        posList = self.posList(**self.posCoordinateSystem)
        return out[posList]

    def setPlotStyle(self):
        super().setPlotStyle()
        # add additional formatting options here

    def posDict(self,
                posKey_frmt: str = '{coord} ({units}, {coordSys})',
                coordTuple: tuple = ('z', 'y', 'x'),
                units: str = 'um',
                coordSys: str = 'rheo_sedHeight'):
        return super().posDict(posKey_frmt=posKey_frmt,
                               coordTuple=coordTuple,
                               units=units,
                               coordSys=coordSys)

    def posList(self,
                posKey_frmt: str = '{coord} ({units}, {coordSys})',
                coordTuple: tuple = ('z', 'y', 'x'),
                units: str = 'um',
                coordSys: str = 'rheo_sedHeight'):
        return super().posList(posKey_frmt=posKey_frmt,
                               coordTuple=coordTuple,
                               units=units,
                               coordSys=coordSys)

    def log(self): pass

    def gelGlobal2Local(self, gelGlobalGen): super().gelGlobal2Local()

    def avgStrainPlots(self):
        """
        avgStrain keys to be used as columns within sns:
        'Ref: mean fl 2exz (%)',
        'Ref: mean fl 2eyz (%)',
        'Ref: mean fl 2ezz (%)',
        'Ref: mean fl vM',
        'Ref: boundary gap min mean gamma (%)',
        'Ref: boundary gap mean mean gamma (%)',
        'Ref: boundary gap max mean gamma (%)',
        'Ref: x displacement upper boundary (um)',
        'Ref: y displacement upper boundary (um)',
        'Ref: z displacement upper boundary (um)',
        'Ref: x displacement lower boundary (um)',
        'Ref: y displacement lower boundary (um)',
        'Ref: z displacement lower boundary (um)',
        #'Ref: stress xz (mPa)',
        #'Ref: stress yz (mPa)',
        #'Ref: stress zz (mPa)',
        'Ref: residual mean fl - boundary min (%)',
        'Ref: residual mean fl - boundary mean (%)',
        'Ref: residual mean fl - boundary max (%)',
        'dt1: mean fl 2exz (%)',
        'dt1: mean fl 2eyz (%)',
        'dt1: mean fl 2ezz (%)',
        'dt1: mean fl vM',
        'dt1: boundary gap min mean gamma (%)',
        'dt1: boundary gap mean mean gamma (%)',
        'dt1: boundary gap max mean gamma (%)',
        'dt1: x displacement upper boundary (um)',
        'dt1: y displacement upper boundary (um)',
        'dt1: z displacement upper boundary (um)',
        'dt1: x displacement lower boundary (um)',
        'dt1: y displacement lower boundary (um)',
        'dt1: z displacement lower boundary (um)',
        #'dt1: stress xz (mPa)',
        #'dt1: stress yz (mPa)',
        #'dt1: stress zz (mPa)',
        'dt1: residual mean fl - boundary min (%)',
        'dt1: residual mean fl - boundary mean (%)',
        'dt1: residual mean fl - boundary max (%)'
        """

        """ ### Displacement vs. Time ### """
        #

        """### Strain vs Time ### """
        avgStrain = pd.read_hdf(self.strainDataDir +'/frameAverage.h5')

        # comparison of displacements of upper and lower plate
        g = sns.lineplot(data=avgStrain, x=avgStrain.index, y='Ref: x displacement upper boundary (um)',
                         sort=False, label='Upper boundary')
        sns.lineplot(data=avgStrain, x=avgStrain.index, y='Ref: x displacement lower boundary (um)',
                     sort=False, label='Lower Boundary')
        g.set_ylabel('Displacment (um)')
        g.set_xlabel('Frame # (3 min/frame)')

        # Strain two ways
        keys = {1: {'y': 'Ref: mean fl 2exz (%)', 'label': 'Volume Avg Strain 2exz (%)'},
                2: {'y': 'Ref: boundary gap mean mean gamma (%)', 'label': 'Boundary Strain gamma (%)'},
                'ylabel': 'Shear Strain (%)',
                'xlabel': 'Frame # (3 min/frame)'}

        g = sns.lineplot(data=avgStrain, x=avgStrain.index, y=keys[1]['y'], sort=False, label=keys[1]['label'])
        sns.lineplot(data=avgStrain, x=avgStrain.index, y=keys[2]['y'], sort=False, label=keys[2]['label'])
        g.set_ylabel(keys['ylabel'])
        g.set_xlabel(keys['xlabel'])

        # Residual strain
        keys = {1: {'y': 'Ref: residual mean fl - boundary mean (%)', 'label': 'Residual Difference'},
                'ylabel': 'Strain (%)',
                'xlabel': 'Frame # (3 min/frame)'}
        g = sns.lineplot(data=avgStrain, x=avgStrain.index, y=keys[1]['y'], sort=False, label=keys[1]['label'])
        g.set_ylabel(keys['ylabel'])
        g.set_xlabel(keys['xlabel'])

        # stress strain
        # should modify to have only loading and unloading separately, with larger points
        # also trend line?
        # add zero stress/strain
        keys = {1: {'y': 'Ref: stress xz (mPa)', 'x': 'Ref: mean fl 2exz (%)', 'label': 'Stress vs. Strain'},
                'ylabel': 'Stress (mPa)',
                'xlabel': 'Strain (%)'}
        g = sns.lineplot(data=avgStrain, x=keys[1]['x'], y=keys[1]['y'], sort=False, label=keys[1]['label'])
        sns.scatterplot(data=avgStrain, x=keys[1]['x'], y=keys[1]['y'], label=keys[1]['label'])
        g.set_ylabel(keys['ylabel'])
        g.set_xlabel(keys['xlabel'])

        # strain comparison dt1
        keys = {1: {'y': 'dt1: mean fl 2exz (%)', 'label': 'Volume Avg Strain 2exz (%)'},
                2: {'y': 'dt1: boundary gap mean mean gamma (%)', 'label': 'Boundary Strain gamma (%)'},
                'ylabel': 'Shear Strain (%)',
                'xlabel': 'Frame # (3 min/frame)'}

        g = sns.lineplot(data=avgStrain, x=avgStrain.index, y=keys[1]['y'], sort=False, label=keys[1]['label'])
        sns.lineplot(data=avgStrain, x=avgStrain.index, y=keys[2]['y'], sort=False, label=keys[2]['label'])
        g.set_ylabel(keys['ylabel'])
        g.set_xlabel(keys['xlabel'])

        # residual difference dt1 strain comp
        keys = {1: {'y': 'dt1: residual mean fl - boundary mean (%)', 'label': 'Residual strain difference'},
                'ylabel': 'Shear Strain (%)',
                'xlabel': 'Frame # (3 min/frame)'}

        g = sns.lineplot(data=avgStrain, x=avgStrain.index, y=keys[1]['y'], sort=False, label=keys[1]['label'])
        g.set_ylabel(keys['ylabel'])
        g.set_xlabel(keys['xlabel'])

        # scatter plot of incremental stress and and incremental strain
        #how to set x and y labels?
        keys = {1: {'y': 'dt1: mean fl 2exz (%)',
                    'x': 'dt1: stress xz (mPa)',
                    'label': 'dt1'},
                'ylabel': 'Shear Strain (%)',
                'xlabel': 'Frame # (3 min/frame)'}
        g = sns.jointplot(data=avgStrain, x=keys[1]['x'], y=keys[1]['y'])

        # incremental strain time series
        keys = {1: {'y': 'dt1: mean fl 2exz (%)', 'label': 'Volume Avg Strain 2exz (%)'},
                'ylabel': 'Shear Strain (%)',
                'xlabel': 'Frame # (3 min/frame)'}

        g = sns.lineplot(data=avgStrain, x=avgStrain.index, y=keys[1]['y'], sort=False, label=keys[1]['label'])
        sns.scatterplot(data=avgStrain, x=avgStrain.index, y=keys[1]['y'])
        # sns.lineplot(data=avgStrain, x=avgStrain.index, y = keys[2]['y'], sort=False, label=keys[2]['label'])
        g.set_ylabel(keys['ylabel'])
        g.set_xlabel(keys['xlabel'])

        """ ### Strain Distributions ### """
        # distribution for a single frame and strain type
        # generate table inset of mean, std, fraction of pts included in hist
        # set labels.
        # load data
        # create log version in which gaussian is scaled to peak height to emphasize nonlinear tail.
        df = self.strain.getStrain('falkLanger', 0, 25)
        # get the counts
        count_raw = len(df.index)
        df = df[(df['nnb count'] >= 4)] #filter nnb >4
        _ = df.describe()['exz']
        std, mean, count_nnb = _.loc['std'], _.loc['mean'], _.loc['count']
        df = df[abs(df['exz']) <= mean + 4 * std]
        count_4sigma = len(df.index)
        print(std, mean, count_raw, count_nnb / count_raw, count_4sigma / count_raw)
        sns.distplot(df['exz']) #plot histogram
        #plot gaussian with similar parameter to test for tail
        x_min, x_max = -0.06, 0.06
        x = np.linspace(x_min, x_max, 100)
        y = scipy.stats.norm.pdf(x, 0.8 * mean, 0.75 * std)
        plt.plot(x, y, color='coral')

        #pdf of vM strain distrubutions correlation with D2Min
        df = self.strain.getStrain('falkLanger', 0, 25)
        # throw out low nnb count
        df = df[df['nnb count'] > 9]
        df = da.tukeyList(df, ['vonMises', 'D2_min']) # apply Tukey window for each column
        sns.jointplot(data=df, x='vonMises', y='D2_min', kind='hex')

        # particle based correlation (not spatial) of various strain measures
        df = self.strain.getStrain('falkLanger', 0, 25)
        _ = da.tukey(df, ['nnb count'], k=2.2)
        # _ = tukeyList(_,['D2_min', 'vonMises'])
        corr = _.corr() # also look at spearman, however I dont think it is appropriate as the data is continuous
        colList = ['D2_min', 'vonMises', 'exz', 'rxz', 'eyz', 'ryz', 'exy', 'rxy', 'exx', 'eyy', 'ezz', 'nnb count']
        corr = corr.reindex(colList)
        corr = corr[colList]
        corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)

        """ ### stress vs. strain ### """
        # --> three differenent stress measurements:
        #     - displacement of lower boundary of sed
        #     - avg strain of tracers in sample
        #     - displacement of upper boundary of gel
        #
        # ref = 0: stress vs volume avg FL strain on loading
        # ref = 0: stress vs volume avg FL strain on unloading
        # ref = 0: stress vs. volume avg FL strain full cycle
        #
        # ref = cur -1: absolute stress measurement vs. volume avg FL strain
        #               -> does the rate of STZ activation increase with stress?
        #
        return True



    def plot_zBinDisp(self, frame: int,
                      key: str = 'disp x (um, rheo_sedHeight)',
                      xlim: tuple = (-1,10),
                      labels : dict = dict(x='Displacement (um)', y='Height above sed/gel interface',
                                           title='Average displacement in shear direction vs. height'),
                      outFolder: str = '/zBinnedDisplacement',
                      **kwargs):



        # compute the data you need to make this plot
        avgDisp, sedBins, gelBins = self.strain.compute_zBinDisp(frame, **kwargs)

        # If sedBins has 17 bins, discard the last bin and drop everything with that bin from avgDisp
        if len(sedBins) == 17:
            print('Dropping 17th bin because of hard coded value in plots module')
            dropKey = pd.IntervalIndex.from_tuples(sedBins).map(lambda x: round(x.left + x.right)/2,1)[-1]
            avgDisp = avgDisp[avgDisp['bin mid'] != dropKey]
            sedBins = sedBins[0:-1]

        # pivot on the passed key.
        pivot = avgDisp.pivot(columns='bin mid', values=key)

        # do some simple computation on the number of sed and gel bins to set
        # up the colormap
        Ng, Ns = len(gelBins), len(sedBins)
        N = max([Ng, Ns]) + 1 # add one because of some extra stray bin that I dont udnerstand

        # color palette
        p = sns.diverging_palette(300, 185, n=2*N, s=100)
        m, g = p[0:N+1], p[N+1:2*N] #maroon and green color
        m.reverse()
        g.reverse()
        p = m + g # sharp contrast in the middle with white at boundaries

        # where is the zero?
        zeroLoc= sorted(avgDisp['bin mid'].unique(), reverse=True).index(0)

        # sns magic on pivoted table
        f = sns.boxplot(data=pivot,
                    orient='h',
                    order=sorted(avgDisp['bin mid'].unique(), reverse=True),
                    showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5},
                    whis=(5, 95),  # what whiskers to put on? (5,95) will draw whiskers to include mid 90% of data
                    palette=p[zeroLoc - N: zeroLoc + N], # this is the color pallete that was made to be diverging and match colors on expt
                    showfliers=False)

        if xlim is not None: plt.xlim(xlim[0],xlim[1])
        plt.xlabel(labels['x'])
        plt.ylabel(labels['y'])
        plt.title(labels['title'])

        path = self.figPath + outFolder
        if not os.path.exists(path): os.mkdir(path)
        if not os.path.exists(path+'/{}'.format(key)): os.mkdir(path +'/{}'.format(key))
        plt.savefig(path + '/{}'.format(key) + '/displacement_sedGel_merged_t{t:02}.png'.format(t=frame))
        return dict(pivot=pivot, sedBins=sedBins, gelBins=gelBins, zeroLoc=zeroLoc, palette = p)

    def errorBarPlot(self, xArray, yArray, yError, color):
        """
        Typcial call would be
        xArray = shear Strain values
        yArray = shear stress values
        yError = 4900*(s.theilSen['m95_xz'] - s.theilSen['m05_xz'])*68/90
        """
        sns.lineplot(y=yArray,
                     x=xArray, marker=None, sort=False, color=color)
        plt.errorbar(y=yArray,
                     x=xArray,
                     yerr=yError, ls='none', marker='o', markerfacecolor='none', color=color)












