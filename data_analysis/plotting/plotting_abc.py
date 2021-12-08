from abc import ABC, abstractmethod

import os
import
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl

import yaml
from particleLocating import dplHash_v2 as dpl
from data_analysis.analysis_abc.dataCleaning import Cleaning
from data_analysis.analysis_abc.strain import Strain
from data_analysis.analysis_abc.stress import Stress

class Plotting(ABC):

    def __init__(self, globalParamFile, stepParamFile):

        self.abcParam = dict(globalParamFile=globalParamFile, stepParamFile=stepParamFile)

        with open(globalParamFile, 'r') as f: self.globalParam = yaml.load(f,Loader=yaml.SafeLoader)
        with open(stepParamFile, 'r') as f: self.stepParam = yaml.load(f,Loader=yaml.SafeLoader)

        self.frames = self.strain.frames
        self.stepList: list = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        self.relPath_frmt: str = './tfrGel10212018A_shearRun10292018{}'
        self.expt_wdirs: dict = {step: self.relPath_frmt.format(step) for step in self.stepList}
        self.stem: str = '/Users/zsolt/Colloid/DATA/tfrGel10212018x'

        self.relax: int = 9
        self.color: dict = {step: sns.color_palette('tab10')[n] for n,step in enumerate(self.stepList)}

        # create a dictionary of s
        os.chdir(self.stem)
        self.expt = {}
        for step, relPath in self.expt_wdirs.items():
            if step == 'stem': continue
            else:
                os.chdir(relPath)
                p = dict(globalParamFile='../tfrGel10212018A_globalParam.yml', stepParamFile='./step_param.yml')
                self.expt[step] = dict(strain=Strain(**p), stress=Stress(**p), cleaning=Cleaning(**p))
                os.chdir(self.stem)

    @abstractmethod
    def goto(self, step:str):
        if step != 'stem':
            os.chdir(self.stem)
            os.chdir(self.expt_wdirs[step])
        else: os.chdir(self.stem)

    @abstractmethod
    def strain(self, step: str): return self.expt[step]['strain']

    @abstractmethod
    def stress(self, step: str): return self.expt[step]['stress']

    @abstractmethod
    def cleaning(self, step: str): return self.expt[step]['cleaning']

    @abstractmethod
    def frames(self, step:str): return self.strain(step).frames

    @abstractmethod
    def midPt(self, step: str):
        frames, relax = self.stress(step).frames, self.relax
        return int((self.frames-relax)/2)

    @abstractmethod
    def scatterLinePlt(self, step: str, data: pd.DataFrame, xKwrd: str, yKwrd: str):

        self.goto(step)
        midPt = self.midPt(step)
        frames = self.frames(step)

        # lineplot, all points
        g = sns.lineplot(data=data, x=xKwrd, y=yKwrd, sort=False, color = self.color[step])

        #loading scatter
        sns.scatterplot(data=data[0:midPt], x=xKwrd, y=yKwrd, **dict(s=150, color=self.color[step], marker='^'))

        #peak scatter
        sns.scatterplot(data=data[midPt:midPt+1], x=xKwrd, y=yKwrd, **dict(s=150, color=self.color[step], marker='D'))

        #unloading scatter
        sns.scatterplot(data=data[midPt+1:frames-self.relax], x=xKwrd, y=yKwrd, **dict(s=150, color=self.color[step], marker='v'))

        #relax scatter
        sns.scatterplot(data=data[frames-self.relax: frames], x=xKwrd, y=yKwrd, **dict(s=150, color=self.color[step], marker='o'))

        return plt.gcf()
