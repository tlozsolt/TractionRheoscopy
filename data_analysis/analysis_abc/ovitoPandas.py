from data_analysis.analysis_abc.analysis_abc import Analysis
from data_analysis.analysis_abc.strain import Strain
from data_analysis import static as da
import pandas as pd
import freud
import numpy as np
import seaborn as sns
import trackpy as tp
from datetime import datetime
import ovito.io
import ovito.modifiers
import ovito.data
import ovito.pipeline
import os
from functools import partial
from scipy.spatial import cKDTree


class ovitoPandas(Analysis):

    def __init__(self, globalParamFile, stepParamFile):
        super().__init__(globalParamFile, stepParamFile)

        step, letter = self.step, self.step.split('_')[0]
        exptPath = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/'
        #xyzFileFrmt = '/{step}/xyz/cleanSedGel_keepBool/step{letter}_{mat}_t*.xyz'
        xyzFileFrmt = '/{step}/xyz/cleanSedGel_keepBool/ovt_step{letter}_{mat}_t*.xyz'

        self.xyzSed = exptPath + xyzFileFrmt.format(step=step, letter=letter, mat='sed')
        self.xyzGel = exptPath + xyzFileFrmt.format(step=step, letter=letter, mat='gel')

        self.ovitoResults = {}

        # what coordinate system to map to ovito position variables. This will be used to
        # calcualte all particle level properties using ovito, while all others pd columns
        # will be mapped to scalars
        self.coordinateSystem = '(um, rheo_sedHeight)'
        self.identityLambda = lambda df: np.ones(df.shape[0], dtype=bool)
        self.cleanParticles = lambda df: (df['sed_Colloid_core'] > df['fluorescent_chunk_core'])

        self.peakStrain = dict(a=5, b=6, c=6, d=12, e=21,f=51)

    def __call__(self):
        pass

    def setPlotStyle(self):
        super().setPlotStyle()
        #optionally set other plot style parameters here

    def sed(self, frame, selection = lambda df: np.ones(df.shape[0], dtype=bool)):
        """
        Access sediment positions and pass a boolean selection function, that by default is
        elect everything

        A more complicated, multipart selection would be something like this:
        >> d.sed(0, lambda df: (df['cleanSedGel_keepBool']) &
                               (df['z (um, imageStack)'] < 130) &
                               (df['hashValue'] == 50)
                )
        """
        if frame < 0:
            print('querying for frame {}, which is less than 0. return frame =0')
            frame = 0
        sedPos =  super().sed(frame)
        #potentiall do more things like boolean selection on sedPos by passing a lambda selection
        # function directly to sedPos
        return sedPos[selection(sedPos)]

    def gel(self, frame, step=None, gelGlobal: bool = True):
        gelPos = super().gel(frame, step=step, gelGlobal=gelGlobal)
        # potentially do more thing like boolean section for Cleaning specific behavior
        return gelPos

    def gelGlobal2Local(self, gelGlobalGen): return super().gelGlobal2Local(gelGlobalGen)

    def log(self):
        #args = super().getArgs()
        ## pop off anything you dont want output to yaml dict
        #args.pop('globalParam')
        #args['Cleaning'] = self.__dict__
        #args.pop('refFit')
        #args.pop('interfaceIdx')
        #args.pop('interfaceFits')
        # I dont know how to make this work to log input vbariables for when a method was called
        # some combination or manipulationi of locals() and self.__dict__ to remove attributes like
        # dataFrames and fits, while keep attibutes like parameters
        #local = locals()
        #local.pop('self')
        return {'global': self.globalParam,
                'step': self.stepParam,
                'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    def posDict(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'rheo_sedHeight'): pass

    def posList(self,
                posKey_frmt: str= '{coord} ({units}, {coordSys})',
                coordTuple: tuple=('z','y','x'),
                units: str= 'um',
                coordSys: str = 'rheo_sedHeight'): pass

    def load_sed(self, frame, data, selection):
        pos_df = self.sed(frame, selection)  # pos_gen will be something like inst.sed(frame)
        data.particles = ovito.data.Particles(count=pos_df.index.shape[0])
        coordinates = data.particles_.create_property('Position')
        coordinates[:, 0] = pos_df['x {}'.format(self.coordinateSystem)].values
        coordinates[:, 1] = pos_df['y {}'.format(self.coordinateSystem)].values
        coordinates[:, 2] = pos_df['z {}'.format(self.coordinateSystem)].values
        id = data.particles_.create_property('Particle Identifier')
        id[:] = pos_df.index.values

        # load the rest of the dataFrame as just scalars
        for key in pos_df.keys():
            loadBool = False
            try:
                tmp = key.split(' ')[1] == self.coordinateSystem
                if tmp == self.coordinateSystem: pass
                else: loadBool = True
            except IndexError: loadBool = True
            if loadBool:
                #print(key)
                vals = pos_df[key].values
                dtype, components = vals.dtype, vals.shape[0]

                #cast boolean to int as boolean is not support dtype in ovito
                # note that you must use `==`, and NOT `is` for testing types
                # as we need *value* equivalence, not reference equivalence
                if dtype == bool: dtype=int

                _ = data.particles_.create_property(key, dtype=dtype, components=1)
                _[:] = vals.astype(dtype)

            # initalize the ``simulation'' cell with coordinate specific to rheo_sedHeight
            if not data.cell: data.cell = ovito.data.SimulationCell(pbc=(False, False, False))
            data.cell_[:,:3] = [[236.0,0.0,0.0],[0,236.0, 0],[0,0,236]] # axes
            data.cell_[:,3] = [-116, -116, 0] #origin


    def load_gel(self, frame, data):
        pos_df = self.gel(frame)  # pos_gen will be somthing like inst.sed(frame)
        data.particles = ovito.data.Particles(count=pos_df.index.shape[0])
        coordinates = data.particles_.create_property('Position')
        coordinates[:, 0] = pos_df['x {}'.format(self.coordinateSystem)].values
        coordinates[:, 1] = pos_df['y {}'.format(self.coordinateSystem)].values
        coordinates[:, 2] = pos_df['z {}'.format(self.coordinateSystem)].values
        id = data.particles_.create_property('Particle Identifier')
        id[:] = pos_df.index.values

        # load the rest of the dataFrame as just scalars
        for key in pos_df.keys():
            if key.split(' ')[1] == self.coordinateSystem:
                pass
            else:
                _ = data.particles_.create_property(key)
                _[:] = pos_df[key].values

    def makePipeline_pandas(self, mat: str, selectionFunction= None):
        if selectionFunction is None: selectionFunction = self.identityLambda
        # partial(self.load_sed, selection=lambda df: (df['hashValue'] < 60) & (df['hashValue'] > 55))
        if mat == 'sed': pipeline = ovito.pipeline.Pipeline(
                             source=ovito.pipeline.PythonScriptSource(function=
                             partial(self.load_sed, selection= selectionFunction ) ))
        if mat == 'gel': pipeline = ovito.pipeline.Pipeline(
            source=ovito.pipeline.PythonScriptSource(function= self.load_gel))

        self.ovitoPipeline = pipeline

        return pipeline

    def makePipeline(self, mat: str, selectionFunction=None):
        if selectionFunction is None: selectionFunction = self.identityLambda
        # partial(self.load_sed, selection=lambda df: (df['hashValue'] < 60) & (df['hashValue'] > 55))

        if mat == 'sed':
            dummyFile = self.xyzSed
            pipeline = ovito.io.import_file(dummyFile,
                               columns = ['Particle Identifier', 'Position.Z', 'Position.Y', 'Position.X', 'None', 'None', 'None', 'dz', 'dy', 'dx', 'dTotal', 'Dist Intferface'])
            pipeline.modifiers.append(partial(self.load_sed, selection= selectionFunction))

        elif mat == 'gel':
            dummyFile = self.xyzGel
            pipeline = ovito.io.import_file(dummyFile,
                                            columns = ['Particle Identifier', 'Position.Z', 'Position.Y', 'Position.X', 'None', 'None', 'None', 'dz', 'dy', 'dx', 'dTotal', 'Dist Intferface'])
            pipeline.modifiers.append(partial(self.load_sed, selection= selectionFunction ))

        self.ovitoPipeline = pipeline

        return pipeline

    def ovitoObj2Pandas(self, ovitoTable, key: str):

        """
        Guesses data type given dimensions in ovitoTable and then maps to scalar, vector, tensor
        """
        scalar = ['']  # Dont append anything for scalar
        vector = ['.x', '.y', '.z']
        quaternion = ['.x', '.y', '.z', '.w']
        tensor = ['.xx', '.yy', '.zz', '.xy', '.xz', '.yz']

        try:
            dim = ovitoTable.shape[1]
        except IndexError:
            dim = 1

        if dim == 1:
            return pd.DataFrame({key + scalar[n]: ovitoTable[...][:] for n in range(dim)})
        elif dim == 3:
            return pd.DataFrame({key + vector[n]: ovitoTable[...][:, n] for n in range(dim)})
        elif dim == 4:
            return pd.DataFrame({key + quaternion[n]: ovitoTable[...][:, n] for n in range(dim)})
        elif dim == 6:
            return pd.DataFrame({key + tensor[n]: ovitoTable[...][:, n] for n in range(dim)})
        else:
            print('Did not recognize data type for key {} when converting ovitoObj to pd.DataFrame.'.format(key))
            return pd.DataFrame({key + '.{}'.format(str(n)): ovitoTable[...][:, n] for n in range(dim)})

    def _compute(self, ovitoPipeline, frame):
        dataOvito = ovitoPipeline.compute(frame)

        # extract particle level data to pandas data Frame
        df_pos = pd.DataFrame([])
        tensorMapping = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        quaternionMapping = ['x', 'y', 'z', 'w']
        vectorMapping = ['x', 'y', 'z']
        colorMapping = ['R', 'G', 'B']
        for key in list(dataOvito.particles.keys()):

            col = dataOvito.particles[key][...]  # use ellipsis to generate array from ovito dataObject

            if col.ndim == 1: df_pos[key] = col
            else:
                if key == 'Position': mappingList = vectorMapping
                elif key == 'Stretch Tensor' or key == 'Strain Tensor':
                    mappingList = tensorMapping
                elif key == 'Rotation': mappingList = quaternionMapping
                elif key == 'Color': mappingList = colorMapping
                else:
                    print('Not sure what the data type in ovito particle level multi-dim data for key {}'.format(key))
                    mappingList = [n for n in range(col.shape[1])]

                for dim in range(col.shape[1]):
                    df_pos[key + '.{}'.format(mappingList[dim])] = col[:, dim]

        # extract dataTables
        df_tables= {}
        for key, ovitoObj in dataOvito.tables.items():
            tmp = {}
            if key == 'coordination-rdf': tmp['bin_mid'] = self.ovitoObj2Pandas(ovitoObj.xy()[:,0],'bin-mid')
            colStr = [ovitoObj.values()[elt].identifier for elt in range(len(ovitoObj.values()))]
            for colName in colStr:  # data type could array of scalar, vectors, tensors etc
                tmp[colName] = np.array(ovitoObj.get(colName)[...])
                tmp[colName] = self.ovitoObj2Pandas(ovitoObj.get(colName)[...], colName)
            df_tables[key] = pd.concat([tmp[col] for col in tmp.keys()], axis=1)

        #df_pos = df_pos.set_index('Particle Identifier').join(self.sed(frame))
        self.ovitoResults[frame] = dict(pos = df_pos, tables = df_tables, dataOvito = dataOvito)

        return df_pos, df_tables, dataOvito

    def ovito2pd(self, dataOvito=None):
        # extract particle level data to pandas data Frame
        if dataOvito is None: dataOvito = self.dataOvito

        df_pos = pd.DataFrame([])
        tensorMapping = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        quaternionMapping = ['x', 'y', 'z', 'w']
        vectorMapping = ['x', 'y', 'z']
        colorMapping = ['R', 'G', 'B']
        for key in list(dataOvito.particles.keys()):

            col = dataOvito.particles[key][...]  # use ellipsis to generate array from ovito dataObject

            if col.ndim == 1:
                df_pos[key] = col
            else:
                if key == 'Position':
                    mappingList = vectorMapping
                elif key == 'Stretch Tensor' or key == 'Strain Tensor':
                    mappingList = tensorMapping
                elif key == 'Rotation':
                    mappingList = quaternionMapping
                elif key == 'Color':
                    mappingList = colorMapping
                else:
                    print('Not sure what the data type in ovito particle level multi-dim data for key {}'.format(key))
                    mappingList = [n for n in range(col.shape[1])]

                for dim in range(col.shape[1]):
                    df_pos[key + '.{}'.format(mappingList[dim])] = col[:, dim]

        # extract dataTables
        df_tables = {}
        for key, ovitoObj in dataOvito.tables.items():
            tmp = {}
            if key == 'coordination-rdf': tmp['bin_mid'] = self.ovitoObj2Pandas(ovitoObj.xy()[:, 0], 'bin-mid')
            colStr = [ovitoObj.values()[elt].identifier for elt in range(len(ovitoObj.values()))]
            for colName in colStr:  # data type could array of scalar, vectors, tensors etc
                tmp[colName] = np.array(ovitoObj.get(colName)[...])
                tmp[colName] = self.ovitoObj2Pandas(ovitoObj.get(colName)[...], colName)
            df_tables[key] = pd.concat([tmp[col] for col in tmp.keys()], axis=1)

        #df_pos = df_pos.set_index('Particle Identifier').join(self.sed(frame))
        #self.ovitoResults[frame] = dict(pos=df_pos, tables=df_tables, dataOvito=dataOvito)
        return df_pos, df_tables, dataOvito

    def rdf(self, pipeline):
        pipeline.modifiers.append(ovito.modifiers.CoordinationAnalysisModifier( cutoff = 10.0, number_of_bins = 350, enabled = True))
        return pipeline

    def atomicStrain(self,pipeline, refConfig=-1, cutoff=2.8):
        """
        Parameters that should be passed:
        - cutoff distance to search for nnb
        - reference configuration
        """
        # Atomic strain:
        """
        # Atomic strain:
        mod = AtomicStrainModifier()
        mod.cutoff = 2.8
        mod.reference = FileSource()
        mod.frame_offset = -5
        pipeline.modifiers.append(mod)
        mod.reference.load('/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/d_imageStack/xyz/cleanSedGel_keepBool/stepd_sed_t028.xyz')
        
        ~~~
        I think that I should specify the reference configuration direction using something like
        mod.reference=PythonScriptSource()
        mod.reference.function = partial(self.sed, selection)
        mod.reference._evaluate(refFrameNumber)
        
        """
        if type(refConfig) == int and refConfig < 0:
            pipeline.modifiers.append(ovito.modifiers.AtomicStrainModifier(
                cutoff=cutoff,
                output_strain_tensors=True,
                output_stretch_tensors=True,
                output_rotations=True,
                output_nonaffine_squared_displacements=True,
                use_frame_offset=True,
                frame_offset=refConfig))

        elif refConfig == 0:
            pipeline.modifiers.append(ovito.modifiers.AtomicStrainModifier(
                cutoff=2.8,
                output_strain_tensors=True,
                output_stretch_tensors=True,
                output_rotations=True,
                output_nonaffine_squared_displacements=True ))
        else:
            print('Ovito strain pipeline is skipped as refConfig param is not recognized')

        self.ovitoPipeline = pipeline
        return pipeline

    def clusterStrain(self, pipeline,
                      boolExp: str = 'ShearStrain < 0.03', outputStr: str = 'LowStrainBool',
                      clusteringCutoff:float = 2.2):
        """
        Add a step to ovito Pipeline pipeline that computes a new property given by outputStr
        for every particle that satisfies boolExp
        Then carry out spatial cluster on the complement of boolExp upto cutoff of clusteringCutoff.

        Default values of boolExp and outputStr are set to compute particles that no not undrgo large strains
        and hence cluster on particles that do undergo large strains.
        """
        # Assing LowStrainBool = True if ShearStrain < 0.025 - Expression selection:
        #pipeline.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression='ShearStrain < 0.025'))

        # Assing LowStrainBool = True if ShearStrain < 0.025 - Delete selected:
        #pipeline.modifiers.append(ovito.modifiers.DeleteSelectedModifier(enabled=False))

        # Assing LowStrainBool = True if ShearStrain < 0.025 - Assing LowStrainBool = True if ShearStrain < 0.025:
        pipeline.modifiers.append(ovito.modifiers.ComputePropertyModifier(
            expressions=(boolExp),
            output_property=outputStr))

        # Select Not LowStrain:
        pipeline.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression='{} ? 0 : 1'.format(outputStr)))

        # Cluster High Strain Particles:
        pipeline.modifiers.append(ovito.modifiers.ClusterAnalysisModifier(
            cutoff=clusteringCutoff,
            only_selected=True,
            sort_by_size=True,
            compute_com=True,
            compute_gyration=True,
            cluster_coloring=True))

        self.ovitoPipeline = pipeline
        return pipeline

    def makePipeline_fromFile(self, file:str, refConfig: int = -1, selectionFunction=None):
        if selectionFunction is None: selectionFunction = self.identityLambda
        pipeline = ovito.io.import_file(file,
                               columns = ['Particle Identifier', 'Position.Z', 'Position.Y', 'Position.X', 'None', 'None', 'None', 'dz', 'dy', 'dx', 'dTotal', 'Dist Intferface'])
        #pipeline = import_file('/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/f_imageStack/xyz/cleanSedGel_keepBool/stepf_sed_t000.xyz',
        #                      columns = ['Particle Identifier', '', 'Position.Z', 'Position.Y', 'Position.X', '', '', '', '', '', '', '', ''])

        # this is a ass backwards. Initalize from file, then modify with a reloading request? hmmm
        pipeline.modifiers.append(partial(self.load_sed, selection= selectionFunction ))

        # Coordination analysis:
        pipeline.modifiers.append(ovito.modifiers.CoordinationAnalysisModifier( cutoff = 10.0, number_of_bins = 350, enabled = True))

        # Assign ParticleVolume:
        pipeline.modifiers.append(ovito.modifiers.ComputePropertyModifier(
            expressions = ('4/3*pi*(2.043/2)^3',),
            output_property = 'ParticleVolume'))

        # Assign N = 1:
        pipeline.modifiers.append(ovito.modifiers.ComputePropertyModifier(
            expressions = ('1',),
            output_property = 'N'))

        # Atomic strain:
        if type(refConfig) == int and refConfig <0:
            pipeline.modifiers.append(ovito.modifiers.AtomicStrainModifier(
                cutoff = 2.8,
                output_strain_tensors = True,
                output_stretch_tensors = True,
                output_rotations = True,
                output_nonaffine_squared_displacements = True,
                use_frame_offset = True,
                frame_offset = refConfig))

        elif refConfig == 0:
            pipeline.modifiers.append(ovito.modifiers.AtomicStrainModifier(
                cutoff=2.8,
                output_strain_tensors=True,
                output_stretch_tensors=True,
                output_rotations=True,
                output_nonaffine_squared_displacements=True))
        else:
            print('Ovito strain pipeline is skipped as refConfig param is not recognized')


        # Assing LowStrainBool = True if ShearStrain < 0.025 - Expression selection:
        pipeline.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression = 'ShearStrain < 0.025'))

        # Assing LowStrainBool = True if ShearStrain < 0.025 - Delete selected:
        pipeline.modifiers.append(ovito.modifiers.DeleteSelectedModifier(enabled = False))

        # Assing LowStrainBool = True if ShearStrain < 0.025 - Assing LowStrainBool = True if ShearStrain < 0.025:
        pipeline.modifiers.append(ovito.modifiers.ComputePropertyModifier(
            expressions = ('ShearStrain < 0.03',),
            output_property = 'LowStrainBool'))

        # Select Not LowStrain:
        pipeline.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression = 'LowStrainBool ? 0 : 1'))

        # Cluster High Strain Particles:
        pipeline.modifiers.append(ovito.modifiers.ClusterAnalysisModifier(
            cutoff = 2.2,
            only_selected = True,
            sort_by_size = True,
            compute_com = True,
            compute_gyration = True,
            cluster_coloring = True))

        self.ovitoPipeline = pipeline

        return pipeline

    def sedPipeline(self,refConfig, selectionFunction=None):
        return self.makePipeline_fromFile(file=self.xyzSed, refConfig=refConfig, selectionFunction=selectionFunction)
    def gelPipeline(self,refConfig=None): return self.makePipeline_fromFile(file=self.xyzGel, refConfig=refConfig)

    def computeLocalStrain(self, refPos, curPos, nnbArray):
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
        out = np.zeros((nnbArray.shape[0], 11))
        for n in range(len(nnbArray)):
            nnbList = nnbArray[n]
            # get the coordinates of central particle
            # note that each elt of nnbArray is a padded array
            r0_ref = refPos[n]
            r0_cur = curPos[n]

            # initialize X and Y matrices (see Falk and Langer, PRE Eqn 2.11-2.14)
            X = np.zeros((3, 3))
            Y = np.zeros((3, 3))

            # now loop over the indices of nnb
            for m in range(len(nnbList)):
                # this is raising a bug when compiling with numba
                # https://github.com/numba/numba/issues/5680
                # if n == m: pass # this is just the center particle
                # if nnbList[m] == - 1: pass # this is padding
                # else:

                # get particle id
                pid = nnbList[m]

                # get the reference and current positions of the neighboring particle
                r_ref = refPos[pid]
                r_cur = curPos[pid]

                # compute X and Y matrices and add element wise result to stored matrix
                X += np.outer((r_cur - r0_cur), (r_ref - r0_ref))
                Y += np.outer((r_ref - r0_ref), (r_ref - r0_ref))

            # once X and Y have been calculated over the full nnb list, compute deformation tensor
            try:
                epsilon = X @ np.transpose(np.linalg.inv(Y)) - np.identity(3)
            # note, numba has **very** strict limitationon the excpetions, and you cannot pass numpy exceptions
            # but LinAlgError will match under the general Exception class matches
            except Exception:
                epsilon = np.zeros((3, 3))
                epsilon[:] = np.nan

            # with deformation tensor, compute $D^2_{min}$, which caputes mean squared deviation
            # of the deomfration tensor and the actual deformation of the particles.

            # initialize to zero
            D2_min = 0.0
            # loop over all neareast neighbors like the previous for loop
            for m in range(len(nnbList)):
                # if n == 0: pass
                # if nnbList[n] == -1: pass
                # else:
                pid = nnbList[m]
                r_ref = refPos[pid]
                r_cur = curPos[pid]
                # Eqn 2.11 in F+L (except for rolling outer sum on nnb)
                D2_min += np.sum(
                    ((r_cur - r0_cur) - (epsilon + np.identity(3) @ (r_ref - r0_ref))) ** 2)

            # get symmetric and skew symmetric parts of the matrix
            epsilon_sym = 0.5 * (epsilon + np.transpose(epsilon))
            epsilon_skew = 0.5 * (epsilon - np.transpose(epsilon))

            # flatten the array and select the components we care about
            sym_flat = np.array([epsilon_sym[0, 0],
                                 epsilon_sym[0, 1],
                                 epsilon_sym[0, 2],
                                 epsilon_sym[1, 1],
                                 epsilon_sym[1, 2],
                                 epsilon_sym[2, 2]])
            skew_flat = np.array([epsilon_skew[0, 1],
                                  epsilon_skew[0, 2],
                                  epsilon_skew[1, 2]])

            # compute von Mises strain
            vM = np.sqrt(1 / 2 * ((epsilon_sym[0, 0] - epsilon_sym[1, 1]) ** 2
                                  + (epsilon_sym[1, 1] - epsilon_sym[2, 2]) ** 2
                                  + (epsilon_sym[2, 2] - epsilon_sym[1, 1]) ** 2)
                         + 3 * (epsilon_sym[0, 1] ** 2 + epsilon_sym[1, 2] ** 2 + epsilon_sym[0, 2] ** 2))

            # add results to output array
            out[n, :] = np.concatenate((np.array([D2_min, vM]), sym_flat, skew_flat))
        return out

    def computeStrainClusters(self):
        globalOut = {}

        ## run ovito to compute strain ##

        # set up parameters *these should be added as attributes
        strainParam = dict(refConfig=-2, cutoff=2.8)
        clusterParam = dict(boolExp='ShearStrain < 0.030', outputStr='LowStrainBool', clusteringCutoff=2.2)
        frames = range(2, self.frames)
        selectionFunction = lambda df: (df['sed_Colloid_core'] > df['fluorescent_chunk_core'])

        # set up pipeline, perhaps this should be atomized to a different step
        pipeline = self.makePipeline('sed', selectionFunction)
        pipeline = self.atomicStrain(pipeline, **strainParam)
        pipeline = self.clusterStrain(pipeline, **clusterParam)

        # set up output dictionary
        strainRotation = dict(step=self.step, strainParam=strainParam, frames = frames, selectionFunction = selectionFunction)

        # run the pipeline over all the frames in this step
        for frame in frames:
            if frame % 10 == 0: print('Computing frame {}'.format(frame))
            pos, tables, ovitoObj = self.ovito2pd(pipeline.compute(frame))
            #pos = pos[pos['Strain Tensor.xx'] != 0]
            strainRotation[frame] = {'pos': pos, 'tables': tables, 'ovitoObj': ovitoObj}

        globalOut['strainRotation'] = strainRotation

        # this point you can compute cluster size distribution over all the computed frames
        clusterSizeDict = {}
        for frame in frames:
            # pos = strainRotation[frame]['pos']
            # if frame % 10 == 0: print(frame)
            cluster = strainRotation[frame]['tables']['clusters']
            clusterSizeDict[frame] = cluster['Cluster Size']
        out = []
        for frame in frames: out.append(clusterSizeDict[frame])

        clusterSize = pd.concat(out).reset_index()
        print(
            'Adding clusterSize to global out dictionary. See analysis_abc/ovitoPandas/eselbyAlgo for plotting comment')
        globalOut['clusterSize'] = clusterSize
        ##_ = clusterSize[(clusterSize['Cluster Size'] > 7) & (clusterSize['Cluster Size'] < 40)]
        ##sns.displot(data=_['Cluster Size'], binwidth=1, element='step')

        return globalOut

    def eshelbyAlgo(self, computeStrainClusters_out, rotationMethod: str,
                    clusterSizeBounds:tuple=None, subtractAvg: bool = True):
        """
        Given dataFrames of pos/strain and cluster tables output from ovito,
        return a dataFrame of diagonalized strain matrix, position of particles in
        local basis and rotation matrix taking global rheo_sedHeight coordinates to local
        diagonal coordinates for each cluster, and the neighboring region.

        Three rotation methods are implemented:
        - identity: no rotation. Just average the strains. This is implemented as forcing R to be identity
                    and carrying through the same algorithm. It should change nothing.
        - average: average the strain computed for each particle in the cluster. Diagonalize the average, and use that as
                   rotation matrix.
        - FL_cluster: recompute the strain using Falk-Langer for the cluster...ie take all the raw positions in the cluster,
              compute the displacements, and a single strain for the cluster using Falk-Langer. Diagonalize that..

        To visualize output:
        _ = computeStrainCluster()
        out = eshelbyAlgo(_, 'average')
        for n in range(e.frames):
            try: out[n]['frame'] = n
            except KeyError: pass

        # set up bins
        bins = np.arange(-7.5,7.6,1.0)
        for c in ['x','y','z']: localStrain['bin.{}'.format(c)] = pd.cut(localStrain['Position.{}'.format(c)], bins)
        # apply groupby
        grouped = localStrain.groupby(['bin.{}'.format(c) for c in ['x','y','z']]).mean().dropna()
        grouped['Shear Strain'] = grouped.apply(da.vM_ovito ,axis=1)

        # so now how do I plot it? We need to export to tiff, or xyz and visualize in ovito?
        posKeys = ['Position.{}'.format(c) for c in ['x','y','z']]
        strainKeys = ['Strain Tensor.{}'.format(c) for c in ['xx','yy','zz','xy','xz','yz']] + ['Shear Strain']
        path = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/ovitoDataExport/avgClusterStrain_ovtPandas/'
        fName = 'e_t02_test.xyz'
        da.df2xyz(grouped.reset_index()[posKeys+strainKeys],path,fName)

        """

        """
        #>> Now in self.computeStrainClusters() 
        globalOut = {}

        ## run ovito to compute strain ##

        # set up parameters *these should be added as attributes
        strainParam = dict(refConfig=-2, cutoff=2.8)
        clusterParam = dict(boolExp = 'ShearStrain < 0.025', outputStr = 'LowStrainBool', clusteringCutoff=2.2)
        frames = range(2,self.frames)
        selectionFunction = lambda df: (df['sed_Colloid_core'] > df['fluorescent_chunk_core'])

        # set up pipeline, perhaps this should be atomized to a different step
        pipeline = self.makePipeline('sed',selectionFunction)
        pipeline = self.atomicStrain(pipeline, **strainParam)
        pipeline = self.clusterStrain(pipeline,**clusterParam)

        # set up output dictionary
        strainRotation = dict(step=self.step, strainParam=strainParam)

        # run the pipeline over all the frames in this step
        for frame in frames:
            if frame % 10 == 0: print('Computing frame {}'.format(frame))
            pos, tables, ovitoObj = self.ovito2pd(pipeline.compute(frame))
            strainRotation[frame] = {'pos': pos, 'tables': tables, 'ovitoObj': ovitoObj}

        globalOut['strainRotation'] = strainRotation

        # this point you can compute cluster size distribution over all the computed frames
        clusterSizeDict = {}
        for frame in frames:
            # pos = strainRotation[frame]['pos']
            #if frame % 10 == 0: print(frame)
            cluster = strainRotation[frame]['tables']['clusters']
            clusterSizeDict[frame] = cluster['Cluster Size']
        out = []
        for frame in frames: out.append(clusterSizeDict[frame])

        clusterSize = pd.concat(out).reset_index()
        print('Adding clusterSize to global out dictionary. See analysis_abc/ovitoPandas/eselbyAlgo for plotting comment')
        globalOut['clusterSize'] = clusterSize
        ##_ = clusterSize[(clusterSize['Cluster Size'] > 7) & (clusterSize['Cluster Size'] < 40)]
        ##sns.displot(data=_['Cluster Size'], binwidth=1, element='step')
        """

        """
        # Now for the main show:
        # - decide on a frame number (outer for loop)
        # - create local variables pos, cluster
        # - for each cluster in a frame, diagonalize the average strain components...
        # I am not sure this work...I think I need to find the rotation that makes the local strain the cluster
        # closest to pure shear with min middle eigen value...Not the same as averaging the components.
        # maybe the same as averaging the rotation matrix found for each particle? The assumption is that there
        # is a single local basis that simultaneously diagonalizes all the particles in the core as the shear strain
        # are constant in Eshelby. How to find this matrix though?
        # what if I rerun Falk-Langer on the core particles to find the single best fit affine deformation to the whole
        # core?
        #
        # Three options to try and implement in the future:
        # - call falk langer again on the particles in the core, and maybe their nearest neighbors
        # - average the strain compnents in the core and diagonalize the average
        # - diangonalize each particle in the core and somehow come up with an average basis from diagnolization matrices
        # - uuuhghghghg this is tough.
        # These should all be tested on model eshelby inclusions that have been rotated, with and without
        # additive positional noise. 
        """

        globalOut = computeStrainClusters_out
        strainRotation = computeStrainClusters_out['strainRotation']
        frames = strainRotation['frames']
        selectionFunction = strainRotation['selectionFunction']
        strainParam = strainRotation['strainParam']
        globalOut['subtractAvg'] = subtractAvg

        # initialize all fixed parameters
        r = 7 # cKDTree search radius\
        if clusterSizeBounds is None: clusterSizeBounds = (7,17) # min and max cluster size to be considered Eshelby

        # set up some keys to query pos/strain and cluster tables
        posKeys = ['Position.{}'.format(x) for x in ['x', 'y', 'z']]
        epsilonKeys = ['Strain Tensor.{}'.format(x) for x in ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']]
        strainKeys = epsilonKeys + ['Shear Strain', 'Volumetric Strain']
        clusterKeys = ['Cluster']
        ids = ['index', 'Particle Identifier']
        labels = ['Particle Identifier', 'Cluster']

        # start main loop over frames
        for frame in frames:
            if frame % 5 == 0 : print(frame)
            pos = strainRotation[frame]['pos']
            cluster = strainRotation[frame]['tables']['clusters']
            clusterPos = pos[pos['LowStrainBool'] != 1]

            clusterId = cluster[(cluster['Cluster Size'] < clusterSizeBounds[1]) &
                                (cluster['Cluster Size'] > clusterSizeBounds[0])]
            cluster_centerMass_xyz = clusterId[['Center of Mass.{}'.format(x) for x in ['x', 'y', 'z']]].to_numpy()
            cluster_centerMass_xyz_index = clusterId[['Center of Mass.{}'.format(x) for x in ['x', 'y', 'z']]].index

            # create KDTree
            tree = cKDTree(pos[posKeys])
            # query for all nnb upto distance r around the center of mass of all the clusters
            # this has the same indexing as pos
            nnbList = tree.query_ball_point(cluster_centerMass_xyz,r)

            posBlocks = {_c:pos[posKeys].to_numpy()[nnbList[_c]] for _c in range(len(nnbList))}

            # I think I need a 2nd for loop over all the clusters...
            blocks = []
            rotDict = {}
            for _c in range(len(nnbList)):
                if _c  % 100 == 0: print('Processing cluster {} of {}'.format(_c,len(nnbList)))
                c = cluster_centerMass_xyz_index[_c] +1

                # _c indexes into posBlocks, nnbList, and cluster_centerMass_xyz
                # c indexes into pos
                # indices in nnbList index into pos
                # _c and c correspond to the same actual cluster, albeit in different arrays that have different indices

                if subtractAvg:
                    tmp = {'pos':posBlocks[_c], 'nnbList':nnbList[_c], 'strain': pos[epsilonKeys].to_numpy()[nnbList[_c]]}

                    # this is an index of all the bulk particles in the current cluster nbList[_c]
                    # pos['Cluster'] -> cluster id of all the particles
                    # pos['Cluster'].to_numpy() -> same but as a numpy array
                    # pos['Cluster'].to_numpy()[nnbList[_c]] -> use fancy indexing to get just the row corresponding to this cluster indexed by _c
                    # pos['Cluster'].to_numpy()[nnbList[_c]] ==0  ---> boolean expression for rows in cluster with cluster id ==0
                    matrixIdx = np.where(pos['Cluster'].to_numpy()[nnbList[_c]] ==0)

                    #use fancy indexing into strain to compute average
                    avgMatrixStrain = tmp['strain'][matrixIdx].mean(axis=0)

                #c = clusterIndex + 1

                if rotationMethod == 'average':
                    eigenVal, R = np.linalg.eigh(
                        da.flat2Mat(pos[pos['Cluster'] == c][strainKeys].mean()[epsilonKeys].values))

                elif rotationMethod == 'identity':
                    R = np.identity(3)
                    eigenVal = np.array([1,1,1])
                elif rotationMethod == 'FL_cluster':
                    # get Falk-Langer Ids
                    FL_ids = pos.loc[pos[pos['Cluster'] == c]['Particle Identifier'].index][
                        ['Particle Identifier']].values.squeeze()

                    # get ref and cur positions and make into subarrays
                    _rheoSed =  ['{} (um, rheo_sedHeight)'.format(c) for c in ['x','y','z']]
                    ref = self.sed(frame + strainParam['refConfig'], selectionFunction).loc[FL_ids][_rheoSed]
                    cur = self.sed(frame, selectionFunction).loc[FL_ids][_rheoSed]

                    # compute distance from cluster center in ref and get sorted list so that center particle is 1st
                    # ordering isimportant for the computeLocalStrain
                    ref['dist'] = np.sqrt(np.sum((ref.to_numpy() - ref.mean().to_numpy())**2, axis=1))
                    idx = ref.sort_values('dist').index

                    # make numpy array over just the particles in the core.
                    # ToDo: This should be expanded to include nnb, maybe.
                    ref_np,cur_np = ref.loc[idx][_rheoSed].to_numpy(), cur.loc[idx][_rheoSed].to_numpy()

                    # compute strain
                    # nnbList is trivial
                    npStrain = self.computeLocalStrain(ref_np, cur_np, np.array([np.arange(ref_np.shape[0])]))
                    # form pandas dataFrame, keeping in mind the different signature in computeLocalStrain
                    colList = ['D2min', 'vM', 'e00', 'e01', 'e02', 'e11', 'e12','e22', 'r01', 'r02', 'r12']
                    FL_cluster = pd.DataFrame(npStrain, columns=colList)

                    #form a matrix and change signature
                    A = da.flat2Mat(FL_cluster[['e00', 'e11', 'e22', 'e01', 'e02', 'e12']].to_numpy().squeeze())

                    # subtract the matrix average when computing the rotation? I dont know if this the right answer or not.
                    if subtractAvg: A = A - da.flat2Mat(avgMatrixStrain)

                    eigenVal, R = np.linalg.eigh(A)

                else: raise KeyError('rotationMethod {} is not implemented. '\
                                     'Options are (average, identity, FL_cluster). Typo?'.format(rotationMethod))

                # store the rotation matrix for each cluster
                if rotationMethod != 'identity': rotDict[c] = R

                centerOfMass = pos[pos['Cluster'] == c][posKeys].mean().values

                # translate and rotate all positions
                rotatedPositions = np.tensordot(R,posBlocks[_c] - centerOfMass, axes=(1,1))

                # rotate all strains
                localStrain = pos[epsilonKeys].to_numpy()[nnbList[_c]]
                if subtractAvg: localStrain = localStrain - avgMatrixStrain
                epsilonDF = pd.DataFrame(localStrain, columns=epsilonKeys)
                #blocks = {_:(pos[epsilonKeys].to_numpy()[nnbList[_]],
                #             pos[posKeys].to_numpy()[nnbList[_]]) for _ in range(len(nnbList))}

                # could be rewritten with symIndex to replace df.apply operations.
                # ToDo: break down into several steps. Too much going on: compute, rotate, join, etc
                out = pd.DataFrame({posKeys[_]: rotatedPositions[_] for _ in range(3)}).join(
                    pd.DataFrame(np.stack(
                        epsilonDF.apply(lambda x: da.mat2Flat(
                            R.T@da.flat2Mat(x)@R),axis=1)),
                        columns=epsilonKeys)).join(pos.loc[nnbList[_c]][labels].reset_index())
                out['Central Cluster Id'] = c
                out['frame'] = frame
                blocks.append(out)

            globalOut[frame] = pd.concat(blocks)
            globalOut['R.{}'.format(frame)] = rotDict

        return globalOut

    def rotation(self, axis, theta):
        """
        Return a rotation matrix to rotate the angle ``theta'' (radians) about the axis
        Defined using the rodgriues formula

        For rotating into max shear strain from eigenbasis u,v,w take axis = v = [0,1,0] and rotate
        theta = Pi/4 (2Pi/2/2/2). This will rotate in the plane spanned by upper and lower eigenvectors
        (with normal given by the middle eigenvector v)
        """
        k= axis
        K = np.array(((0,-k[2],k[1]),(k[2],0,-k[0]),(-k[1],k[0],0)))
        R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*K@K
        return R



if __name__ == '__main__':
    #stem = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp'
    #xyzDir = stem + '/d_imageStack/xyz/cleanSedGel_keepBool/stepd_sed_t*.xyz'
    #testPath = stem + '/d_imageStack'
    #param = dict(globalParamFile = '../tfrGel23042022_strainRamp_globalParam.yml',
    #             stepParamFile = testPath +'/step_param.yml')
    #os.chdir(testPath)
    #inst = ovitoPandas(**param)
    ##pipeline = inst.makePipeline(xyzDir)
    #pipeline = inst.sedPipeline(refConfig=0)
    #pos, tables, dataOvito = inst.compute(pipeline, 10)
    stem = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp'
    xyzDir = stem + '/e_imageStack/xyz/cleanSedGel_keepBool/stepd_sed_t*.xyz'
    testPath = stem + '/e_imageStack'
    param = dict(globalParamFile='../tfrGel23042022_strainRamp_globalParam.yml',
                 stepParamFile=testPath + '/step_param.yml')
    os.chdir(testPath)

    stepStr = 'e'
    e = ovitoPandas(**param)

    e.frames = 5
    computeStrainClusters_globalOut = e.computeStrainClusters()
    rotationMethods = ['identity', 'average', 'FL_cluster']
    FL_cluster = e.eshelbyAlgo(computeStrainClusters_globalOut,rotationMethods[2])




