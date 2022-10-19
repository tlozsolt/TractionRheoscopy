from ovito.io import *
from ovito.modifiers import *
from ovito.pipeline import *
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import freud

# This consists of two functions. One to read xyz files and form pandas dataFrames
# and another to take a pandas dataFrame, stream an xyz file to ovito and load into ovito

#testFile = 'outFile.xyz'

def makePipeline(file:str):
    pipeline = import_file(file,
                           columns = ['Particle Identifier', 'Position.Z', 'Position.Y', 'Position.X', 'None', 'None', 'None', 'dz', 'dy', 'dx', 'dTotal', 'Dist Intferface'])
    #pipeline = import_file('/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/f_imageStack/xyz/cleanSedGel_keepBool/stepf_sed_t000.xyz',
    #                      columns = ['Particle Identifier', '', 'Position.Z', 'Position.Y', 'Position.X', '', '', '', '', '', '', '', ''])

    # Coordination analysis:
    pipeline.modifiers.append(CoordinationAnalysisModifier( cutoff = 10.0, number_of_bins = 350, enabled = True))

    # Assign ParticleVolume:
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = ('4/3*pi*(2.043/2)^3',),
        output_property = 'ParticleVolume'))

    # Assign N = 1:
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = ('1',),
        output_property = 'N'))

    # Atomic strain:
    pipeline.modifiers.append(AtomicStrainModifier(
        cutoff = 2.8,
        output_strain_tensors = True,
        output_stretch_tensors = True,
        output_rotations = True,
        output_nonaffine_squared_displacements = True,
        use_frame_offset = True,
        frame_offset = -1))

    # Assing LowStrainBool = True if ShearStrain < 0.025 - Expression selection:
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'ShearStrain < 0.025'))

    # Assing LowStrainBool = True if ShearStrain < 0.025 - Delete selected:
    pipeline.modifiers.append(DeleteSelectedModifier(enabled = False))

    # Assing LowStrainBool = True if ShearStrain < 0.025 - Assing LowStrainBool = True if ShearStrain < 0.025:
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = ('ShearStrain < 0.03',),
        output_property = 'LowStrainBool'))

    # Select Not LowStrain:
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'LowStrainBool ? 0 : 1'))

    # Cluster High Strain Particles:
    pipeline.modifiers.append(ClusterAnalysisModifier(
        cutoff = 2.2,
        only_selected = True,
        sort_by_size = True,
        compute_com = True,
        compute_gyration = True,
        cluster_coloring = True))

    return pipeline

def sed():
    # make an ovito pipeline that just imports the sed xyz positions, then parses

def gel()
    # analogous, but for gel position

def compute(ovitoPipeline, frame):
    dataOvito = ovitoPipeline.compute(frame)

    # extract particle level data to pandas data Frame
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
    df_tables= {}
    for key, ovitoObj in dataOvito.tables.items():
        tmp = {}
        if key == 'coordination-rdf': tmp['bin_mid'] = ovitoObj2Pandas(ovitoObj.xy()[:,0],'bin-mid')
        colStr = [ovitoObj.values()[elt].identifier for elt in range(len(ovitoObj.values()))]
        for colName in colStr:  # data type could array of scalar, vectors, tensors etc
            tmp[colName] = np.array(ovitoObj.get(colName)[...])
            tmp[colName] = ovitoObj2Pandas(ovitoObj.get(colName)[...], colName)
        df_tables[key] = pd.concat([tmp[col] for col in tmp.keys()], axis=1)

    return df_pos, df_tables, dataOvito

def flat2Mat(strain_flat):
    """From a list of strain values ordered as in ovito, return an array"""
    xx,yy,zz = strain_flat[0], strain_flat[1],strain_flat[2]
    xy, xz, yz = strain_flat[3], strain_flat[4],strain_flat[5]
    return np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])


def ovitoObj2Pandas(ovitoTable, key: str):
    """
    Guesses data type given dimensions in ovitoTable and then maps to scalar, vector, tensor
    """
    scalar = [''] # Dont append anything for scalar
    vector = ['.x','.y','.z']
    quaternion = ['.x','.y','.z','.w']
    tensor = ['.xx','.yy','.zz','.xy','.xz','.yz']

    try: dim = ovitoTable.shape[1]
    except IndexError: dim = 1

    if dim ==1: return pd.DataFrame({key + scalar[n]: ovitoTable[...][:] for n in range(dim)})
    elif dim ==3: return pd.DataFrame({key + vector[n]: ovitoTable[...][:,n] for n in range(dim)})
    elif dim ==4: return pd.DataFrame({key + quaternion[n]: ovitoTable[...][:,n] for n in range(dim)})
    elif dim ==6: return pd.DataFrame({key + tensor[n]: ovitoTable[...][:,n] for n in range(dim)})
    else:
        print('Did not recognize data type for key {} when converting ovitoObj to pd.DataFrame.'.format(key))
        return pd.DataFrame({key + '.{}'.format(str(n)) : ovitoTable[...][:,n] for n in range(dim)})

if __name__ == '__main__':

    testFile = '/Users/zsolt/Colloid/DATA/tfrGel23042022/strainRamp/d_imageStack/xyz/cleanSedGel_keepBool/stepd_sed_t*.xyz'
    pipeline = makePipeLine(testFile)
    pos, dataTables, dataOvito = compute(pipeline, 11)

    # plot rdf for each time point
    """
    print(data.tables['coordination-rdf'].xy())
    rdf_dict={}
    for frame in range(pipeline.source.num_frames):
        print('Processing frame {}'.format(frame))
        data = pipeline.compute(frame)
    
        #plot the rdf
        plt.clf()
        rdf = pd.DataFrame(data.tables['coordination-rdf'].xy(),columns=['r (um)', 'g(r)'])
        rdf_dict[frame] = rdf
        if frame %10 == 0:
            sns.lineplot(data=rdf, x='r (um)', y='g(r)', sort=False)
    """

    # data tables on cluster
    """
    data.tables # returns dict of key value, with values internal objects in ovito
    data = pipeline.compute(10)
    data.tables['clusters'].values() # rerturn list of ovito object with str key...ie [Property('Cluster Identifier')]
    data.tables['clusters'].get('Cluster Size') # return array of 'Cluster Size' values from Property('Cluster Size')
    """
    # how to index cluster index into particle level data? There is a column in particle level data of 'Cluster' that
    # is the cluster id equal 0 reserved for 'Not in any cluster' -> yes
    """
    df[df['Cluster'] == 1] # returns a dataFrame of all particle that are in largest cluster (near grid)
    """
    # compute an average of any property over clusters
    """
    df.groupby('Cluster').mean()
    """


