from ovito.pipeline import Pipeline, PythonScriptSource
from ovito.io import export_file
from ovito.data import Particles
import numpy
import pandas as pd

# Define a data source function, which fills an empty DataCollection with
# some data objects.


def pos_gen(frame):
    pos_df = pd.DataFrame(
        {'x': [1.0, 2.0, 3.0], 'y': [-1, 0, 1], 'z': [0.1, 0.2, 0.1], 'Particle Identifier': [1, 2, 5]})
    if frame == 0: return pos_df
    else: return None

def create_particles(frame, data):
    pos_df = pos_gen(frame) # pos_gen will be somthing like inst.sed(frame)
    data.particles = Particles(count = pos_df.index.shape[0])
    coordinates = data.particles_.create_property('Position')
    coordinates[:,0] = pos_df['x'].values
    coordinates[:,1] = pos_df['y'].values
    coordinates[:,2] = pos_df['z'].values
    id = data.particles_.create_property('Particle Identifier')
    id[:] = pos_df['Particle Identifier'].values

# Create a data pipeline with a PythonScriptSource, which executes our
# script function defined above.
pipeline = Pipeline(source = PythonScriptSource(function = create_particles))

# Export the results of the data pipeline to an output file.
# The system will invoke the Python function defined above once per animation frame.
#export_file(pipeline, 'output/trajectory.xyz', format='xyz',
#    columns=['Position.X', 'Position.Y', 'Position.Z'],
#    multiple_frames=True, start_frame=0, end_frame=10)