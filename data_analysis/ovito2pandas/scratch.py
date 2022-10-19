from ovito.pipeline import Pipeline, PythonScriptSource
from ovito.io import export_file
from ovito.data import Particles
import ovito.modifiers
import numpy

# Define a data source function, which fills an empty DataCollection with
# some data objects.
def create_particles(frame, data):
    data.particles = Particles(count = 200)
    coordinates = data.particles_.create_property('Position')
    coordinates[:,0] = numpy.linspace(0.0, 50.0, data.particles.count)
    coordinates[:,1] = numpy.cos(coordinates[:,0]/4.0 + frame/5.0)
    coordinates[:,2] = numpy.sin(coordinates[:,0]/4.0 + frame/5.0)

    if not data.cell: data.cell = ovito.data.SimulationCell(pbc=(False, False, False))
    data.cell_[:, :3] = [[50.0, 0.0, 0.0], [0, 1.0, 0], [0, 0, 1]]  # axes
    data.cell_[:, 3] = [0, 0, 0]  # origin

# Create a data pipeline with a PythonScriptSource, which executes our
# script function defined above.
pipeline = Pipeline(source = PythonScriptSource(function = create_particles))

pipeline.modifiers.append(ovito.modifiers.AtomicStrainModifier(
                cutoff=5,
                output_strain_tensors=True,
                output_stretch_tensors=True,
                output_rotations=True,
                output_nonaffine_squared_displacements=True
                ))

tmp = pipeline.compute(3)
