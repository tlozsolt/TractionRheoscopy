import sys
sys.path.extend(['/Users/zsolt/Colloid/SCRIPTS/tractionForceRheology_git/\
                  kilfoil_locating_python/track'])

import feature3D
import trackpy as tp

#   ToDo:
#     [ ] run the python locating code to make sure it compiles for this python version etc
#     [ ] try both feature3D and feature3D gaussian
#     [ ] modify the functions to take keyword arguments so that I can pass parameter\
#         dictionaries including parameters that are hard coded in kilfoils code like
#           - percentile
#           - number of fracShift iterations
#     [ ] find out what exactly llmx3D is doing and what does rodrigo suggest I do.
#     [ ] wrap this into a class that can be interfaced with
