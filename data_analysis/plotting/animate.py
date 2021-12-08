import os
import shutil
import re

#wdir = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/figures/vonMises_stepd_zeroPadded'
#wdir = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018d/plots/ovito/vonMises_strainClusters'
wdir = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/figures/zBinDisp/fixedx'

os.chdir(wdir)
os.mkdir('./animate/')
N = 30 # fps is generally 30, and N/fps will give the time spent on each frame in the video

# ToDo: - incorporate into plotting class
#       - This function should take as input folder and a search string
#         It should create a folder in a SCRATCH/animate/searchString
#         and output the animated frames there.
#       - maybe also write a file with the path to non-interpolated frames which will not be kept in scratch
#       - how to handle overwrite mistake files? Make an optional, by defualt, empty suffix string for
#         animate/str(searchString + suffix) folder that is created
#

for fName in os.listdir(wdir):
    if not re.search('.png', fName): continue
    tStr = fName.split('_')[-1][1:4]
    stem = fName.split('_')[0] +'_t{:04}.png'
    t = int(tStr)
    for frame in range(N):
        #stem = 'step_d_animate_t{:04}.png'
        tInterp = (t-1)*N + frame
        shutil.copy(fName,'./animate/'+stem.format(tInterp))

