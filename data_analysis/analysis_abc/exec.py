import os
from data_analysis.analysis_abc.dataCleaning import Cleaning
from data_analysis.analysis_abc.stitchTrack import StitchTrack
from data_analysis.analysis_abc.stress import Stress
from data_analysis.analysis_abc.strain import Strain
import pickle as pkl
from datetime import date

""" Preamble """
wdir_frmt = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018{}/'
param = dict(globalParamFile = '../tfrGel10212018A_globalParam.yml',
             stepParamFile = './step_param.yml')
#steps = ['ref','a','b','c','d','e','f','g']
steps = ['a','b','c','d','e','f','g']
#steps = ['g']
boolDict = dict(dataCleaning=False,
                stress=False,
                strain=True,
                pickle=True)
for step in steps:

    print('Starting step {}'.format(step) )
    # change directories
    os.chdir(wdir_frmt.format(step))

    #print('running stitch track')
    #stitchTrack = StitchTrack(**param)
    #stitchTrack()


    # dataCleaning
    clean = Cleaning(**param)
    if boolDict['dataCleaning']:
        print('Starting dataCleaning on step {}'.format(step))
        clean()

    ## ToDo: get top and bottom surface ids by manually slicing cleaning xyz file in ovito and save to
    #       ./ovito/sed_t{:03}_topSurface.xyz and ./ovito/sed_t{:03}_bottomSurface.xyz
    #
    # stress
    stress = Stress(**param)
    if boolDict['stress']:
        print('Starting stress on step {}'.format(step) )
        stress()

    # strain
    # ToDo: Parallelize strain run over steps
    strain = Strain(**param)
    if boolDict['strain']:
         print('Starting strain on step {}'.format(step) )
         strain.verbose=True

         ref1 = [(1, cur, 'falkLanger') for cur in range(1, strain.frames)]
         ref1 += [(1, cur, 'boundary') for cur in range(1, strain.frames)]
         strainPaths = dict(ref1=ref1)

         strain(strainPaths=strainPaths)

    #pickle all the instances?
    if boolDict['pickle']:
        inst_dict = dict(clean=clean, stress=stress, strain = strain)
        dateStr = date.today().strftime("%d_%m_%Y")
        with open('./analysis_step_{step}_{date}.pkl'.format(step=step, date=dateStr),'wb' ) as f:
            pkl.dump(inst_dict,f)

    """
    # Ideally what is the order of all the steps? 
    
    + hashed xyz files are copied to kaposzta and all directory files are created
    + gel is stitched for all steps (parallel across frames and steps) 
    + sed is stitched for all steps (parallel across frames and steps)
    + gel is tracked locally (parallel across steps)
    + sed is tracked within each step
    + first and last xyz files are generated and placed in ovito for sed in (um, imageStack) coordinates
    - reference stack has sed surface fit and cleaned
    - gel is tracked globally
    - ovito interface ids are found (manually) by plotting xyz files of complete particles for first and last frames
        - upper and lower surface of sediment: sed_t{:03}_topSurface.xyz
        - sed_complete_bottom_surface.xyz (just the particle ids are exported)
    - stress is determined and in particular rigid body displacement of sample is computed for every time globally
    - Optional on a deep copy: sed is re-tracked with rigid body displacement removed at each step
    - data cleaning for both sed and gel
    - strain on dataCleaned particles
    - stress strain and plots, etc
    - orientation analysis
    """


