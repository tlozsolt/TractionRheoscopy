# copy directories to PROJECT and SCRATCH on iMac on large external storage
cd /Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018b
mkdir /Volumes/TFR/SCRATCH/tfrGel10212018A/tfrGel10212018A_shearRun10292018b
mkdir /Volumes/TFR/SCRATCH/tfrGel10212018A/tfrGel10212018A_shearRun10292018b/rawTiff
mkdir /Volumes/TFR/SCRATCH/tfrGel10212018A/tfrGel10212018A_shearRun10292018b/decon
mkdir /Volumes/TFR/SCRATCH/tfrGel10212018A/tfrGel10212018A_shearRun10292018b/pyFiji


# create symbolic links to the directory on ext storage within the home directory
ln -s /Volumes/TFR/SCRATCH/tfrGel10212018A/tfrGel10212018A_shearRun10292018b SCRATCH
ln -s /Volumes/TFR/PROJECT/tfrGel10212018A_shearRun10292018b PROJECT

# cp metaDataYaml file from odsy to *.yaml._odsy_Original
# mv metaDataYaml file *.yml to home directory that contain sym links to PROJECT and SCRATCH
cp PROJECT/tfrGel10212018A_shearRun10292018b_metaData.yaml PROJECT/tfrGel10212018A_shearRun10292018b_metaData.yaml_odsyOriginal
mv PROJECT/tfrGel10212018A_shearRun10292018b_metaData.yaml .


# modify the _IMAC filepaths accordingly with absolute filepaths in the first step

# run stitching on modified metaData filepath
# should write *h5 files for each timestep in PROJECT/locations
python
from particleLocating import locationStitch as ls
inst = ls.ParticleStitch('tfrGel10212018A_shearRun10292018b_metaData.yaml')
inst.parStitchAll(0,21)

# run time stitching and tracking with trackpy to create single large h5 file in home or analysis directory
# note wont work on ext storage files.

# upload the whole shebang (locating output) and (tracking) as two separate files to google drive
# see the following for how to do this from the command line
>> https://www.serverkaka.com/2018/05/upload-file-to-google-drive-from-the-command-line-terminal.html
# and also consider uploading to AWS, also from the command line following the AWS bucket listed in the metaData yaml