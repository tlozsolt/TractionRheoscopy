# /usr/bin/env bash
#PROJECT=$1 # path to project directory. should be the same as projectDirectory_ODSY
#YAMLFNAME=$2
#TARNAME=$3

YAMLPATH=$1

# use grep to get Project directory
PROJECT=$(grep 'projectDirectory_ODSY' $YAMLPATH | awk '{print $2}')

# use grep to get Transfer directory
TRANSFER=$(grep 'TRANSFER_ODSY' $YAMLPATH | awk '{print $2}')

# use grep to get fileNamePrefix/global
fName_global=$(grep -A1 'fileNamePrefix:' $YAMLPATH | grep 'global' | awk '{print $2}')
tfrGel=$(echo $fName_global | cut -d'_' -f1)
TARNAME=$fName_global$(date '+%d%b%Y').tar
#echo $TARNAME


# make subdirectories in TRASNFER
mkdir $TRANSFER/$fName_global'locating'
#echo $TRANSFER/$fName_global'locating'

#mkdir tfrGel10212018A_shearRun10292018f/dplPath/submissionLogs
mkdir $PROJECT'/dplPath/submissionLogs'
#echo $PROJECT'/dplPath/submissionLogs'

#mv $PROJECT/dplPath/*log $PROJECT/dplPath/submissionLogs
cp $YAMLPATH $PROJECT
#tar -cvf $TARNAME \
#         $PROJECT/log/ \
#         $PROJECT/locations/ \
#         $PROJECT/dplPath/submissionLogs/ \
#         $PROJECT/dplPath/*_hv00000_* \
#         $PROJECT/dplPath/*sbatch \
#         $PROJECT/$YAMLFNAME \
#         -C $TRANSFER
#check this works on a small subset of files
tar -cvf $TRANSFER/$tfrGel/$TARNAME \
         $PROJECT/log/*0.yaml \
         $PROJECT/locations/*0.csv \
         $PROJECT/dplPath/submissionLogs/*00000* \
         $PROJECT/dplPath/*_hv00000_* \
         $PROJECT/dplPath/*sbatch \
         $PROJECT/$YAMLFNAME \
         -C $TRANSFER