# /usr/bin/env bash
#salloc -p test -n 1 -N 1 -t 0-01:30 --mem 4000

YAMLPATH=$1

# use grep to get Project directory
PROJECT=$(grep 'projectDirectory_ODSY' $YAMLPATH | awk '{print $2}')
cd $PROJECT

# use grep to get Transfer directory
TRANSFER=$(grep 'TRANSFER_ODSY' $YAMLPATH | awk '{print $2}')

# use grep to get fileNamePrefix/global
fName_global=$(grep -A1 'fileNamePrefix:' $YAMLPATH | grep 'global' | awk '{print $2}')
tfrGel=$(echo $fName_global | cut -d'_' -f1)
TARNAME=$fName_global"locatingArchive_"$(date '+%d%b%Y').tar
#echo $TARNAME


# make subdirectories in TRASNFER
mkdir $TRANSFER/$fName_global'locating'
#echo $TRANSFER/$fName_global'locating'

#mkdir tfrGel10212018A_shearRun10292018f/dplPath/submissionLogs
mkdir $PROJECT'/dplPath/submissionLogs'
mv $PROJECT'/dplPath/*log' $PROJECT'/dplPath/submissionLogs'
mv $PROJECT'/dplPath/*sbatch' $PROJECT'/dplPath/submissionLogs'
mv $PROJECT'/dplPath/*_hv00000_*' $PROJECT'/dplPath/submissionLogs'
#echo $PROJECT'/dplPath/submissionLogs'

#mv $PROJECT/dplPath/*log $PROJECT/dplPath/submissionLogs
cp $YAMLPATH $PROJECT
tar -cvf $TRANSFER/$tfrGel/$TARNAME \
         $PROJECT/log/ \
         $PROJECT/locations/ \
         $PROJECT/dplPath/submissionLogs/ \
         $PROJECT/$YAMLFNAME

#check this works on a small subset of files
#tar -cvf $TRANSFER/$tfrGel/$TARNAME \
#         $PROJECT/log/*00000.yaml \
#         $PROJECT/locations/*hv0000*.csv \
#         $PROJECT/dplPath/submissionLogs/*00000* \
#         $PROJECT/dplPath/*_hv00000_* \
#         $PROJECT/dplPath/*sbatch \
#         $PROJECT/$YAMLFNAME