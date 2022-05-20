# /usr/bin/env bash

# set up variables including number of time frames, output path
p=$1
N=122
fName='tfrGel23042022_shearRun01052022f_imageStack'

cd $p
#mkdir tarFiles/
for t in {0..$N}
do
  T=$(printf "%06d" $t)
  echo "_t${T}_"
  #ls $p | grep -E _t${T}_
  #echo "${fileList}"
  #ls $p | grep -E _t${T}_ | tar -T - -cvf tarFiles/${T}.tar
  ls $p | grep -E _t${T}_ | tar -T - -cvf - | ssh odyZsolt "cat > /n/holylfs02/TRANSFER/jzterdik/tfrGel23042022/${fName}/${T}.tar"

done
# loop over every time step, use grep to match file pattern, and tar only those
# write directly to odsy using