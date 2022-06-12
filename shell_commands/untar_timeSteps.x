# /usr/bin/env bash

T=$(printf "%06d" $1)
fName="${T}.tar"
path="/n/holyscratch01/spaepen_lab/zsolt/mnt/serverdata/zsolt/zsolt/tfrGel23042022_shearRun01052022_strainRamp/g_imageStack/rawTiff"
#echo $fName
tar -xf $fName -C $path

