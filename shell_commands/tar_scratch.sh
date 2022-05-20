# set up variables including number of time frames, output path
p=$1
N=2
fName='tfrGel23042022_shearRun01052022f_imageStack'

cd $p
#mkdir tarFiles/
#for t in {0..$N}
for ((t=0; t<=N; t++));
do
  T=$(printf "%06d" $t)
  echo "_t${T}_"
done
