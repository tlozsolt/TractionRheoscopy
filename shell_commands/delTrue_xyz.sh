step=$1
for fName in step$step_sed_t*xyz; do
    outName='ovt_'$fName
    echo $outName
    sed 's/True\ //g' $fName > $outName
done
