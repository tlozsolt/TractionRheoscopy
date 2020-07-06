#!/usr/bin/env bash
for LAMBDA_exp in {5..7}; do
  LAMBDA=$(bc<<< "scale=6; 5*10^(-1*$LAMBDA_exp)")
  echo $LAMBDA
  LAMBDA_dec=$(printf '%f' $LAMBDA)
  echo $LAMBDA_dec
  java  -Xms1024m -Xmx16g -jar \
  /Applications/Fiji.app/plugins/DeconvolutionLab_2.jar Run  \
  -image file /Volumes/TFR/tfrGel10212018A_shearRun10292018f/flatField/tfrGel10212018A_shearRun10292018f_flatField_hv00108.tif \
  -psf file /Users/zsolt/Colloid/DATA/DeconvolutionTesting_Huygens_DeconvolutionLab2/OddysseyHashScripting/psfPath/psf_dim730x730x167z_typeSED_absZ171.tif \
  -algorithm RLTV 30 $LAMBDA_dec \
  -path /Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon \
  -out stack noshow tfrGel10212018A_shearRun10292018f_decon_hv00108_5e$LAMBDA_exp \
  -monitor no \
  -apo NO NO \
  -pad E2 E2
done