#!/usr/bin/env bash

#SBATCH -J f_resub_qc
#SBATCH -o tfrGel23042022_f_%A_%a.out
#SBATCH -e tfrGel23042022_f_%A_%a.err
#SBATCH -p serial_requeue
#SBATCH --array=0-94
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-16:00
#SBATCH --mem=16000
#SBATCH --requeue
#SBATCH --account=weitz_lab

declare -a missingHV=(9975 13825 15925 16275 17675 18550 13480 16105 19605 16110  1240  3865 13840 18940  1420  5620 10345 16645 17170 18745  1576  4551  2106  4556  9281 16281 18031 1061  1411  1586  2111 15061 19786   891  2291  9816 7546  9821 12796  6127 13302 14527 18552  2282  3857 7707 16112 19787    17   367  1592 10342 17342  1947 16472 18572 21197  4553 12428 14528 14878 16978 18553 3858  4033  8583 12433 17508    13  1413  4038  7738 368  2118 11568 13323 18573 12998   354  9454 18729 18904  4559  9459  7889 12789 18214  1069  5619 17169 19094  5644  2299  3524 10699)

HASHVALUE=${missingHV[$SLURM_ARRAY_TASK_ID]}
echo "starting hashvalue $HASHVALUE with array index $SLURM_ARRAY_TASK_ID"
echo "resubmission on t0 t121 with hv upto 22k+. hv found from quality control. particle counts were < tukey_lower for gel. Added more training for ilastik network   "

chmod +x /n/holylfs02/TRANSFER/jzterdik/PROJECT/tfrGel23042022/strainRamp/f_imageStack/dplPath/tfrGel23042022_shearRun01052022f_imageStack_dplScript_exec_pipeline.x
/n/holylfs02/TRANSFER/jzterdik/PROJECT/tfrGel23042022/strainRamp/f_imageStack/dplPath/tfrGel23042022_shearRun01052022f_imageStack_dplScript_exec_pipeline.x $HASHVALUE
