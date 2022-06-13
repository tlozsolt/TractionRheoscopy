#!/usr/bin/env bash

#SBATCH -J f_0t54
#SBATCH -o tfrGel23042022_shearRun01052022f_imageStack_dplLocating_%A_%a.out
#SBATCH -e tfrGel23042022_shearRun01052022f_imageStack_dplLocating_%A_%a.err
#SBATCH -p serial_requeue
#SBATCH --array=0-9624
#SBATCH -n 4
#SBATCH -N 1 
#SBATCH -t 0-16:00
#SBATCH --mem=16000
#SBATCH --requeue

HASHVALUE=$((SLURM_ARRAY_TASK_ID+0))
chmod +x /n/holylfs02/TRANSFER/jzterdik/PROJECT/tfrGel23042022/strainRamp/f_imageStack/dplPath/tfrGel23042022_shearRun01052022f_imageStack_dplScript_exec_pipeline.x
/n/holylfs02/TRANSFER/jzterdik/PROJECT/tfrGel23042022/strainRamp/f_imageStack/dplPath/tfrGel23042022_shearRun01052022f_imageStack_dplScript_exec_pipeline.x $HASHVALUE
