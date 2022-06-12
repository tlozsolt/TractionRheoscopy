#!/usr/bin/env bash

#SBATCH -J g_untar
#SBATCH -o tfrGel23042022_untar_%A_%a.out
#SBATCH -e tfrGel23042022_untar_%A_%a.err
#SBATCH -p serial_requeue
#SBATCH --open-mode=append
#SBATCH --array=0
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH --mem=2000
#SBATCH --requeue

TIME=$((SLURM_ARRAY_TASK_ID+0))
chmod +x /n/holylfs02/TRANSFER/jzterdik/tfrGel23042022/tfrGel23042022_shearRun01052022g_imageStack/untar_timeSteps.x
/n/holylfs02/TRANSFER/jzterdik/tfrGel23042022/tfrGel23042022_shearRun01052022g_imageStack/untar_timeSteps.x $TIME
