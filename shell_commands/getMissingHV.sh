# log on to storage server
# launch cisco anyconnect to connect to harvard's network
ssh zsolt@10.243.56.39

# in log directory, this returns just the string (hv00010 for example) of files
# that *do not* contain the search string 'particles:' The file name handling is cut to get 4th field
# and then cut again to remove file extension
# After this, I need:
#  [+] to strip 'hv'
#  [+] strip leading zeros, check special case that hv00000 failed...leading zeros are not *all* zeros.
#  [+] replace /n with spaces (solved using printf in awk with a format string "%s " that does not contain new line)
#  [ ] and write to a file
# that can be be read into the sbatch
# Some caveats:
# - The job is 'complete' if we have particle locations, but this may not be the right criterion...for example I may
# still need visualization output
#
# create directory for incomplete logs
mkdir incompleteJobLogs/Jul_6_2020
grep -L "particles: " *yaml | awk '{print "mv", $0, "incompleteJobLogs/Jul_6_2020"}' > mvIncomplete.x
chmod +x mvIncomplete.x
./mvIncomplete.x
# now all the incomplerte logs are in one place and running grep -L on that directory should still work
# if run in the directory containing the incomplete logs
grep -L 'particles: ' *.yaml | cut -d'_' -f4 | cut -d '.' -f1 | cut -c 3- | awk '{printf "%s,", $0 + 0}'
# if run in the dpl directory
grep -L 'particles: ' ../log/incompleteJobLogs/Jul_6_2020/*.yaml | cut -d'_' -f6 | cut -d '.' -f1 | cut -c 3- | awk '{printf "%s,", $0 + 0}'
#since coomand line operations **overide** anything in the script I could pipe the missing hv directly and
# reuse the script as before
sbatch --array=4,8,15,16,23,42  job_script.sbatch
# although I am a bit worried that wont work for few reasons:
# - I need to strip the final comma
# - I need to deal with leaf sizes which will give me duplicate hashValues (hv00001 could hashValue 00001 or 10001)
# I dont know how to pipe the output back to shell, but copy paste would work....

# start interactive session on odsy with 60 min, 4 cores and 16Gb of ram
srun --pty -p test -t 60 --mem 16000 -n 4  /bin/bash
module load

# look at time elapsed for job id (not great formatting)
sacct -j 65507205 --format=JobID,elapsed,nodelist,MaxVMSize

# ssh into storage server
# log into vpn w/ cisco anyconnect
ssh zsolt@10.243.56.39 #kulcs is tfrWeitzlab

# SSH ControlMaster
# in ~/.ssh/config
"""
Host odyZsolt
User jzterdik
HostName login.rc.fas.harvard.edu
ControlMaster auto
ControlPath ~/.ssh/%r@%h:%p
"""
# then start background ssh process
ssh -CX -o ServerAliveInterval=30 -fN odyZsolt
# which I expect to generate a login with 2 factor auth
# all subsequent data copying should work like this, and no require logging in
tar cvf - /mnt/serverdata/zsolt/zsolt/tfrGel09052019x/tfrGel09052019A/shear18052019x/tfrGel09052019a_shearRun20052019e_20190520_75051\ PM_20190521_21001\ AM | ssh odyZsolt "cat > /n/holylfs02/TRANSFER/jzterdik/tfrGel09052019x/tfrGel09052019a_shearRun20052019e.tar"

# steps to submitting a locating run for the first time in a while
# - copy calibration files
#

# start interactive session for 1 hour with 16Gb of ram, 4 cores on 1 node
salloc -p test -n 4 -N 1 -t 0-01:00 --mem 16000
module load Anaconda3/5.0.1-fasrc02
source activate tractionRheoscopy
python
    >>> import sys
    >>> sys.path.append('/n/home04/jzterdik/TractionRheoscopy')
    >>> from particleLocating import dplHash_v2 as dpl
    >>> dplInst = dpl.dplHash('/n/home04/jzterdik/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml')
    >>> dplInst.makeDPL_bashScript()
    >>> dplInst.makeSubmitScripts()
    >>> exit()
# change directories to dplPath
# make additional directories:q
mkdir log locations
chmod +x tfrGel10212018A_shearRun10292018f_dplScript_exec_pipeline.x
./tfrGel10212018A_shearRun10292018f_dplScript_exec_pipeline.x 1

# update anaconda environment from local machine
# make sure the working environment is activated and then run
conda env export > tractionRheoscopy_env_04_18_2021.yml
scp tractionRheoscopy_env_04_18_2021.yml jzterdik@login.rc.fas.harvard.edu:/n/home04/jzterdik
# on odsy, run to create new environment TractionRheoscopy
conda-env create --prefix /n/home04/jzterdik/.conda/envs/TractionRheoscopy -f=/n/home04/jzterdik/tractionRheoscopy_env_04_18_2021.yml

# install mpi4py in order to try srun and --distriubuted flags on ilastik
# following: https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py
# and: https://docs.rc.fas.harvard.edu/kb/mpi-software-on-odyssey/
# and: https://www.ilastik.org/documentation/basics/headless
# this didnt work...contacted RC Help. The alternative to set RAM and processors manually
# I think did work
module load gcc/8.2.0-fasrc01
module load openmpi/4.0.1-fasrc01
export MPICC=$(which mpicc)
echo $MPICC
    >>> /n/helmod/apps/centos7/Comp/gcc/8.2.0-fasrc01/openmpi/4.0.1-fasrc01/bin/mpicc
pip install mpi4py

# debug ilastik
# add following configuration to $HOME/.ilastikrc
[lazyflow]
total_ram_mb=12000
threads=8
#
# then in python, after loading modules etc and delete previous h6 probabilities output file
import sys, subprocess
sys.path.append('/n/home04/jzterdik/TractionRheoscopy')
from particleLocating import dplHash_v2 as dpl
from particleLocating import ilastik
meta = '/n/home04/jzterdik/TractionRheoscopy/metaDataYAML/tfrGel10212018A_shearRun10292018f_metaData.yaml'
pxThreshold = ilastik.ilastikThreshold(meta,computer='ODSY')
pxThreshold._sethv(1)
exec, project, decon = pxThreshold.getPathIlastik('exec'), pxThreshold.getPathIlastik('project'), pxThreshold.getPathIlastik('decon')
run_args = pxThreshold.dpl.metaData['ilastik']['pxClassifier']['run_args']
pxClassifier = ilastik.pixelClassifier(run_args,exec,project)
args = pxClassifier.parseArgs(decon)
subprocess.run(args_srun)

# make resubmit script
# add the following lines
#SBATCH --array=0-189 (modified to be 0 to length of missing hv array
declare -a missingHV=( 123 125 127 128 129 130 131 132 133 134 135 136 586 1504 1557 1577 1589 2133 2606 2611 2768 2769 2770 2776 2777 2778 2834 2973 3077 3302 3429 3454 4078 4081 4119 4414 4425 4429 4433 4451 4455 4458 4459 4460 4467 4480 4499 4514 4752 4786 4787 4788 5060 5063 5092 5093 5098 5100 5101 5102 5103 5106 5107 5108 5109 5110 5111 5116 5117 5120 5122 5125 5126 5127 5128 5131 5133 5134 5135 5136 5143 5145 5146 5148 5149 5150 5154 5155 5156 5157 5164 5165 5167 5168 5176 5177 5178 5182 5183 5184 5190 5209 5210 5220 5221 5222 5223 5233 5236 5237 5238 5239 5247 5248 5249 5391 5392 5393 5719 6177 6201 6226 6434 6779 6784 6840 6845 6854 6906 6907 6909 6922 7017 7104 7112 7251 7252 7575 7581 7583 7589 7590 7591 7593 7594 7595 7596 7601 7602 7814 7839 8047 8058 8096 8097 8121 8122 8144 8145 8493 8527 8697 9434 9459 9986 10176 10197 10198 10199 10200 10201 10202 10203 10204 10272 10445 10457 10458 10499 10500 10514 10650 10688 10846 10847 11051 11052 11075 11250 )
HASHVALUE=${missingHV[$SLURM_ARRAY_TASK_ID]}
echo "starting hashvalue $HASHVALUE with array index $SLURM_ARRAY_TASK_ID"

# shell command for selecting initial and last (or actually 89th) time form hashTable.text
awk '{ if($5 == 0 || $5 == 89) printf "%s ", $1}' hashTable.text

# shell command for checking home directory useage.
df -h ~

# comand for getting summary of running and pending jobs on ODSY
squeue -u jzterdik -h -t pending,running -r -O "state" | uniq -c

# list all files modified after a certain date
find . -type f -newermt '1/30/2017 0:00:00'
find . -type f -newermt '01/07/2022 0:00:00' -name 'tfrGel23042022_shearRun01052022g_imageStack_locations_hv*_gel_trackPy.csv'

# partial command to strip only hv from name of location file
 cut -d'_' -f5 | cut -c 3- | awk '{printf "%s ", $0 + 0}'

# python command to read csv files and find intersection or set difference of indices in two csv files
submitted_hv = pd.read_csv('./gel_hv.csv', sep=' ').T.index
completed_hv = pd.read_csv('./completed_gelResub.text', sep=' ').T.index
submitted_hv[~submitted_hv.isin(completed_hv)]
pd.DataFrame(submitted_hv[~submitted_hv.isin(completed_hv)].to_numpy()).to_csv('../missing_gelResub_24JUN2022.text', sep=' ')