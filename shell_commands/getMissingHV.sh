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
# I dont know how to pipe the output back to shell, but copy paste would work...

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
./tfrGel10212018A_shearRun10292018f_dplScript_exec_pipeline.x

# update anaconda environment from local machine
# make sure the working environment is activated and then run
conda env export > tractionRheoscopy_env_04_18_2021.yml
scp tractionRheoscopy_env_04_18_2021.yml jzterdik@login.rc.fas.harvard.edu:/n/home04/jzterdik
# on odsy, run to create new environment TractionRheoscopy
conda-env create --prefix /n/home04/jzterdik/.conda/envs/TractionRheoscopy -f=/n/home04/jzterdik/tractionRheoscopy_env_04_18_2021.yml


