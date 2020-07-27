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
grep -L 'particles: ' *.yaml | cut -d'_' -f4 | cut -d '.' -f1 | cut -c 3- | awk '{printf "%s", $0 + 0}'
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

