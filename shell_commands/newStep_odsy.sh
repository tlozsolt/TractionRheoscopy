salloc -p test -n 4 -N 1 -t 0-01:00 --mem 16000

metaDataYAMLPath=$1

module load Anaconda3/5.0.1-fasrc02
source activate tractionRheoscopy

python
import sys
sys.path.append('/n/home04/jzterdik/TractionRheoscopy')
from particleLocating import dplHash_v2 as dpl
dplInst = dpl.dplHash($metaDataYAMLPath)
os.mkdir('dplPath', 'log', 'locations') #this clearly wont work, but the idea is to create the directories in python
os.mkdir('rawTiff', 'decon', 'flatField')
dplInst.makeSubmitScripts()
dplInst.makeDPL_bashScript()
exit()

#fix make sbatch command

chmod +x tfrGel10212018A_shearRun10292018f_dplScript_exec_pipeline.x
./

