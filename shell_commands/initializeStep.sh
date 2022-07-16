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