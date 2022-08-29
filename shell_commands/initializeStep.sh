# steps to initialize metaData yaml file form template
cd /n/holyscratch01/spaepen_lab/zsolt/PROJECT/tfrGel23042022/strainRamp
mkdir a_preShearFullStack
cd a_preShearFullStack
mkdir dplPath
mkdir log
mkdir locations
cp cp ~/TractionRheoscopy/metaDataYAML/tfrGel23042022_shearRun01052022STEP_ref_metaData.yaml .
mv tfrGel23042022_shearRun01052022STEP_ref_metaData.yaml tfrGel23042022_shearRun01052022a_preShearFullStack_metaData.yaml

vi tfrGel23042022_shearRun01052022a_preShearFullStack_ref_metaData.yaml
> %s/STEP/a_preShearFullStack/g
>> 6 substituions on 6 lines
cp tfrGel23042022_shearRun01052022a_preShearFullStack_ref_metaData.yaml ~/TractionRheoscopy/metaDataYAML/
# manually check fileNamePrefix/rawTiff and ocmpare with stem in SCRATCH/rawTiff directory

salloc -p test -n 4 -N 1 -t 0-01:00 --mem 32000 # use more ram for ref stack due to OOM during flatFielding.
module load Anaconda3/5.0.1-fasrc02
source activate tractionRheoscopy
python
    >>> import sys
    >>> sys.path.append('/n/home04/jzterdik/TractionRheoscopy')
    >>> from particleLocating import dplHash_v2 as dpl
    >>> dplInst = dpl.dplHash('/n/home04/jzterdik/TractionRheoscopy/metaDataYAML/tfrGel23042022_shearRun01052022f_postShearFullStack_metaData.yaml')
    >>> dplInst.makeDPL_bashScript()
    >>> dplInst.makeSubmitScripts()
    >>> exit()


 + "e_postShearFullStack"
 + "f_postShearFullStack"
 + "f_preShearFullStack"
 + "g_preShearFullStack"
 + "a_preShearFullStack"
 + "b_preShearFullStack"
 + "c_preShearFullStack"
 o "d_postFullShearStack"
 o "d_preShearFullStack"
 - "e_preShearFullStack"