>>> import sys
>>> sys.path.append('/n/home04/jzterdik/TractionRheoscopy')
>>> from particleLocating import postLocating as pl
>>> dplMetaPath = '/n/home04/jzterdik/TractionRheoscopy/metaDataYAML/tfrGel23042022_shearRun01052022f_imageStack_metaData.yaml'
>>> log = '/n/holylfs02/TRANSFER/jzterdik/PROJECT/tfrGel23042022/strainRamp/f_imageStack/log/'
>>> outliers, binHash, particleCount = pl.test(log, dplMetaPath)
>>> qcDict = {'outliers': outliers, 'binHash': binHash, 'particleCount': particleCount}
>>> with open('/n/holylfs02/TRANSFER/jzterdik/PROJECT/tfrGel23042022/strainRamp/f_imageStack/dpl_quality_control_14JUN2022.pkl','wb') as f: pkl.dump(qcDict, f)
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'pkl' is not defined
>>> import pickle as pkl
>>> with open('/n/holylfs02/TRANSFER/jzterdik/PROJECT/tfrGel23042022/strainRamp/f_imageStack/dpl_quality_control_14JUN2022.pkl','wb') as f: pkl.dump(qcDict, f)
...
>>>