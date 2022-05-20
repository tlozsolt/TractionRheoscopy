import pandas as pd
import xml.etree.ElementTree as ET
import seaborn as sns
from matplotlib import pyplot as plt

def xml2Pandas(xmlRheoPath: str, wdir: str = None):
    """
    Converts exported xml output from AresRheometer to pandas dataFrame.
    Currently ignores all of the metaData keys and only processes xml tags that begin
    with 'ExperimentalData'


    ToDo: Add ability to recognize if step is curing, freq sweep, or strain sweep
    ToDo: export additional or extended data column including axial strain
    ToDo: Add function to loop over all the xml files in a directory
    ToDO: Possibly parse separate files on the same sample
          -> ie pdps947_03012021 (1).xml is the same sample as pdps947_03102021.xml
          Although, this is likely best done manually. Afterall there arent that many rheology calibration
          steps
    ToDo: Parse metaData information (also probably best done manually.
    ToDo: Add labnotebook information to metaData
    ToDo: Save a full rheological calibration dataFrame (metaData and steps) to a pickle object or
          some other output.

     --Zsolt, Jan 5 ,2022
    """
    if wdir is None: wdir = '/Volumes/PROJECT/Rheology/25122021_Rheology_sp3_sp6/xml_export'

    # open the file
    tree = ET.parse(wdir+xmlRheoPath)
    root =  tree.getroot()

    dfDict = {}
    for n, val in enumerate(root):
        # skip over metaData for now
        if val.tag == 'ExperimentalData':
            exptKey = 'Step {}'.format(n -2)
            out = []
            #out[exptKey] = []
            #stepData = out[exptKey]
            colLabel = []
            for n, dataPt in enumerate(val):
                data = []
                for nCol, val in enumerate(dataPt):
                    if n == 0: colLabel.append(val.tag + ' ({})'.format(val.attrib['unit']))
                    data.append(float("".join(val.itertext())))
                out.append(data)
            dfDict[exptKey] = pd.DataFrame(out, columns=colLabel)
    return dfDict

if __name__ == '__main__':
    wdir = '/Volumes/PROJECT/Rheology/25122021_Rheology_sp3_sp6/xml_export'
    fName = '/test.xml'
    """
    This required significant editing, but I am optimistic that the majority can be automated. 
    First, I parsed the raw file using online program: https://jsonformatter.org/xml-parser
    Second I deleted the inital lines to make the xml file flatter
        I deleted the keys Source. Sample, Equipment, ExperimentalData
        and kept only RheoML_DataSet and DMA. With these changes, the file resembled the flat structure given in pandas 
        doc
    Third, open the file using the encoding 'utf-8-sig' which I think lets the program know that there is embedded
        encoding in the file. Without this, the xml parser threw an error as the first character it encounterd was the
        xml start char '<'
    Finally, import only using default parameters. 
    
    I didnt specify xpath, although I think taht could be used to
    skip the second step, or if the file contains multiple steps ie ExperpimentalData has itself multiple rows.
    Maybe the key structure should be walked and then passed as xml to pandas to parse when it encounters ExperimentalData
    row. This would create a dictionary of pandas dataFrames for each experimental step.
    """
    with open(wdir + fName, mode='r', encoding='utf-8-sig') as f:
        #test = pd.read_xml(f, xpath='./RheoML_Dataset/ExperimentalData/', parser='lxml' )
        test = pd.read_xml(f, parser='lxml' )
    test.head()


    raw = '/pdps_952_23 (1).xml'
    dfDict = xml2Pandas(raw, wdir=wdir)

    # plot freq sweep on log-log scale
    # ToDo: set ticks on white grid
    # ToDo: plot strain sweep
    # ToDo: Add legend
    # ToDo: Add measurement points to show sampling.
    sns.set_style('whitegrid')
    g = sns.lineplot(data=dfDict['Step 2'],x = 'Frequency (Hz)', y='LossModulus (Pa)')
    g = sns.lineplot(data=dfDict['Step 2'],x = 'Frequency (Hz)', y='StorageModulus (Pa)')
    g.set(xscale='log', yscale='log')
    plt.show()


    """
    tree = ET.parse(wdir + raw)
    root = tree.getroot()

    # walk the raw xml structure and create a dict to pass to pandas
    # syntax is root[step + 2][data entry][column index]
    step = 1
    t = 100

    out = []
    colLabel = []
    for n, dataPt in enumerate(root[step+2]):
        data = []
        for nCol, val in enumerate(dataPt):
            if n == 0: colLabel.append(val.tag + ' ({})'.format(val.attrib['unit']))
            data.append(float("".join(val.itertext())))
            out.append(data)
    df = pd.DataFrame(out,columns=colLabel)

    #colLabel = root[step + 2][t][0].tag + ' ({})'.format(root[3][100][0].attrib['unit'])
    #dataPt = float("".join(root[3][t][0].itertext()))
    """