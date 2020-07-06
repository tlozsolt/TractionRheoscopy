import glob
"""
This file simply takes a location file in pixels and converts to xyz assuming x,y,z coordinates are $1,$2,$3
The hope is that this will save time with the standard awk oneliner I have been using 
"""

def location2xyz(file,px2Micron = [0.115,0.115,0.15]):
    """
    Reads in the file, and multiplies the first three columns by px2Micron
    and write a new file in xyz format where the first line is the number of particles
    and the second line is comment of blank, and the rest of hte lines are the xyz positons
    now in microns.
    It save the file to the same directory with the same root filenmae of _micron.xyz appended

    :param file:
    :return: path to saved file
    """
    lineCount = 0
    outStr = ''
    for line in open(file).readlines():
        lineFloat = [float(field) for field in line.split('\t')]
        outStr += 'X {x} {y} {z}\n'.format(x=lineFloat[0]*px2Micron[0],\
                                                    y=lineFloat[1]*px2Micron[1],\
                                                    z=lineFloat[2]*px2Micron[2])
        lineCount += 1
        #print("lineCount is: {}".format(lineCount))
    outStr = str(lineCount)+'\n\n'+ outStr


    # now write outStr to a file.
    fName_xyz = file.split('.')[0] + '_micron.xyz'
    with open(fName_xyz,'w') as f: f.write(outStr)
    return fName_xyz

if __name__ == "__main__":
    locationPath = '/mnt/serverdata/zsolt/ParticleLocations/tfrGel09102018b_shearRun09232018b/particleLocations/'
    locationFile = 't0035_xyz_coordinates.txt'

    #print(location2xyz(locationPath + locationFile))
    #for fName in glob.glob(locationPath+'*.txt'):
    #    print(location2xyz(fName))

