import serial
from datetime import datetime
import time
import threading
import json
"""
Copied to git from DropBox on Dec 6 2021
This is likely the script and Class used to carry out shear experiments prior to Dec 2021
Zsolt Dec 6 2021
"""

class piezoSerial(serial.Serial):
    """ This class contains methods and attributed required to control
        a linear piezo drive in closed loop operation with serial
        commands.
        The methods include:
        self.mov(float) which moves the piezo to the absolute position
            given by float in um
        self.whoAmI(): returns a ID number, and date/time of instantiation
            along with any comments given in self.comments()
        self.benchmark(): run through a series of benchmarking operations

    """

    def __init__(self):
        # Read in the path to the serial port and assign it to self.path2Serial()
        # self.path2serial = input('Enter the 'full path' (in quotes) to serial port connected to piezo:')
        # self.path2serial = '/dev/tty.usbserial' # Mac
        self.path2serial = '/dev/ttyUSB0'  # Linux

        # self.baudRate = int(input('Enter the 'baudrate' in quotes Default is '115200': '))
        self.baudRate = 115200

        # self.timeOut = int(input('Enter the desired timeout  duration in seconds: '))
        self.timeOut = 1

        # self.hardWareHandShake = True
        serial.Serial.__init__(self, self.path2serial, self.baudRate, rtscts=1, timeout=self.timeOut)

        self.birth = 'This instance was created at ' + str(datetime.now())
        self.path2microscopy = '/path/to/microscope/images/during/benchmark/routines/'
        self.comments = 'This is string file containing comments \n'
        # we want a class attributbe self.data to contain a dictionary of key:value pairs
        # for every motion set called. This should contain:
        # Comments on what is begin called...possibly text input at command lines to answer 'what is the purpose of this call'
        # function call will full parameter list
        # Date
        # pos: pandas data frame object containing tuples (time in seconds, requested position, actual position)
        self.data = {
            'description': 'this attribute contains a keyed entry for every motion that was called during the lifetime of the instance. The entries are populated using the makeDataEntry method'}

    def listAttr(self):
        """ This function should list all class attritubtes that can be pickled or stored in JSON array"""
        return ['path2serial', 'baudRate', 'timeOut', 'birth', 'path2microscopy', 'comments', 'data']

    def getPos(self, output='tupleFloat'):
        """ This function returns the current position of the piezoObject in microns """
        self.write(b'POS? A\n')  # the b prefix means 'convert to bytes' before sending to serial
        actualPos = self.readline()
        now = datetime.now()
        self.write(b'MOV? A\n')  # convert to bytes then pass string
        targetPos = self.readline()
        if output == 'tuple':
            return (float(actualPos), float(targetPos), now)
        elif output == 'tupleFloat':
            return (float(actualPos), float(targetPos), 60.0 * now.minute + now.second + (now.microsecond) / 1000000.)
        elif output == 'printData':
            return str(float(actualPos)) + ', ' + str(60.0 * now.minute + now.second + (now.microsecond) / 1000000.)
        else:
            print(self.getPos.__doc__)
            raise NameError('the output type for getPos() is not valid. See piezoSerial.getPos.__doc__')

    def mov(self, newPos, **kwargs):
        """ This function moves the piezo to absolute position 'newPos' and then returns the current position"""
        self.write(bytes('MOV A' + str(newPos) + '\n', 'UTF-8'))  # concatenate, convert to bytes, then write to serial
        return self.getPos(**kwargs)

    def movList(self, posList, dataOut={}, **kwargs):
        dataOut['posList'].append(self.mov(posList.pop(0), **kwargs))
        # print dataOut['posList'][-1]
        return posList

    def servoOn(self):
        """ This function turns the servo state to ON and returns state of the servo (== True if on)"""
        self.write(b'SVO A1\n')
        self.write(b'SVO? A\n')
        return bool(self.readline())

    def makeDataEntry(self, **kwargs):
        """ This function makes a data entry in the form of a dictionary to added to self.data
        It should be run every time a larger series of motions is performed.
        It should store a pandas data frame containing tuples (time in seconds, requested position, actual position)
        It should store the date and time
        It should store a full programming description of what function was called
        in addition to a full written description of 'why the function was called' which should be added
        possibly on the command line before the motion is executed.
        At the end of the day, I need to have enough information stored in self.data and other attributes
        to 1) replot whatever I'd like, using whatever program I'd like (but i will probably have to write code to handle import etc)
        2) connect the serial commands and outputs to microscopy images of what is going on in the sample
        3) be able to convincingly answer any low level information about piezo position, velocity, and any associated uncertainties
        4) Be able to connect serial output with whatever i was thinking at the time of running the piezo...
             ie, "I am oscilating the piezo in order to align the grid. There are no particles currently in the sample."
             or "I am carrying out a SRFS deformation of STAS particle colloidal glass which was initially prepared on XX with comments in SP2:63, M1:23
             and previous calibration and alignment  on this same sample is  contained in <this other self.data entry>  or possibly this other
             piezoSerial instance stored at <path/to/other/piezoSerial/instance/>
        """
        name = input('Enter a keyword to name this motion as if this was a filename.')
        dataEntry = {}
        dataEntry['comments'] = input('What motion of the piezo are you calling and what is the purpose?: ')
        # dataEntry['functionCall'] = input('What method are you calling? ')
        dataEntry['date'] = str(datetime.now())
        dataEntry['posList'] = [self.getPos(**kwargs)]
        self.data[name] = dataEntry
        return name

    def do_every(self, interval, worker_func, params, dataOut, iterations=0, **kwargs):
        if iterations != 1:
            thread = threading.Timer(
                interval,
                self.do_every, [interval, worker_func, params, dataOut, 0 if iterations == 0 else iterations - 1]
            )
            thread.start()
        worker_func(params, dataOut)

    def oscillate(self, cycles=1, a=0.1, b=40, A=60, velocity=2, **kwargs):
        """ This function carries out a 60um (+/- 30um) amplitude linear ramp at fixed speed of 2 um/s
        for a specified number of cycles.
        Each cycle takes 1 minute to complete using the default parameters.
        This function calls the piezo and carries out the oscillation and generates an output list of the results

        Return: list of triples giving (time,position,deviation from requested position)
        Keywords
        cycles: integer specifying the number of cycles
        a: float giving the step size in microns (default 0.1)
        b: float giving the absolute starting position in microns of the piezo. the default is 40um, ie the center of the 80um maximal stroke
        A: float specifying the amplitude in mircons of the oscillation measured peak to trough. (default is 60 um)
        velocity: float specifying the velocity of the imposed motion in um/s (default is 2 um/s)
        """
        # Make the ramp up and down
        posList1 = [round((a * x + b), 3) for x in range(int(A / (a * 2.0)))]
        posList2 = [round((-a * x + b + A / 2), 3) for x in range(int(A / (a * 1.0)))]
        posList3 = [round((a * x + b - A / 2), 3) for x in range(int(A / (a * 2.0)))]
        posList = cycles * (posList1 + posList2 + posList3)  # concatenate and multiply by number of cycles
        interval = float(a / velocity)
        name = self.makeDataEntry()
        self.data[name]['functionCall'] = self.oscillate.__doc__ + '\n' + 'cycles: ' + str(cycles) + 'a: ' + str(
            a) + '\n b:' + str(b) + '\n A:' + str(A) + '\n velocity: ' + str(velocity) + '\n'
        self.do_every(interval, self.movList, posList, self.data[name], len(posList))
        return name

    def benchmark(self):
        """ This function carries out a battery of three benchmark tests
        1) Histogram of thermal drift: gather 1000 positions at a constant position
        and record to a file. This file can be plotted as histogram and the FWHM determined

        2) Three speed linear ramp
        3) Settling time: move to a series of new positions and query the position as fast as possible after the initial motion.
        """
        return True


##### Code to test class ###

piezo = piezoSerial()
"""
print piezo.name
print piezo.birth
print piezo.path2microscopy
piezo.path2microscopy = '/new/path'
print piezo.path2microscopy
piezo.path2microscopy = piezo.path2microscopy + '/with/appended/directory/'
print piezo.path2microscopy
piezo.comments = piezo.comments + 'This is the second line \n'
print piezo.comments
piezo.write('*IDN? \n')
print('Master unit version number is: ' + piezo.readline())
print(piezo.name)
piezo.write('*IDN?\n')
print('Master unit version number is: ' + piezo.readline())

print('The piezo servo is on?: ' + str(piezo.servoOn()))
"""
piezo.servoOn()
print("The position of the piezo is: " + str(piezo.getPos(output='tupleFloat')))
piezo.mov(40)
print("The position of the piezo is: " + str(piezo.getPos(output='tupleFloat')))
print(threading.active_count())
print(piezo.listAttr())
print(piezo)
initPos = piezo.getPos()
print("start position is: " + str(initPos))
dictKey = piezo.oscillate(A=70, cycles=5, b=40, a=0.1,
                          velocity=0.5)  # all position output is saved to self.data[dictKey]
# for a gap of h= 100 um, a velocity v = 0.01 um/s gives a strain rate \epsilon^\dot = 10E-4, but this estimate is *HIGH* because a part of the impose strain goes into deforming the gel and hence the effective strain and strain rate is lower...unless plasticity sets in.
# Additionally the total time for the deformation is 2*cycles*A/velocity
while threading.active_count() > 1:  # This is an imperfect and indirect test of whether the threaded operations have completed
    print("wait for threaded function to complete")
    print("number of active threads is:" + str(threading.active_count()))
    time.sleep(20)
else:
    name = input('please enter a prefix to use for saving the data')
    f = open(name + '.text', 'w')
    f.write('# time (sec), actual position (um), deviation from requested pos (um)\n')
    for elt in piezo.data[dictKey]['posList']:
        print(elt)
        f.write(str(elt[2]) + ' ' + str(elt[0]) + ' ' + str(elt[1] - elt[0]) + '\n')
    f.close()
    with open(name + '.json', 'w') as g:
        # attach all attributes to new dictionary to be stored in json
        jsonOut = {}
        for attr in piezo.listAttr():
            print(attr)
            jsonOut['piezoSerial.' + attr] = eval('piezo.' + attr)
        json.dump(jsonOut, g)
        g.close()
    print("position data saved to " + name + '.text')
    print("piezoSerial instance pickled to: " + name + '.pickle')