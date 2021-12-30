import serial
from datetime import datetime
import time
import threading
import numpy as np
import pandas as pd
import json

"""
Copied to git from Dropbox/SCRIPTS 

Zsolt Dec 6 2021
I think this was called everytime a shear step was made with inst.oscillate()

Started editing from legacy_piezoControlClass
-Zsolt Dec 12 2021

"""
class Piezo(serial.Serial):
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
        #self.path2serial = '/dev/ttyUSB0'  # Linux
        self.path2serial = '/dev/ttyS0'

        # self.baudRate = int(input('Enter the 'baudrate' in quotes Default is '115200': '))
        self.baudRate = 115200

        # self.timeOut = int(input('Enter the desired timeout  duration in seconds: '))
        self.timeOut = 1

        # self.hardWareHandShake = True
        serial.Serial.__init__(self, self.path2serial, self.baudRate, rtscts=1, timeout=self.timeOut)

        self.birth = 'This instance was created at ' + str(datetime.now())
        self.path2microscopy = '/path/to/microscope/images/during/benchmark/routines/'
        self.comments = 'This is string file containing comments \n'

        self.delay = 0.005 # pause in seconds between consecuative writes to peizo
        # we want a class attributbe self.data to contain a dictionary of key:value pairs
        # for every motion set called. This should contain:
        # Comments on what is begin called...possibly text input at command lines to answer 'what is the purpose of this call'
        # function call will full parameter list
        # Date
        # pos: pandas data frame object containing tuples (time in seconds, requested position, actual position)
        self.data = {
            'description': 'this attribute contains a keyed entry for every motion that was called during the lifetime of the instance. The entries are populated using the makeDataEntry method'}

        # job control
        self.stop_threads = False
        self.currentStep = None
        self.movingBool = False # is the piezo currently shearing?
        self.currentPosTuple = None

        # timing and stepSize
        self.repRate = 0.05 # how often should I query the piezo?
        self.stackTime = float(input('How long, in seconds, between sucessive stacks?'))
        self.maxStepPerStack = float(input('What is the max acceptable displacement in um of the shear plate between stacks?'))
        self.maxStep = self.maxStepPerStack/(self.stackTime/self.repRate)

        # deformation paramters?
        self.posList = dict(a=self.ramp(40,50)) # This should be a dictionary of steps created with functions ramp, hold, etc
        self.data = {key:[] for key in self.posList.keys()}

        # logging and filepaths
        self.instName = input('Enter a string for this piezo instance')
        self.dataDir ='/home/zsolt/piezoData/'

    def __call__(self):
        pass

    def listAttr(self):
        """ This function should list all class attritubtes that can be pickled or stored in JSON array"""
        return ['path2serial', 'baudRate', 'timeOut', 'birth', 'path2microscopy', 'comments', 'data']

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

    def servoOn(self):
        """ This function turns the servo state to ON and returns state of the servo (== True if on)"""
        self.write(b'SVO A1\n')
        self.write(b'SVO? A\n')
        return bool(self.readline())

    def getPos(self, output='tupleFloat'):
        """ This function returns the current position of the piezoObject in microns """

        self.write(b'POS? A\n')  # the b prefix means 'convert to bytes' before sending to serial
        actualPos = self.readline()
        now = datetime.now()
        self.pause()

        self.write(b'MOV? A\n')  # convert to bytes then pass string
        targetPos = self.readline()
        self.pause()

        self.currentPosTuple = (float(actualPos), float(targetPos), now)

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
        self.pause()
        return self.getPos()

    def pause(self, seconds: float = None):
        """ This function simply pauses its a certain amount of time in seconds before finishing.
            By default it is used to pause a millisecond or so to no overload the communication rate to the piezo
            but could be extended to any pause.
            It does not communicate with the piezo.
        """
        if seconds is None: seconds = self.delay
        time.sleep(seconds)

    def runStep(self, stopFn, step: str):
        """
        ToDo:
        + modify function to call mov
        - modify function, probably by calling a different logging function, to log the output as opposed to printing
          to screen
        - initialize posList within the class so that it scans over steps
        - when should posList be created? Should be loaded from yaml? Or created from param passed to yaml?
        - note that runStep should only run a single step, and self.stepList is probably the full deformation sequence
        + the first step in clearing up the ambiguity has been done by creating step parameter that indexes (probably
          with kwarg into posList to retrieve the step
        - this construction may somehow fail as now runStep will be threaded and takes arguements other than stopFn...
          it should work, but I may not know how to do it.
        """
        self.currentStep = step
        self.movingBool = True

        stepList = self.posList[step]

        holdPt = stepList[-1]
        startTime = time.time()  # this time should be started when the shear is started.
        while len(stepList) > 0:
            #print('shearing {}'.format(stepList.pop()))
            #print(datetime.now())
            #time.sleep(random.uniform(0, 1) / 10)  # sleep for upto 100 ms to test the locking to system clock
            self.mov(stepList.pop(0)) # this will update self.currentPosTuple during the embedded call to self.getPos()
            self.data[step].append(self.currentPosTuple)
            elapsedTime = time.time() - startTime
            time.sleep((self.repRate - elapsedTime) % (self.repRate))
            if len(stepList) == 0:
                print('Step {} complete! Holding at pos {}'.format(self.currentStep, holdPt))
                self.log()
                self.movingBool = False
        while len(stepList) == 0:
            #if not datetime.now().second % 10:
            #    print('Hold at pos {}'.format(holdPt))
            self.mov(holdPt)
            time.sleep(self.repRate)
            if stopFn(): break

    def query(self):
        #global stop_threads
        i = input(' next: go to next step\n info: print info \n log: print pos and log if not shearing\n')
        if i == 'n':
            self.stop_threads = True
        elif i == 'info':
            #print('here is some info, probably from the class')
            #print('stopThreads is {}'.format(stop_threads))
            print('number of active threads: {}'.format(threading.active_count()))
            #try:
            #    print('Current target position of the piezo: {}'. format(self.posList[self.currentStep[0]]))
            #except IndexError:
            #    print('Current step is completed, and current holding')
            print('Current pos tuple is: {}'.format(self.currentPosTuple))
            print('Current step is: {}'.format(self.currentStep))
            print('The piezo is currently in the middle of a step? {}'.format(self.movingBool))
            print('\n\n')
            self.query()
        elif i =='log':
            #print(self.data[self.currentStep])
            print(self.currentPosTuple)
            print('\n\n')
            if not self.movingBool : self.log(self.currentStep)
            self.query()
        else:
            print('input not recognized, try again')
            self.query()

    def log(self, step: str = None):
        """
        Writes a pandas dataFrame of posList to file f
        """
        if step is None: steps = self.data.keys()
        else: steps = [step]
        for step in steps:
            if len(self.data[step]) == 0: pass
            else:
                df = pd.DataFrame(self.data[step],
                                  columns=['actual pos (um)', 'target pos (um)', 'time (datetime obj)'])
                t0 = df['time (datetime obj)'][0]
                df['elapsed time (s)'] = df['time (datetime obj)'].apply(lambda x: (x - t0).total_seconds())
                df.to_hdf(self.dataDir + '/{}_step{}_piezoData.h5'.format(self.instName, step), 'piezoData')

    def takeStep(self, step: str):
        #global stop_threads
        # shearTmp is a function of one variable 'stop'.
        # step is fixed parameter that is passed when shearTmp is created. It is not callable with shearTmp
        shearTmp = lambda stop: self.runStep(stop, step)
        t1 = threading.Thread(target=shearTmp, args=(lambda: self.stop_threads,))
        t1.start()
        self.query()
        # stop_threads = not bool(input('Press enter to kill thread'))
        t1.join()
        print('step finished killed')

    def ramp(self, start: float, stop: float,
             time: float = None,
             endPt: bool = False):
        """
        Creates list of position that constitute a ramp from start to stop in time t.
        Makes sure that step size is at most self.maxStep and assumes a repRate of self.repRate
        Does not move the piezo. Only creates a list of values to move through
        """
        if time is not None: step = min(self.maxStep, (start-stop)/(time/self.repRate))
        else: step = self.maxStep

        if endPt: return list(np.arange(start,stop + step, step))
        else: return list(np.arange(start,stop, step))

    def triangle(self, start: float, stop: float, time: float = None):
        if time is not None: time /= 2
        up = self.ramp(start,stop,time, endPt=True)
        down = self.ramp(start, stop, time, endPt=False)
        down.reverse()
        return up + down

    def hold(self, position: float, time: float):
        hold = np.empty(time/self.repRate, dtype=float)
        hold[:] = position
        return  list(hold)

    def upHoldDown(self, start: float, stop: float, holdTime: float):
        up = self.ramp(start,stop, endPt=False)

        hold = self.hold(stop,holdTime)

        down = self.ramp(start,stop,endPt=False)
        down.reverse()

        return up + hold + down

    def bumpPauseBump(self, start: float, stop:float, pause: float):
        bump = self.triangle(start,stop)
        hold = self.hold(start,pause)
        return bump + hold + bump

    def benchmark(self):
        """ This function carries out a battery of three benchmark tests
        1) Histogram of thermal drift: gather 1000 positions at a constant position
        and record to a file. This file can be plotted as histogram and the FWHM determined

        2) Three speed linear ramp
        3) Settling time: move to a series of new positions and query the position as fast as possible after the initial motion.
        """
        for step in ['ramp', 'triangle', 'upHoldDown', 'bumpPauseBump']: pass

        return True

    def _movList(self, posList, dataOut={}, **kwargs):
        dataOut['posList'].append(self.mov(posList.pop(0), **kwargs))
        #print(dataOut['posList'][-1])
        return posList


    def _do_every(self, interval, worker_func, params, dataOut, iterations=0, **kwargs):
        if iterations != 1:
            thread = threading.Timer(
                interval,
                self.do_every, [interval, worker_func, params, dataOut, 0 if iterations == 0 else iterations - 1]
            )
            thread.start()
            #print(iterations)
        worker_func(params, dataOut)



    def _oscillate(self, cycles=1, a=0.1, b=40, A=60, velocity=2, **kwargs):
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

    def _triangleHalfWave(self, cycles=1, a=0.1, b=0, A=20, velocity=2, **kwargs):
        """ This function carries out a 20um  amplitude triangle bump  at fixed speed of 2 um/s
        for a specified number of cycles.
        Each cycle takes 1 minute to complete using the default parameters.
        This function calls the piezo and carries out the oscillation and generates an output list of the results

        Return: list of triples giving (time,position,deviation from requested position)
        Keywords
        cycles: integer specifying the number of cycles
        a: float giving the step size in microns (default 0.1)
        b: float giving the absolute starting position in microns of the piezo. the default is 0 um
        A: float specifying the amplitude in mircons of the triangle bump (default is 20 um)
        velocity: float specifying the velocity of the imposed motion in um/s (default is 2 um/s)
        """
        # Make the ramp up and down
        posList1 = [round((a * x + b), 3) for x in range(int(A / a))]
        posList2 = [round((-a * x + b + A), 3) for x in range(int(A / a + 1))]
        # posList3 = [round((a*x +b-A/2),3) for x in range(int(A/(a*2.0))) ]
        # posList = cycles*(posList1 + posList2 + posList3) #concatenate and multiply by number of cycles
        posList = cycles * (posList1 + posList2)  # concatenate and multiply by number of cycles
        interval = float(a / velocity)
        name = self.makeDataEntry()
        self.data[name]['functionCall'] = self.triangleHalfWave.__doc__ + '\n' + 'cycles: ' + str(cycles) + 'a: ' + str(
            a) + '\n b:' + str(b) + '\n A:' + str(A) + '\n velocity: ' + str(velocity) + '\n'
        self.do_every(interval, self.movList, posList, self.data[name], len(posList))
        return name

    def _strainSweep(self, gap, cycles=2, pause=15, strainList=[0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.30],
                    strainRate=0.00001):
        """
        This function carries out a strain sweep going through the following hard coded strains:
	    0.1, 1, 2, 5, 10, 15, 30 percent
        going through 2 cycles at each strain and pausing for 15 min between cycles
        at strain rates dictated by the velocity v (\dot{\gamma} = velocity/gap)
        total run time is sum_{strains} cycles*strain/strainRate*4
        """
        for strain in strainList:
            peak2troughAmplitude = 2 * gap * strain
            v = strainRate * gap
            for i in range(cycles):
                self.oscillate(self, cycles=1, a=0.01, b=40, A=peak2troughAmplitude, velocity=v)
                self.pause(pause * 60 * 15)

    def _strainCycleNoOscillate(self, gap=100, strain=0.1, time=3600):
        """ This function carries out a strain out and back, with no sign reversal.
            The strain rate is set by the strain and the time.
        """
        # Compute the amplitude and velocity of the top plate
        amplitude = gap * strain
        velocity = 2 * amplitude / time  # so that it goes out and back in the allotted time
        strainRate = velocity / gap

        a = velocity / 3  # so that the piezo takes 3 steps every second
        b = 40  # start position is middle of the range

        posList1 = [round((a * x + b), 3) for x in range(int(amplitude / a))]  # poslist for going out
        posList2 = [round((-a * x + b + amplitude), 3) for x in range(int(amplitude / a))]
        posList = posList1 + posList2
        name = self.makeDpwd
        self.makeDataEntry()
        self.data[name]['functionCall'] = self.strainCycleNoOscillate.__doc__ + '\n' + 'gap: ' + str(
            gap) + 'strain: ' + str(strain) + '\n time:' + str(time) + '\n amplitude:' + str(
            amplitude) + '\n strainRate: ' + str(strainRate) + '\n'
        self.do_every(interval, self.movList, posList, self.data[name], len(posList))
        return name


if __name__ == '__main__':

    p = Piezo()
    print('initialized')