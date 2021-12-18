import threading
import time

"""
This script is meant to demonstrate the use of two threads. 
An outer loops iterate through items in dict with values of a list
This list is passed to one thread iterates over a list of unspecified length printing values at regular intervals.
The other thread checks to see which step you are on
"""

state = [0, True]
#stepList = [list(range(3)),list(range(10))]
stepDict = dict(a=list(range(3)), b=list(range(10)))
stepList = [val for val in stepDict.values()]
out = []

def do_every(interval, worker_func, params, dataOut, iterations=0, **kwargs):
    iterations = len(params)
    if iterations != 1:
        state[1] = False
        thread = threading.Timer(
            interval,
            do_every, [interval, worker_func, params, dataOut, 0 if iterations == 0 else iterations - 1]
        )
        thread.start()
        #print(iterations)
    else:
        state[1] = True
        state[0] += 1
    worker_func(params, dataOut)

def printPop(l: list, out: list):
    tmp = l.pop()
    out.append(tmp)
    print(tmp)

def checkState()-> bool:
    return state[1]

def startNewStep():
    if checkState():
        n = input('Enter step to start given current step is {}:'.format(state[0]))
        posList = stepList[int(n)]
        do_every(1,printPop,posList, out)

class shear(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self.event = threading.Event()
        self.posList = list(range(10))
        self.startHold = False

    def run(self):
        while not self.event.is_set():
            if len(self.posList) == 0:
                self.event.set()
                print("completed shear")
                self.startHold = True
                break
            else:
                print("shear is running")
                print(self.posList.pop())
                self.event.wait(1)

class hold(threading.Thread):
    def __init__(self):
        stop_threads = False
        threading.Thread.__init__(self, daemon=True)
        self.event = threading.Event()
        self.holdValue = 40
        self.startHold = False

    def run(self, shearThread):
        while not self.event.is_set():
            self.startHold = shearThread.startHold
            if self.startHold:
                print("starting hold at value {}".format(self.holdValue))
                self.event.wait(5)
            else: self.event.wait(0.1)

class query(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True )
        self.event = threading.Event()
        #self.holdValue = 40
        self.query = False

    def run(self):
        global stopHold
        while not self.event.is_set():
            self.query = stopHold
            if self.query:
                input('Press enter to kill hold')
                break
            else: self.event.wait(0.1)


class inputThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self.event = threading.Event()
        self.endHold = False

    def run(self):
        while not self.event.is_set():
            se3

    def run(self):
        while not self.event.is_set():
            global stopThread
            stopThread = not bool(input('Press enter to end thread'))


def main():
    s = shear()
    h = hold()
    q = query()
    #q.run(h)
    s.start()
    q.start()
    h.run(s)
    #stop_threads = not bool(input('Press enter to kill thread'))
    s.join()
    h.join()
    q.join()

    #try:
    #    while True:
    #        time.sleep(0.5)

    #except KeyboardInterrupt:
    #    print("Closing threads")
    #    s.join()
    #    h.join()

if __name__ == '__main__':
    main()

# create one thread that checks whether a step is being run say every ten seconds
# if thread is not being run: start a hold pattern and raise a prompt to continue.
# #

#do_every(10,startNewStep(),[],[])



#for step, posList in stepList.items():
#    #print(step)
#    do_every(1, printPop, posList,out)
#    input('Press enter to save step to file')
#    input('Press enter to start next step')
