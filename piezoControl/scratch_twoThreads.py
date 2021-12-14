import threading
import schedule

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

# create one thread that checks whether a step is being run say every ten seconds
# if thread is not being run: start a hold pattern and raise a prompt to continue.
#

do_every(10,startNewStep(),[],[])



#for step, posList in stepList.items():
#    #print(step)
#    do_every(1, printPop, posList,out)
#    input('Press enter to save step to file')
#    input('Press enter to start next step')
