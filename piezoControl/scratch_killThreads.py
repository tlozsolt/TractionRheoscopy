import threading
import time
from datetime import datetime
import random


#def run(stop):
#    while True:
#        print('thread running')
#        time.sleep(3)
#        if stop():
#            break
#def hold(start, stop):
#    while True:
#        if start():
#            print('holding at pos {}'.format(40))
#            time.sleep(3)
#        if stop(): break
repRate = 100 # how quickly to repeat task in milliseconds
def runStep(stop, posList):
    startTime = time.time() # this time should be started when the shear is started.
    while len(posList) > 0:
        print('shearing {}'.format(posList.pop()))
        print(datetime.now())
        time.sleep(random.uniform(0,1)/10)# sleep for upto 100 ms to test the locking to system clock
        elapsedTime = time.time() - startTime
        time.sleep((repRate/1000 - elapsedTime)%(repRate/1000))
        if len(posList) == 0:
            print('Hold at pos {}'.format(40))
    while len(posList) == 0:
        if not datetime.now().second % 10:
            print('Hold at pos {}'.format(40))
        time.sleep(0.1)
        if stop(): break

def query():
    global stop_threads
    i = input('n: next step, info: print info')
    if i =='n': stop_threads =True
    elif i == 'info':
        print('here is some info, probably from the class')
        print('stopThreads is {}'.format(stop_threads))
        print('number of active threads: {}'.format(threading.active_count()))
        print('active threads: {}'.format(threading.enumerate(nf)))
        query()
    else:
        print('input not recognized, try again')
        query()

def main(n):
    global stop_threads
    shearTmp = lambda stop: runStep(stop, list(range(n)))
    t1 = threading.Thread(target=shearTmp, args=(lambda: stop_threads,))
    t1.start()
    query()
    #stop_threads = not bool(input('Press enter to kill thread'))
    t1.join()
    print('step finished killed')


for step in range(3):
    stop_threads = False
    print('starting step {}'.format(step))
    main(step+3)