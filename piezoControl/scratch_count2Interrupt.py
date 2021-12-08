import time
try:
    from console_thrift import KeyboardInterruptException as KeyboardInterrupt
except ImportError:
    pass

stepDict = dict(a=0.001, b=0.01, c=0.1, d=1)

def count2Interrupt():
    t = 0
    try:
        while True:
            t +=1
            time.sleep(1)
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")
        print(t)
        return t


out = {}
for n,step in stepDict.items():
    print('Starting step {}'.format(step))
    out[step] = count2Interrupt()

