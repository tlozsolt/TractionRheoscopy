from pynput import keyboard
import threading
import scratch_twoThreads as twoThreads

"""
This script demonstrates the use of keyboard keystrokes detection within a thread that is constantly monitored.
The keyboard.listener takes the place of main() in scratch_twoThreads
"""
from pynput import keyboard

def on_press(key):
    global stopThread
    try:
        if key.char == 'n':
            print("moving to next step")
            #twoThreads.main()
            s = twoThreads.shear()
            h = twoThreads.hold()
            s.start()
            h.run(s)

        elif key.char == 'p':
            print("pausing")
        elif key.char == 's':
            print("setting global var stopThread to True")
            stopThread=True
        elif key.char == 'i': print("getting current pos")
        else: print('Alphanumeric key pressed: {0}, and this doesnt have a function'.format(key.char))
    except AttributeError:
        print('special key pressed: {0}'.format(
            key))

def on_release(key):
    print('Key released: {0}'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

#listener = keyboard.Listener(
#    on_press=on_press,
#    on_release=on_release)
#listener.start()

