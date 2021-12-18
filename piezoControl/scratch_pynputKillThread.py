from pynput import keyboard
from scratch_pynputKeyboard import *

"""
The purpose of this script is to start some background task and then
kill it with a key press
The background task should just either print to the screen (if that can be make non-blocking)
or write to a file
"""

listener = keyboard.Listener(on_press=on_press, on_release= on_release)
listener.start()

