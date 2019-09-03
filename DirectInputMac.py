#Simulates keypresses on Mac
from Quartz.CoreGraphics import CGEventCreateKeyboardEvent
from Quartz.CoreGraphics import CGEventPost
from Quartz.CoreGraphics import kCGHIDEventTap
import time

#Press the up button to bounce the cube
def bounce():
    CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, 0x7E, True))
    time.sleep(0.05)
    CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, 0x7E, False))

#Press the space bar to restart the level
def restart():
    CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, 0x31, True))
    time.sleep(0.05)
    CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, 0x31, False))
