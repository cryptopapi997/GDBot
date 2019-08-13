#simulates a keypress to bounce the cube on Mac
from Quartz.CoreGraphics import CGEventCreateKeyboardEvent
from Quartz.CoreGraphics import CGEventPost
from Quartz.CoreGraphics import kCGHIDEventTap
import time

def bounce():
    CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, 0x7E, True))
    time.sleep(0.05)
