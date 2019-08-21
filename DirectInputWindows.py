import time
from pynput.mouse import Button, Controller

mouse = Controller()

def bounce():
    mouse.position = (210, 370)
    mouse.click(Button.left, 1)

def restart():
    mouse.position = (210, 440)
    mouse.click(Button.left, 1)

        
