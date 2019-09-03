#Simulates keypresses on Mac
from pynput.mouse import Button, Controller

mouse = Controller()

#Bounce the cube by clicking on the screen
def bounce():
    mouse.position = (180, 350)
    mouse.click(Button.left, 1)

#Restart the level by clicking on the restart button
def restart():
    mouse.position = (180, 425)
    mouse.click(Button.left, 1)
