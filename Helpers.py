#Helper functions
import numpy as np
import pyscreenshot
import cv2
from skimage.measure import compare_ssim

# Gets one frame
def get_screen():
    # 470x410 (size of GD without the things
    # behind the cube being recorded and the top being cut off)
    screen =  np.array(pyscreenshot.grab(bbox=(150, 40, 620, 450)))
    # Simplify image
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    gray_screen = cv2.Canny(gray_screen, threshold1 = 200, threshold2=300)
    return gray_screen

# Compares two following images and returns a boolean for alive. If the image is the "Restart?"
# screen, structural similarity index will be 0.99+ which means the cube is dead. Else, it's alive.
def isalive(screen1, screen2):
    (score, diff) = compare_ssim(screen1, screen2, full=True)
    if(score < 0.995):
        return True
    else:
        return False

# Records and displays the screen
def screen_record():
    while(True):
        gray_printscreen = get_screen()
        cv2.imshow('window',gray_printscreen)
        #press q to exit screen recording
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
