import numpy as np
import pyscreenshot
import cv2
import time
from skimage.measure import compare_ssim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import platform

if(platform.system() == "Windows"):
    from DirectInputWindows import bounce, restart
else:
    from DirectInputMac import bounce, restart


# Gets one frame
def get_screen():
    # 550x600 (size of GD without the things behind the cube being recorded)
    screen =  np.array(pyscreenshot.grab(bbox=(250,40,800,640)))
    # simplify image
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    gray_screen = cv2.Canny(gray_screen, threshold1 = 200, threshold2=300)
    return gray_screen

# Records the screen and displays it
def screen_record():
    alive = True
    i = 0
    gray_printscreen_2 =  None
    start_time = time.time()
    while(True):
        
        gray_printscreen = get_screen()
        
        # Checks if dead every 7 frames
        if (i % 3 == 0):
            gray_printscreen_2 = gray_printscreen
        if(i % 7 == 0 and i % 3 != 0):
            #checks if cube is alive, resets score if it isn't
            if(not isalive(gray_printscreen, gray_printscreen_2, start_time)):
                start_time = time.time()
        i = i +1
        cv2.imshow('window',gray_printscreen)
        #press q to exit screen recording
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# compares two images and returns alive or dead. If the image is the "restart?"
# screen, score will be 0.99+ which means the cube is dead. Else, it's alive
def isalive(screen1, screen2,start_time):
    (score, diff) = compare_ssim(screen1, screen2, full=True)
    if(score < 0.995):
        return True
    else:
        #preliminary scoring for now
        score_this_round = time.time() - start_time    
        print("Dead " + str(score_this_round))
        restart()
        start_time = time.time()
        return False

screen_record()

