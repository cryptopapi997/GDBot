import numpy as np
import pyscreenshot
import cv2
from directinputmac import bounce, restart
import time
from skimage.measure import compare_ssim

def screen_record():
    alive = True
    i = 0
    gray_printscreen_2 =  None
    while(True):
        # 550x600 (size of GD without the things behind the cube being recorded)
        printscreen =  np.array(pyscreenshot.grab(bbox=(250,40,800,640)))
        # simplify image
        gray_printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        gray_printscreen = cv2.Canny(gray_printscreen, threshold1 = 200, threshold2=300)

        # Checks if dead every 7 frames
        if (i % 3 == 0):
            gray_printscreen_2 = gray_printscreen
        if(i % 7 == 0 and i % 3 != 0):
            alive = isalive(gray_printscreen, gray_printscreen_2)
        i = i +1
        cv2.imshow('window',gray_printscreen)
        #press q to exit screen recording
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# compares two images and returns alive or dead. If the image is the "restart?"
# screen, score will be 0.99+ which means the cube is dead. Else, it's alive
def isalive(screen1, screen2):
    start_time = time.time()
    (score, diff) = compare_ssim(screen1, screen2, full=True)
    if(score < 0.995):
        print("Alive: " +str(score))
        return True
    else:      
        print("Dead " + str(score))
        #preliminary scoring for now
        score_this_round = time.time() - start_time
        restart()
        return False

screen_record()

