import numpy as np
import pyscreenshot
import cv2
from bouncemac import bounce


def screen_record(): 
    while(True):
        bounce()
        # 550x600 (size of GD without the things behind the cube being recorded)
        printscreen =  np.array(pyscreenshot.grab(bbox=(250,40,800,640)))
        # simplify image
        gray_printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        gray_printscreen = cv2.Canny(gray_printscreen, threshold1 = 200, threshold2=300)
        cv2.imshow('window',gray_printscreen)
        #press q to exit screen recording
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
