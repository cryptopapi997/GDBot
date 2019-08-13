import numpy as np
import pyscreenshot
import cv2


def screen_record(): 
    while(True):
        # 550x600 (size of GD without the things behind the cube being recorded)
        printscreen =  np.array(pyscreenshot.grab(bbox=(250,40,800,640)))
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        #press q to exit screen recording
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
