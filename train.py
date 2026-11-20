import numpy as np
import time
import os 
import cv2

timer_start = time.perf_counter()

#training data
def get_data():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("Webcam",frame)
        
        last_frame = frame

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return last_frame

def mirror(img):
    h,w = img.shape[:2]
    M_horiz = np.float32([
        [-1, 0, w-1],
        [0,1,0]
    ])
    mirrored_h = cv2.warpAffine(img, M_horiz, (w,h))
    return mirrored_h

def show_image(mirrored_frame):
    cv2.imshow("Mirrored Frame",mirrored_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    


#minimize data
def minimize_data(data):
    return data / 255

frame = get_data()
mirrored_frame = mirror(frame)

show_image(mirrored_frame)

timer_end = time.perf_counter()
print(f"Code execution took: {timer_end-timer_start:.4f} seconds")

