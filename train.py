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
    inverted_frame = invert(frame)
    return inverted_frame()

#minimize data
def minimize_data(data):
    return data / 255

inverted_frame = get_data()

new_data = minimize_data(inverted_frame)
print(new_data)


timer_end = time.perf_counter()
print(f"Code execution took: {timer_end-timer_start:.4f} seconds")
#rawr
