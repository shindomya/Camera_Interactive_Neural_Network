import numpy as np
import time
import os
import cv2

#start timer
start = time.perf_counter()

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

def show_image(frame):
    cv2.imshow("Frame",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def invert(mirrored_image):
    inverted_image = cv2.bitwise_not(mirrored_image)
    return inverted_image

def color_gray(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

def scale(frame):
    scaled_frame = cv2.resize(frame, (28,28), interpolation=cv2.INTER_AREA)
    return scaled_frame

def resize(frame):
    resized_frame = frame.reshape(1,784)
    return resized_frame

#minimize pixel value 
def minimize_data(y):
    return y / 255

#shape 2D vektor - 1D vektor
def shape2_1(x):
    x = x.reshape(-1,784)
    return x

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        #0.1 because of the bell-curve (value might go ober 1.0)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
        
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        #deletes all nums under 0
        self.output = np.maximum(0, inputs)
        

class Activation_Softmax:
    def forward(self,inputs):
        #e raised by all values from last output - e raised to highest value from last output
        #axis=1 calculates the rows instead of default None (coloums)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities



frame = get_data()
frame1 = mirror(frame)
frame2 = invert(frame1)
frame3 = color_gray(frame2)
frame4 = scale(frame3)
frame5 = minimize_data(frame4)
frame6 = resize(frame5)

layer1 = Layer_Dense(784, 64)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(64,10)
activation2 = Activation_Softmax()

if os.path.exists("mnist_weights.npz"):
    saved = np.load("mnist_weights.npz")

    layer1.weights = saved["weights1"]
    layer1.biases = saved["biases1"]

    layer2.weights = saved["weights2"]
    layer2.biases = saved["biases2"]


layer1.forward(frame6)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(f"Guessed number: {np.argmax(activation2.output)}")
print(f"Confidence: {np.max(activation2.output) * 100:.2f}%")


end = time.perf_counter()
print(f"Code execution took : {end-start:.4f} seconds")
