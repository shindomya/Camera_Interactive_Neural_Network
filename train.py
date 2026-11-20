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
#minimize data
def minimize_data(data):
    return data / 255

def ask_solution():
    print("what's the solution?")
    return np.array([int(input())])

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

class Loss_CategoricalCrossentropy:
    def calculate(self,y_pred,y_solution):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        correct_confidences = y_pred_clipped[range(samples),y_solution]
        neg_log = -np.log(correct_confidences)
        return np.mean(neg_log)
    def one_hot_matrix(self,labels):
        #initialize a matrix with 0 only
        clean_matrix = np.zeros((len(labels),10))
        #put the correct answers into the matrix (1=answer,0=everything else)
        clean_matrix[np.arange(len(labels)), labels] = 1
        return clean_matrix
    def backward(self,output_value,clean_matrix,division):
        dvalue = (output_value - clean_matrix) / division
        return dvalue

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

cce = Loss_CategoricalCrossentropy()

solution = ask_solution()

if os.path.exists("mnist_weights.npz"):
    saved = np.load("mnist_weights.npz")

    layer1.weights = saved["weights1"]
    layer1.biases = saved["biases1"]

    layer2.weights = saved["weights2"]
    layer2.biases = saved["biases2"]

learning_rate = 0.01
for epoche in range(50): 
    for i in range(1):
        layer1.forward(frame6)
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        loss = cce.calculate(activation2.output,solution)
        matrix_solution = cce.one_hot_matrix(solution)
    
        dvalues = cce.backward(activation2.output,matrix_solution,1)
    
        layer2.backward(dvalues)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        layer1.weights -= learning_rate * layer1.dweights
        layer1.biases -= learning_rate * layer1.dbiases

        layer2.weights -= learning_rate * layer2.dweights
        layer2.biases -= learning_rate * layer2.dbiases
    print(f"loss {epoche}: {loss}")
    np.savez(
            "mnist_weights.npz",
            weights1 = layer1.weights,
            biases1 = layer1.biases,
            weights2 = layer2.weights,
            biases2 = layer2.biases
            )
        
timer_end = time.perf_counter()
print(f"Code execution took: {timer_end-timer_start:.4f} seconds")

