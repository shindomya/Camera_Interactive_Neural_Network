import numpy as np
import time
import os 
import cv2

timer_start = time.perf_counter()
path = "./numb_data_set"
x_list = []
y_list = []

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

def color_gray(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_frame

def pixel_fix(img):
    _, img = cv2.threshold(img, 50,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def center(img):
    coords = cv2.findNonZero(img)
    x,y,w,h = cv2.boundingRect(coords)
    digit = img[y : y + h, x : x + w]

    digit = cv2.resize(digit, (20,20), interpolation=cv2.INTER_AREA)

    centered = cv2.copyMakeBorder(digit,4,4,4,4, cv2.BORDER_CONSTANT, value=0)
    return centered
    
def scale(frame):
    scaled_frame = cv2.resize(frame, (28,28), interpolation=cv2.INTER_AREA)
    return scaled_frame

def resize(frame):
    resized_frame = frame.reshape(1,784)
    return resized_frame

#minimize data
def minimize_data(data):
    return data / 255

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons) * np.sqrt(2.0//n_inputs)
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

for i in range(19):
    img = cv2.imread(f"{path}/{i:02d}.jpg")
    img = color_gray(img)
    img = invert(img)
    img = scale(img)
    img = pixel_fix(img)
    img = center(img)
    img = minimize_data(img)
    img = resize(img)
    x_list.append(img[0])
    y_list.append(i % 10)

x_all = np.array(x_list)
y_all = np.array(y_list)

layer1 = Layer_Dense(784, 64)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(64,10)
activation2 = Activation_Softmax()

cce = Loss_CategoricalCrossentropy()


if os.path.exists("mnist_weights.npz"):
    saved = np.load("mnist_weights.npz")

    layer1.weights = saved["weights1"]
    layer1.biases = saved["biases1"]

    layer2.weights = saved["weights2"]
    layer2.biases = saved["biases2"]

learning_rate = 0.1
batch_size = 10
for epoche in range(500): 
    random_cases = np.random.choice(len(x_all), batch_size,replace=False)

    x_batch = x_all[random_cases]
    y_batch = y_all[random_cases]

    layer1.forward(x_batch)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = cce.calculate(activation2.output,y_batch)
    matrix_solution = cce.one_hot_matrix(y_batch)
    
    dvalues = cce.backward(activation2.output,matrix_solution,batch_size)
    
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

