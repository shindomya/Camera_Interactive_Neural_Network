import numpy as np
import time
import os 

timer_start = time.perf_counter()

#training data
def get_data():
    pass

#minimize data
def minimize_data(data):
    return data / 255

#shape vektor 2D -> 1D
def shape2_1(data):
    return data.reshape(-1,784)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1
