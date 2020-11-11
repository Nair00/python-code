import numpy as np
import math
import gzip

f = gzip.open("samples/train-images-idx3-ubyte.gz", 'rb')
g = gzip.open("samples/train-labels-idx1-ubyte.gz", 'rb')

image_size = 28
num_images = 100

f.read(16)
g.read(8)

buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype = np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size * image_size)

labels = np.zeros((num_images, 10))

for i in range(num_images):
    buff = g.read(1)
    label = np.frombuffer(buff, dtype=np.uint8).astype(np.uint8)
    labels[i, label] = 1
    #print(label)


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.weights1 = np.random.rand(self.input.shape[1], 16)       #(3,4) generates 12 weights between [0,1)
        self.biases1 = np.zeros((16, 1))
        self.weights2 = np.random.rand(16, 16)
        self.biases2 = np.zeros((16, 1))
        self.weights3 = np.random.rand(16, 10)
        self.biases3 = np.zeros((10, 1))
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.biases1.T)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.biases2.T)
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3) + self.biases3.T)
        return self.layer3

    def backprop(self):
        d_weights3 = np.dot(self.layer2.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
        d_weights1 = np.dot(self.input.T, np.dot(np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1))
        
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3
        '''
        d_biases3 = 2 * (self.y - self.output) * sigmoid_derivative(self.output)
        d_biases2 = np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)
        d_biases1 = np.dot(np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1)

        self.biases1 += d_biases1
        self.biases2 += d_biases2
        self.biases3 += d_biases3
        '''      

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()

#X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
#y=np.array(([0],[1],[1],[0]), dtype=float)

'''
NN = NeuralNetwork(images,labels)

for i in range(1500): # trains the NN 1,500 times
    if i % 100 == 0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")

    NN.train(X,y)

'''
'''
import matplotlib.pyplot as plt
image = np.asarray(data[0]).squeeze()
plt.imshow(image)
plt.show()
'''
#print(labels)

NN = NeuralNetwork(data, labels)

for i in range(1500):
    if i % 100 == 0:
        print("Actual Output: " + str(labels))
        print("Predicted Output: " + str(NN.feedforward()))

    NN.train(data, labels)