#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import time
import random
import os , sys


# In[7]:


class myNeuralNet():
    def __init__(self, sizes, epochs = 15, alpha = 0.9, beta = 0.9):
        self.sizes = sizes
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    # use sigmoid activation function
    # we want our values to be between 0 and 1
    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    # obtain the derivative for backpropagating
    def sigmoidDelta(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
   
    def softmaxDelta(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1'  : np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2'  : np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3'  : np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer),
            'b1'  : np.zeros((hidden_1, 1)) * np.sqrt(1. / input_layer),
            'b2'  : np.zeros((hidden_2, 1)) * np.sqrt(1. / hidden_1),
            'b3'  : np.zeros((output_layer, 1)) * np.sqrt(hidden_2)
        }

        return params
    # execute our neural network in two steps, first of which is
    # feed forward network. choose random weights and optimize 
    # with the second step of back propagation
    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2']) 
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    # second step of our neural network. optimizes our weights to
    # decide which of the input feature is vital for our prediction
    # higher weights = greater importance.
    # here we use the error in our prediction to fine-tune our weights.
    # until lower error values. 
    def backward_pass(self, y_train, output):
        params = self.params
        change_w = {}

        # Calculate W3 update
        delta3 = (output - y_train) 
        error = 2 * delta3 / output.shape[0] * self.softmaxDelta(params['Z3'])
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoidDelta(params['Z2']) # get our error
        change_w['W2'] = np.outer(error, params['A1']) # use it to tune our weight

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoidDelta(params['Z1'])
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

   
    # compute the accuracy of the validation tests
    def compute_accuracy(self, x_val, y_val):

        predictions = []
        validations = []

        for x , y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred)
            validations.append(pred == np.argmax(y))
        pd.DataFrame(predictions).to_csv('test_predictions.csv',header=False, index = False)   
        return (np.mean(validations))
    
    
    def train(self, x_train, y_train):
        start_time = time.time()
        params = self.params

        # initialize for momentum to keep a moving 
        # average of our gradients
        grad_w1 = np.zeros(params["W1"].shape)
        grad_w2 = np.zeros(params["W2"].shape)
        grad_w3 = np.zeros(params["W3"].shape)

        for iteration in range(self.epochs):
            
            for x,y in zip(x_train, y_train):
    
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)

                # Momentum step ----------------------------------------
                grad_w1 = (self.beta * grad_w1 + (1. - self.beta) * changes_to_w['W1'])
                grad_w2 = (self.beta * grad_w2 + (1. - self.beta) * changes_to_w['W2'])
                grad_w3 = (self.beta * grad_w3 + (1. - self.beta) * changes_to_w['W3'])

                # STEP 3: update weights/ gradient decent -----------------------------------------
                params["W1"] = params["W1"] - self.alpha * grad_w1
                params["W2"] = params["W2"] - self.alpha * grad_w2
                params["W3"] = params["W3"] - self.alpha * grad_w3
                


# In[8]:


# Do one-hot encoding for our labels
# e.g 7 => [0 0 0 0 0 0 0 1 0 0]
def one_hot(y):
    digits = 10
    examples = y.shape[0]
    y_ = y.reshape(1, examples)
    y_ = np.eye(digits)[y.astype('int32')]
    y_ = y_.T.reshape(digits, examples)
    return y_


# In[9]:


def buildDataSet(file_name1, file_name2, file_name3, file_name4): 
    
    # load the data
    image_i  = pd.read_csv(file_name1, header = None)
    label_i  = pd.read_csv(file_name2, header = None)
    image_j  = pd.read_csv(file_name3, header = None)
    label_j  = pd.read_csv(file_name4, header = None)
    

    # concatenate
    images = np.concatenate((image_i, image_j))
    labels = np.concatenate((label_i, label_j))
    
    # convert to arrays
    images = np.asarray(images)
    labels = np.asarray(labels)
    
    # normalize image data
    images = np.multiply(images, 1.0 / 255.0)
    images = images.astype('float32')
    
    # do one_hot encoding
    labels_new = one_hot(labels)
    
    # split data into training and validation
    size_train = image_i.shape[0]
    images_train, images_valid = images[:size_train], images[size_train:size_train * 2]
    labels_train, labels_valid =  labels_new[:,:size_train], labels_new[:, size_train: size_train * 2]

    labels_train = labels_train.T
    labels_valid = labels_valid.T

    return (images_train, images_valid, labels_train, labels_valid)



# In[10]:


start_time = time.time()

f1 = os.path.basename(sys.argv[1])
f2 = os.path.basename(sys.argv[2])
f3 = os.path.basename(sys.argv[3])
f4 = os.path.basename(sys.argv[4])

# In[11]:

images_train, images_test, labels_train, labels_test = buildDataSet(f1, f2, f3, f4)

# In[ ]:

neuralnet = myNeuralNet(sizes=[784, 128, 64, 10]) # params = [input_size, hidden_1, hidden_2, output_size]
neuralnet.train(images_train, labels_train)
#neuralnet.compute_accuracy(images_test)
# In[ ]:


acc = neuralnet.compute_accuracy(images_test, labels_test)
print(acc * 100)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:




