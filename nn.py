import numpy as np 

# define sigmoid functions
def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
def sigmoid_derrivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


class NeuralNet:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.w1 = np.random.rand(self.input.shape[1], 4)
        self.w2 = np.random.rand(4, 1)
        self.output = np.zeros(y.shape)

    
    # update the layer(s) and output
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        self.output = sigmoid(np.dot(self.layer1, self.w2))

    # backpropogate to update weights and biases
    def backprop(self):
        d_w2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derrivative(self.output)))
        d_w1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derrivative(self.output), self.w2.T) * sigmoid_derrivative(self.layer1)))
        
        self.w1 += d_w1
        self.w2 += d_w2



if __name__ == "__main__":
    x = np.array([[0, 0, 1],
                  [0, 1, 1], 
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
     
    nn = NeuralNet(x, y)

    for i in range(15000):
         nn.feedforward()
         nn.backprop()

    print(nn.output)

