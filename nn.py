import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNet:
    def __init__(self, x, y):
        try:
            if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
                raise ValueError("Inputs x and y must be numpy arrays.")
            if x.shape[0] != y.shape[0]:
                raise ValueError("Number of samples in x and y must be the same.")
            
            self.input = x
            self.y = y
            self.w1 = np.random.rand(self.input.shape[1], 4)
            self.w2 = np.random.rand(4, 1)
            self.output = np.zeros(y.shape)
        except Exception as e:
            print(f"Error initializing NeuralNet: {e}")

    def feedforward(self):
        try:
            self.layer1 = sigmoid(np.dot(self.input, self.w1))
            self.output = sigmoid(np.dot(self.layer1, self.w2))
        except Exception as e:
            print(f"Error in feedforward: {e}")

    def backprop(self):
        try:
            d_w2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
            d_w1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.w2.T) * sigmoid_derivative(self.layer1)))
            
            self.w1 += d_w1
            self.w2 += d_w2
        except Exception as e:
            print(f"Error in backprop: {e}")

if __name__ == "__main__":
    x = np.array([[0, 0, 1],
                  [0, 1, 1], 
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
     
    nn = NeuralNet(x, y)
    nn.feedforward()
    nn.backprop()