from ReLU import ReLU
from Softmax import Softmax
from MSELoss import MSELoss
from Linear import Linear


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate

        self.linear_layer1 = Linear(input_size, hidden_size)
        self.relu_activation = ReLU()
        self.linear_layer2 = Linear(hidden_size, output_size)
        self.softmax_activation = Softmax()

        self.loss = MSELoss()

    def run_tests(self):
        self.linear_layer1.test()
        self.relu_activation.test()
        self.softmax_activation.test()
        self.loss.test()
    
    def forward(self, data):
        l1 = self.linear_layer1.forward(data)
        a1 = self.relu_activation.forward(l1)
        l2 = self.linear_layer2.forward(a1)
        result = self.softmax_activation.forward(l2)
        return result
    
    def backward(self, X, y_true, output):
        loss_gradient = self.loss.backward()
        d_l2 = self.softmax_activation.backward(loss_gradient)
        d_a1 = self.linear_layer2.backward(d_l2)
        d_l1 = self.relu_activation.backward(d_a1)
        self.linear_layer1.backward(d_l1)

    def update_weights(self):
        self.linear_layer1.update(self.learning_rate)
        self.linear_layer2.update(self.learning_rate)