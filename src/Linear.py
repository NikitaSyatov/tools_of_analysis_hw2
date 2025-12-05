import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Linear:
    """
    output = input * W + b
    """
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))
        self.input = None
        self.dW = None
        self.db = None
    
    def forward(self, data):
        self.input = data
        return np.dot(data, self.W) + self.b
    
    def backward(self, d_out):
        self.d_W = np.dot(self.input.T, d_out)
        self.d_b = np.sum(d_out, axis=0, keepdims=True)
        return np.dot(d_out, self.W.T)
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.d_W
        self.b -= learning_rate * self.d_b

    @staticmethod
    def test():
        # test params
        input_size = 5
        output_size = 3
        batch_size = 4

        my_layer = Linear(input_size, output_size)
        torch_layer = nn.Linear(input_size, output_size, bias=True)

        with torch.no_grad():
            torch_layer.weight.data = torch.FloatTensor(my_layer.W.T.copy())
            torch_layer.bias.data = torch.FloatTensor(my_layer.b.flatten())

        np.random.seed(42)
        X_np = np.random.randn(batch_size, input_size).astype(np.float32)
        X_torch = torch.FloatTensor(X_np)
        X_torch.requires_grad = True

        # test forward

        my_output = my_layer.forward(X_np)
        torch_output = torch_layer(X_torch)

        diff = np.abs(my_output - torch_output.detach().numpy()).max()
        assert diff < 1e-6, f"Linear-forward | ERROR | Outputs differ: max diff = {diff}"
        print("Linear-forward | TEST SUCCESS")

        # test backward

        d_out_np = np.random.randn(batch_size, output_size).astype(np.float32)
        d_out_torch = torch.FloatTensor(d_out_np)

        my_d_input = my_layer.backward(d_out_np)

        torch_output.backward(d_out_torch)
        torch_dW = torch_layer.weight.grad.numpy().T
        torch_db = torch_layer.bias.grad.numpy().reshape(1, -1)
        torch_d_input = X_torch.grad.numpy()

        dw_diff = np.abs(my_layer.d_W - torch_dW).max()
        db_diff = np.abs(my_layer.d_b - torch_db).max()
        d_input_diff = np.abs(my_d_input - torch_d_input).max()

        assert dw_diff < 1e-6, f"Linear-backward | ERROR | Weight gradients differ: max diff = {dw_diff}"
        assert db_diff < 1e-6, f"Linear-backward | ERROR | Bias gradients differ: max diff = {db_diff}"
        assert d_input_diff < 1e-6, f"Linear-backward | ERROR | Input gradients differ: max diff = {d_input_diff}"

        print("Linear-backward | TEST SUCCESS")
