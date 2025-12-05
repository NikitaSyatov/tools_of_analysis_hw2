import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReLU:
    """
    class Rectified Linear Unit
    ReLU(x) = max(0, x)

    data - vector of input data
    """
    def __init__(self):
        self.data = None

    def forward(self, input_data):
        """
        This class is filter negative input data of current layer
        """
        self.data = input_data
        return np.maximum(0, input_data)

    def backward(self, d_output):
        """
        d_out - gradient of next layer
        return: gradient = 1 if x > 0, else gradient = 0
        """
        return d_output * (self.data > 0)

    @staticmethod
    def test():
        my_relu = ReLU()
        torch_relu = nn.ReLU()

        # test data
        np.random.seed(42)
        X_np = np.random.randn(3, 4).astype(np.float32)
        X_torch = torch.FloatTensor(X_np)
        X_torch.requires_grad = True

        # test forward
        my_output = my_relu.forward(X_np)
        torch_output = torch_relu(X_torch).detach().numpy()

        diff = np.abs(my_output - torch_output).max()
        assert diff < 1e-6, f"ReLU-forward | ERROR | ReLU values differ: {diff}"

        print("Relu-forward | TEST SUCCESS")

        # test backward
        d_out_np = np.random.randn(3, 4).astype(np.float32)
        d_out_torch = torch.FloatTensor(d_out_np)

        my_grad = my_relu.backward(d_out_np)
        torch_output = torch_relu(X_torch)
        torch_output.backward(d_out_torch)
        torch_grad = X_torch.grad.numpy()

        diff = np.abs(my_grad - torch_grad).max()
        assert diff < 1e-6, f"ReLU-backward | ERROR | ReLU values differ: {diff}"

        print("Relu-backward | TEST SUCCESS")
