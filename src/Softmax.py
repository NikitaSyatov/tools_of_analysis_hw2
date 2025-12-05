import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Softmax:
    """
    class Softmax
    Softmax(x_i) = exp(x_i) / sum(x_j) for j from 1 to n
    """
    def __init__(self):
        self.data = None

    def forward(self, input_data):
        """
        for stabilization: exp(x_i - max(x_i))
        """
        exp_data = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.data = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        return self.data

    def backward(self, d_output):
        """
        dL/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
        """
        y = self.data
        dL_dy = d_output
        sum_ydL = np.sum(y * dL_dy, axis=1, keepdims=True)

        dL_dx = y * (dL_dy - sum_ydL)
        
        return dL_dx

    @staticmethod
    def test():
        my_softmax = Softmax()
        torch_softmax = nn.Softmax(dim=1)

        np.random.seed(42)
        X_np = np.random.randn(3, 4).astype(np.float32)
        X_torch = torch.FloatTensor(X_np)
        X_torch.requires_grad = True

        # test forward
        my_output = my_softmax.forward(X_np)
        torch_output = torch_softmax(X_torch)

        diff = np.abs(my_output - torch_output.detach().numpy()).max()
        assert diff < 1e-6, f"Softmax-forward | ERROR | Softmax values differ: {diff}"

        print("Softmax-forward | TEST SUCCESS")

        # test backward
        d_out_np = np.random.randn(3, 4).astype(np.float32)
        d_out_torch = torch.FloatTensor(d_out_np)

        my_grad = my_softmax.backward(d_out_np)
        torch_output = torch_softmax(X_torch)
        torch_output.backward(d_out_torch)
        torch_grad = X_torch.grad.numpy()

        diff = np.abs(my_grad - torch_grad).max()
        assert diff < 1e-6, f"Softmax-backward | ERROR | Softmax values differ: {diff}"

        print("Softmax-backward | TEST SUCCESS")