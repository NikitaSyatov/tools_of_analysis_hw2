import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MSELoss:
    """
    Calc MSE eroor between correctable result and prediction result
    MSE = (1/N)*(sum(y_true - y_pred)^2)
    """
    def __init__(self):
        self.y_true = None
        self.y_pred = None
    
    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

    @staticmethod
    def test():
        # test param
        np.random.seed(42)
        batch_size = 4
        num_classes = 3

        my_loss = MSELoss()
        torch_loss = nn.MSELoss(reduction='mean')

        predictions_np = np.random.randn(batch_size, num_classes).astype(np.float32)
        targets_np = np.random.randn(batch_size, num_classes).astype(np.float32)
        predictions_torch = torch.FloatTensor(predictions_np)
        targets_torch = torch.FloatTensor(targets_np)

        # test forward
        my_loss_value = my_loss.forward(predictions_np, targets_np)
        torch_loss_value = torch_loss(predictions_torch, targets_torch).item()

        loss_diff = abs(my_loss_value - torch_loss_value)
        assert loss_diff < 1e-6, f"MSE loss-forward | ERROR | MSE loss values differ: {loss_diff}"

        print("MSE loss-forward | TEST SUCCESS")

        # test backward
        my_grad = my_loss.backward()
        predictions_torch.requires_grad = True
        torch_loss = torch_loss(predictions_torch, targets_torch)
        torch_loss.backward()
        torch_grad = predictions_torch.grad.numpy()

        grad_diff = np.abs(my_grad - torch_grad).max()
        assert grad_diff < 1e-6, f"MSE loss-backward | ERROR | MSE gradients differ: {grad_diff}"

        print("MSE loss-backward | TEST SUCCESS")
