import numpy as np

from NeuralNetwork import NeuralNetwork
from DataLoader import MnistDataloader

TRAINING_IMAGES_FILEPATH = "mnist/train-images.idx3-ubyte"
TRAINING_LABELS_FILEPATH = "mnist/train-labels.idx1-ubyte"
TEST_IMAGES_FILEPATH = "mnist/t10k-images.idx3-ubyte"
TEST_LABELS_FILEPATH = "mnist/t10k-labels.idx1-ubyte"

class Params:
    def __init__(self):
        self.input_size = 784    # 28x28 pixels
        self.hidden_size = 128   # count neurons in hidden layer
        self.output_size = 10
        self.learning_rate = 0.1
        self.epochs = 50
        self.batch_size = 64

class MNIST_Train:
    def __init__(self):
        self.params = Params()
        self.network = NeuralNetwork(self.params.input_size, self.params.hidden_size, self.params.output_size, self.params.learning_rate)
        self.loader = MnistDataloader(TRAINING_IMAGES_FILEPATH, TRAINING_LABELS_FILEPATH, TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH)
        self.train_data, self.test_data = self.loader.load_data()
        self.train_data = (self.transform2flat(self.train_data[0]), self.train_data[1])
        self.test_data = (self.transform2flat(self.test_data[0]), self.test_data[1])

    def run_all_tests(self):
        self.network.run_tests()

    def predict(self, X):
        X_flat = self.transform2flat(X) 
        output = self.network.forward(X_flat)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    @staticmethod
    def transform2flat(data):
        return data.reshape(data.shape[0], -1)

    @staticmethod
    def vec2onehot(labels, num_classes=10):
        return np.eye(num_classes)[labels]

    def train(self, X, y):
        epochs = self.params.epochs
        batch_size = self.params.batch_size
        for epoch in range(epochs):
            y_flat = self.vec2onehot(y)
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y_flat[indices]
            
            total_loss = 0
            batch_count = 0

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.network.forward(X_batch)

                cur_loss = self.network.loss.forward(output, y_batch)
                total_loss += cur_loss
                batch_count += 1

                self.network.backward(X_batch, y_batch, output)

                self.network.update_weights()

            avg_loss = total_loss / batch_count
            train_accuracy = self.evaluate(X, y)
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
