import numpy as np

from mnist_train import MNIST_Train

def main():
    mnist_training = MNIST_Train()

    mnist_training.run_all_tests()

    mnist_training.train(mnist_training.train_data[0], mnist_training.train_data[1])

    test_accuracy = mnist_training.evaluate(mnist_training.test_data[0], mnist_training.test_data[1])
    print(f"------------\nFinal Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()