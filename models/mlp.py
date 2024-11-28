import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))  # Biases as row vectors
        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))  # Biases as row vectors

        self.output = None

    def activation(self, x):
        return np.maximum(0, x)

    def activation_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, x):
        """
        Forward propagation: x must be a 2D array with shape (batch_size, input_size).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]  # Ensure x is 2D
        self.Z1 = np.dot(x, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.output = self.Z2
        return self.output

    def backward(self, x, y, learning_rate):
        """
        Backward propagation: x and y must be 2D arrays with shape (batch_size, ...).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]  # Ensure x is 2D
        if y.ndim == 1:
            y = y[np.newaxis, :]  # Ensure y is 2D

        # Output error
        output_error = self.output - y  # Shape: (batch_size, output_size)

        # Hidden error
        hidden_error = np.dot(output_error, self.W2.T) * self.activation_derivative(self.Z1)

        # Gradients
        dW2 = np.dot(self.A1.T, output_error)  # Shape: (hidden_size, output_size)
        db2 = np.sum(output_error, axis=0, keepdims=True)  # Shape: (1, output_size)
        dW1 = np.dot(x.T, hidden_error)  # Shape: (input_size, hidden_size)
        db1 = np.sum(hidden_error, axis=0, keepdims=True)  # Shape: (1, hidden_size)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return np.mean(output_error ** 2)
