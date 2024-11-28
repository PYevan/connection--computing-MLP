import numpy as np
from models.mlp import MLP

# XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Create an MLP
mlp = MLP(input_size=2, hidden_size=3, output_size=1)

# Training
epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    total_error = 0
    for i in range(len(inputs)):
        output = mlp.forward(inputs[i])  # Forward pass
        total_error += mlp.backward(inputs[i], targets[i])  # Backward pass
    mlp.update_weights(learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Error {total_error}")



# Testing
print("\nTesting XOR problem:")
for i in range(len(inputs)):
    output = mlp.forward(inputs[i])
    print(f"Input: {inputs[i]}, Predicted: {output}, Target: {targets[i]}")
