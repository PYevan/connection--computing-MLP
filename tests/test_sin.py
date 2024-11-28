import numpy as np
from models.mlp import MLP
from utils.data_loader import generate_sin_data

# Generate data
inputs, targets = generate_sin_data(500)
train_inputs, test_inputs = inputs[:400], inputs[400:]
train_targets, test_targets = targets[:400], targets[400:]

# Create an MLP
mlp = MLP(input_size=4, hidden_size=5, output_size=1)

# Training
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    total_error = 0
    for i in range(len(train_inputs)):
        output = mlp.forward(train_inputs[i])  # Ensure forward() is called
        total_error += mlp.backward(train_inputs[i], train_targets[i])  # Then call backward()
    mlp.update_weights(learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Training Error {total_error}")

# Testing
print("\nTesting Sinusoidal Function:")
for i in range(len(test_inputs)):
    output = mlp.forward(test_inputs[i])
    print(f"Input: {test_inputs[i]}, Predicted: {output}, Target: {test_targets[i]}")
