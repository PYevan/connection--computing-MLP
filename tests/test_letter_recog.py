import numpy as np
from models.mlp import MLP
from utils.data_loader import load_letter_recognition_data

# Load data
inputs, targets = load_letter_recognition_data("data/letter_recognition.csv")
train_inputs, test_inputs = inputs[:16000], inputs[16000:]
train_targets, test_targets = targets[:16000], targets[16000:]

# Create an MLP
mlp = MLP(input_size=16, hidden_size=20, output_size=26)

# Training
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    total_error = 0
    for i in range(len(train_inputs)):
        output = mlp.forward(train_inputs[i])  # Forward pass
        total_error += mlp.backward(train_inputs[i], train_targets[i])  # Backward pass
    mlp.update_weights(learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Training Error {total_error}")


# Testing
correct_predictions = 0
for i in range(len(test_inputs)):
    output = mlp.forward(test_inputs[i])
    if np.argmax(output) == np.argmax(test_targets[i]):
        correct_predictions += 1

accuracy = correct_predictions / len(test_inputs)
print(f"Letter Recognition Accuracy: {accuracy * 100:.2f}%")
