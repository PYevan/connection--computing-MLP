from models.mlp import MLP
from utils.data_loader import generate_xor_data, generate_sinusoidal_data, load_letter_recognition_data
import time

def save_results(filename, results):
    """
    Save results to a file.
    """
    with open(filename, "w") as f:
        for line in results:
            f.write(line + "\n")

def train_mlp(inputs, targets, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.01):
    mlp = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    results = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            x = inputs[i:i+1]  # Ensure x has shape (1, input_size)
            y = targets[i:i+1]  # Ensure y has shape (1, output_size)
            mlp.forward(x)  # Forward pass
            total_error += mlp.backward(x, y, learning_rate)  # Backward pass
        if epoch % 100 == 0:
            results.append(f"Epoch {epoch}: Error {total_error:.5f}")
            print(results[-1])

    return mlp, results

def xor_test():
    inputs, targets = generate_xor_data()
    model, results = train_mlp(inputs, targets, input_size=2, hidden_size=4, output_size=1)
    results.append("\nTesting XOR Model:")
    for x, y in zip(inputs, targets):
        prediction = model.forward(x)
        results.append(f"Input: {x}, Predicted: {prediction}, Target: {y}")
    save_results("xor_results.txt", results)

def sinusoidal_test():
    inputs, targets = generate_sinusoidal_data()
    train_inputs, test_inputs = inputs[:400], inputs[400:]
    train_targets, test_targets = targets[:400], targets[400:]

    model, results = train_mlp(train_inputs, train_targets, input_size=4, hidden_size=5, output_size=1)
    results.append("\nTesting Sinusoidal Model:")
    for x, y in zip(test_inputs, test_targets):
        prediction = model.forward(x)
        results.append(f"Input: {x}, Predicted: {prediction}, Target: {y}")
    save_results("sinusoidal_results.txt", results)

def letter_recognition_test():
    inputs, targets = load_letter_recognition_data("data/letter_recognition.csv")
    train_inputs, test_inputs = inputs[:16000], inputs[16000:]
    train_targets, test_targets = targets[:16000], targets[16000:]

    model, results = train_mlp(train_inputs, train_targets, input_size=16, hidden_size=20, output_size=26, epochs=500, learning_rate=0.01)
    results.append("\nTesting Letter Recognition Model:")
    correct = 0
    for x, y in zip(test_inputs, test_targets):
        prediction = model.forward(x)
        if prediction.argmax() == y.argmax():
            correct += 1
    accuracy = correct / len(test_inputs) * 100
    results.append(f"Accuracy: {accuracy:.2f}%")
    save_results("letter_recognition_results.txt", results)

if __name__ == "__main__":
    print("Running XOR Test")
    xor_test()

    print("\nRunning Sinusoidal Test")
    sinusoidal_test()

    print("\nRunning Letter Recognition Test")
    letter_recognition_test()
