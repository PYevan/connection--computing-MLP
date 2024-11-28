import os
import matplotlib.pyplot as plt


# Function to read error values from result files
def read_errors(file_path):

    errors = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return None
    with open(file_path, "r") as file:
        for line in file:
            if "Epoch" in line and "Error" in line:  # Look for lines with "Epoch" and "Error"
                error = float(line.split("Error")[-1].strip())
                errors.append(error)
    return errors


# Function to read accuracy values from result files
def read_accuracy(file_path):

    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return None
    with open(file_path, "r") as file:
        for line in file:
            if "Accuracy" in line:  # Look for "Accuracy" in the file
                return float(line.split(":")[-1].strip().replace('%', ''))
    return None


# Create directory for saving plots
output_dir = "result/imageresult"
os.makedirs(output_dir, exist_ok=True)

# XOR Task Plot
xor_errors_1 = read_errors("result/attempt-1/xor_results.txt")
xor_errors_2 = read_errors("result/attempt-2/xor_results.txt")
xor_errors_3 = read_errors("result/attempt-3/xor_results.txt")

if any([xor_errors_1, xor_errors_2, xor_errors_3]):
    plt.figure(figsize=(10, 6))
    if xor_errors_1:
        plt.plot(list(range(len(xor_errors_1))), xor_errors_1, label='Attempt 1', linestyle='--', marker='o')
    if xor_errors_2:
        plt.plot(list(range(len(xor_errors_2))), xor_errors_2, label='Attempt 2', linestyle='-.', marker='x')
    if xor_errors_3:
        plt.plot(list(range(len(xor_errors_3))), xor_errors_3, label='Attempt 3', linestyle='-', marker='d')

    plt.title("XOR Task: Training Error Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "xor_error_curve.png"))
    plt.close()
else:
    print("No valid XOR results found. Skipping XOR plot generation.")

# Sinusoidal Task Plot
sin_errors_1 = read_errors("result/attempt-1/sinusoidal_results.txt")
sin_errors_2 = read_errors("result/attempt-2/sinusoidal_results.txt")
sin_errors_3 = read_errors("result/attempt-3/sinusoidal_results.txt")

if any([sin_errors_1, sin_errors_2, sin_errors_3]):
    plt.figure(figsize=(10, 6))
    if sin_errors_1:
        plt.plot(list(range(len(sin_errors_1))), sin_errors_1, label='Attempt 1', linestyle='--', marker='o')
    if sin_errors_2:
        plt.plot(list(range(len(sin_errors_2))), sin_errors_2, label='Attempt 2', linestyle='-.', marker='x')
    if sin_errors_3:
        plt.plot(list(range(len(sin_errors_3))), sin_errors_3, label='Attempt 3', linestyle='-', marker='d')

    plt.title("Sinusoidal Task: Training Error Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "sinusoidal_error_curve.png"))
    plt.close()
else:
    print("No valid Sinusoidal results found. Skipping Sinusoidal plot generation.")

# Letter Recognition Accuracy Bar Plot
accuracy_1 = read_accuracy("result/attempt-1/letter_recognition_results.txt")
accuracy_2 = read_accuracy("result/attempt-2/letter_recognition_results.txt")
accuracy_3 = read_accuracy("result/attempt-3/letter_recognition_results.txt")

if any([accuracy_1, accuracy_2, accuracy_3]):
    plt.figure(figsize=(8, 6))
    accuracies = []
    labels = []
    if accuracy_1:
        accuracies.append(accuracy_1)
        labels.append("Attempt 1")
    if accuracy_2:
        accuracies.append(accuracy_2)
        labels.append("Attempt 2")
    if accuracy_3:
        accuracies.append(accuracy_3)
        labels.append("Attempt 3")

    plt.bar(labels, accuracies, color=['blue', 'green', 'orange'][:len(accuracies)])

    plt.title("Letter Recognition: Final Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Attempts")
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, "letter_recognition_accuracy.png"))
    plt.close()
else:
    print("No valid Letter Recognition results found. Skipping Letter Recognition plot generation.")

print(f"Plots generated and saved in {output_dir}")
