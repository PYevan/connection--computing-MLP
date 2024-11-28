import numpy as np
import pandas as pd

def generate_xor_data():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    return inputs, targets

def generate_sinusoidal_data(num_samples=500):
    inputs = np.random.uniform(-1, 1, (num_samples, 4))
    targets = np.sin(inputs[:, 0] - inputs[:, 1] + inputs[:, 2] - inputs[:, 3])
    # Normalize targets to [0, 1]
    targets = (targets - targets.min()) / (targets.max() - targets.min())
    return inputs, targets.reshape(-1, 1)

def load_letter_recognition_data(filepath):
    data = pd.read_csv(filepath, header=None)
    inputs = data.iloc[:, 1:].values
    labels = data.iloc[:, 0].values

    # Normalize inputs (standardize features)
    inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)

    # One-hot encode labels
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    one_hot_labels = np.eye(len(unique_labels))[np.array([label_to_index[label] for label in labels])]
    return inputs, one_hot_labels
