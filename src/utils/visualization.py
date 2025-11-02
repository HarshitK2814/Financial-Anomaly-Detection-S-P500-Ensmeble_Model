from matplotlib import pyplot as plt
import numpy as np

def plot_training_history(history):
    """
    Plots the training and validation loss and accuracy over epochs.

    Parameters:
    - history: A dictionary containing 'loss', 'val_loss', 'accuracy', and 'val_accuracy'.
    """
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_anomalies(original_data, reconstructed_data, anomalies):
    """
    Visualizes the original time series data, reconstructed data, and detected anomalies.

    Parameters:
    - original_data: The original time series data.
    - reconstructed_data: The data reconstructed by the model.
    - anomalies: A list of indices where anomalies are detected.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='orange')
    
    # Highlight anomalies
    plt.scatter(anomalies, original_data[anomalies], color='red', label='Anomalies', marker='x')

    plt.title('Anomaly Detection Visualization')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()