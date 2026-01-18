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


def plot_regime_thresholds(thresholds: dict, output_path: str = None):
    """Bar chart of conditional tail quantile thresholds per volatility regime.

    Args:
        thresholds: dict mapping regime index (0=low, 1=med, 2=high) to threshold value.
        output_path: If provided, save figure to this path instead of showing.
    """
    regimes = sorted([r for r in thresholds.keys() if r >= 0])
    values = [thresholds[r] for r in regimes]
    labels = ['Low Vol', 'Med Vol', 'High Vol'][:len(regimes)]
    colors = ['green', 'orange', 'red'][:len(regimes)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.ylabel('Anomaly Threshold (|return|)')
    plt.title('Conditional Tail Quantile Thresholds by Regime')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_anomaly_distribution(returns, regimes, thresholds: dict, output_path: str = None):
    """Histogram of |returns| by regime with threshold lines.

    Args:
        returns: 1-D array of return values.
        regimes: 1-D array of regime labels (same length as returns).
        thresholds: dict mapping regime index to threshold.
        output_path: If provided, save figure to this path instead of showing.
    """
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    regime_names = ['Low Vol', 'Med Vol', 'High Vol']
    colors = ['green', 'orange', 'red']

    for i, ax in enumerate(axes):
        mask = regimes == i
        if not np.any(mask):
            ax.set_title(regime_names[i] + ' (no data)')
            continue
        abs_ret = np.abs(returns[mask])
        ax.hist(abs_ret, bins=50, alpha=0.7, color=colors[i])
        if i in thresholds:
            ax.axvline(thresholds[i], color='black', linestyle='--',
                       label=f'Threshold={thresholds[i]:.4f}')
        ax.set_title(regime_names[i])
        ax.set_xlabel('|Return|')
        ax.legend()

    axes[0].set_ylabel('Frequency')
    plt.suptitle('Absolute Return Distribution by Regime')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()