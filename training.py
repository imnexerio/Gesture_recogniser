from mediapipe_model_maker import gesture_recognizer
from google.colab import drive
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Load the rock-paper-scissor image archive.
IMAGES_PATH = "/content/drive/MyDrive/hand data_compressed"

# Define hyperparameter ranges
learning_rates = [0.005]#, 0.01, 0.1]
#batch_sizes = [20,35,30]
batch_sizes = [40]
#epochs_values = [50,80,100, 150]
epochs_values = [80]

np.random.seed(42)
tf.random.set_seed(42)

# Create a preprocessing parameters object
preprocessing_params = gesture_recognizer.HandDataPreprocessingParams()

# Data structure to store results
hyperparameter_results = []

# Perform grid search
for learning_rate, batch_size, epochs in product(learning_rates, batch_sizes, epochs_values):
    preprocessing_params.learning_rate = learning_rate
    preprocessing_params.batch_size = batch_size
    preprocessing_params.epochs = epochs

    # Split the archive into training, validation, and test dataset.
    data = gesture_recognizer.Dataset.from_folder(
        dirname=IMAGES_PATH,
        hparams=preprocessing_params
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    # Train the model
    hparams = gesture_recognizer.HParams(
        export_dir="drone_model2",
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    # Evaluate on test data
    loss, acc = model.evaluate(test_data, batch_size=1)

    model.export_model()
    !mv drone_model2/gesture_recognizer.task drone_model22.task

    # Store results
    hyperparameter_results.append({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'test_loss': loss,
        'test_accuracy': acc
    })

    # Print results
    print(f"Hyperparameters: LR={learning_rate}, Batch Size={batch_size}, Epochs={epochs}")
    print(f"Test loss: {loss}, Test accuracy: {acc}\n")

# Find the best hyperparameters based on test accuracy
best_result = max(hyperparameter_results, key=lambda x: x['test_accuracy'])
print("Best Hyperparameters:", best_result)

# Plot the results
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot Test Accuracy
ax[0].plot([result['test_accuracy'] for result in hyperparameter_results], 'o-')
ax[0].set_title('Test Accuracy for Different Hyperparameter Combinations')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Accuracy')

# Plot Learning Rate
ax[1].plot([result['learning_rate'] for result in hyperparameter_results], 'o-')
ax[1].set_title('Learning Rate for Different Hyperparameter Combinations')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Learning Rate')

# Plot Batch Size
ax[2].plot([result['batch_size'] for result in hyperparameter_results], 'o-')
ax[2].set_title('Batch Size for Different Hyperparameter Combinations')
ax[2].set_xlabel('Iteration')
ax[2].set_ylabel('Batch Size')

plt.tight_layout()
plt.show()
