from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Loading and preprocessing MNIST data...")

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize the first few images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.suptitle("Sample MNIST Images", fontsize=16)
plt.tight_layout()
plt.show()

# Data preprocessing
print(f"Original training data shape: {x_train.shape}")
print(f"Original test data shape: {x_test.shape}")

# Reshape data to fit the model
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalize pixel values to [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Preprocessed training data shape: {x_train.shape}")

# Build enhanced CNN model
print("\nBuilding enhanced CNN model...")

model = Sequential(
    [
        # First convolutional block
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Second convolutional block
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Fully connected layers
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

# Model summary
model.summary()

# Compile model
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
    ModelCheckpoint(
        "best_mnist_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

print("\nStarting model training...")

# Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=30,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
)

# Evaluate the model
print("\nEvaluating model performance...")
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")


# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"], label="Training Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_training_history(history)

# Save models
print("\nSaving models...")
try:
    # Save Keras model
    model.save("mnist_cnn_model_enhanced.keras")
    print("✓ Enhanced Keras model saved successfully")

    # Save weights
    model.save_weights("mnist_cnn_model_enhanced_weights.weights.h5")
    print("✓ Model weights saved successfully")

    # Save in older .h5 format for broader compatibility
    model.save("mnist_cnn_model_enhanced.h5")
    print("✓ H5 model saved successfully")

except Exception as e:
    print(f"❌ Error during model saving: {e}")

# Display final information
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")
print(f"Total Parameters: {model.count_params():,}")
print("\nSaved Files:")
print("- mnist_cnn_model_enhanced.keras (Complete Keras model)")
print("- mnist_cnn_model_enhanced.h5 (H5 format model)")
print("- mnist_cnn_model_enhanced_weights.weights.h5 (Model weights)")
print("- best_mnist_model.keras (Best model checkpoint)")
