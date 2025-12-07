import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import time

# ==========================================
# 1. Data Setup (Fashion MNIST)
# ==========================================
def load_and_prep_data():
    print("Loading Fashion-MNIST...")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize and Reshape: (28, 28) -> (28, 28, 1)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

# ==========================================
# 2. Model Architecture
# ==========================================
def build_cnn():
    model = models.Sequential()

    # --- Data Augmentation Layer (Active only during training) ---
    model.add(layers.RandomFlip("horizontal", input_shape=(28, 28, 1)))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    # --- Convolutional Base ---
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Dense Top with Regularization ---
    model.add(layers.Flatten())
    
    # Dense 1 with L2 and Dropout
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    
    # Dense 2 with L2 and Dropout
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.3))
    
    # Output
    model.add(layers.Dense(10, activation='softmax'))

    return model

# ==========================================
# 3. Training & Evaluation
# ==========================================
def run_experiment():
    X_train, y_train, X_test, y_test = load_and_prep_data()
    
    model = build_cnn()
    model.summary()

    # Compile with Adam
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    print("\nStarting Training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.2f} seconds.")

    return history, model, X_test, y_test

# ==========================================
# 4. Visualization
# ==========================================
if __name__ == "__main__":
    history, model, X_test, y_test = run_experiment()

    # Plot Accuracy and Loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('CNN Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('CNN Loss')
    plt.show()
    
    # Evaluate Final Performance
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")