import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # Using keras only to download data easily
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# ==========================================
# 1. Data Preprocessing
# ==========================================
def load_data():
    print("Loading MNIST Data...")
    (X_full, y_full), _ = mnist.load_data()
    
    # Flatten images: (60000, 28, 28) -> (60000, 784)
    X_flatten = X_full.reshape(X_full.shape[0], -1).astype('float32')
    
    # Normalize to [0, 1]
    X_normalized = X_flatten / 255.0
    
    # One-hot encoding
    y_encoded = np.eye(10)[y_full]
    
    # Split 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# ==========================================
# 2. Activation Functions & Derivatives
# ==========================================
class Activations:
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_deriv(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        # Subtract max for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ==========================================
# 3. Multilayer Perceptron Class
# ==========================================
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Architecture: Input -> Hidden1 -> Hidden2 -> Output
        hidden_sizes: list, e.g., [128, 64]
        """
        self.params = {}
        self.layers = [input_size] + hidden_sizes + [output_size]
        
        # He Initialization for ReLU layers
        np.random.seed(42)
        for i in range(1, len(self.layers)):
            # W = randn * sqrt(2/n_in)
            self.params[f'W{i}'] = np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i-1])
            self.params[f'b{i}'] = np.zeros((1, self.layers[i]))

    def forward(self, X):
        self.cache = {'A0': X}
        L = len(self.layers) - 1
        
        # Forward through hidden layers (ReLU)
        for i in range(1, L):
            Z = np.dot(self.cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            A = Activations.relu(Z)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
            
        # Forward through output layer (Softmax)
        Z_last = np.dot(self.cache[f'A{L-1}'], self.params[f'W{L}']) + self.params[f'b{L}']
        A_last = Activations.softmax(Z_last)
        self.cache[f'Z{L}'] = Z_last
        self.cache[f'A{L}'] = A_last
        
        return A_last

    def backward(self, Y, learning_rate):
        m = Y.shape[0]
        L = len(self.layers) - 1
        grads = {}
        
        # 1. Output Layer Error (Softmax + Cross Entropy derivative is A - Y)
        dZ = self.cache[f'A{L}'] - Y
        
        # 2. Backprop Loop
        for i in range(L, 0, -1):
            prev_A = self.cache[f'A{i-1}']
            
            # Gradients
            dW = (1/m) * np.dot(prev_A.T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Update parameters immediately (SGD)
            self.params[f'W{i}'] -= learning_rate * dW
            self.params[f'b{i}'] -= learning_rate * db
            
            # Prepare dZ for the next layer (moving backwards)
            if i > 1:
                dA_prev = np.dot(dZ, self.params[f'W{i}'].T)
                dZ = dA_prev * Activations.relu_deriv(self.cache[f'Z{i-1}'])

    def compute_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        # Categorical Cross Entropy
        # Add epsilon to avoid log(0)
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m
        return loss

# ==========================================
# 4. Training Execution
# ==========================================
def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    # Architecture from requirements: 784 -> 128 -> 64 -> 10
    model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10)
    
    epochs = 20
    batch_size = 64
    learning_rate = 0.1
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Starting Training: LR={learning_rate}, Batch={batch_size}")
    
    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        # Mini-batch Gradient Descent
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            model.forward(X_batch)
            model.backward(y_batch, learning_rate)
            
        # Evaluation
        train_pred = model.forward(X_train)
        val_pred = model.forward(X_test)
        
        train_loss = model.compute_loss(y_train, train_pred)
        val_loss = model.compute_loss(y_test, val_pred)
        
        val_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(val_pred, axis=1))
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, history, X_test, y_test

# ==========================================
# 5. Visualization & Main
# ==========================================
if __name__ == "__main__":
    model, history, X_test, y_test = train_model()
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves (MLP)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    
    # Final Metrics
    y_pred = np.argmax(model.forward(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n--- Final Classification Report ---")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()