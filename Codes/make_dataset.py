# Step 1: Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = np.load('mnist.npz')   # File you uploaded in left panel

X_train_full = data['x_train']  # Training images ( handwritten digits)
y_train_full = data['y_train']  # Training labels ( digit classes)
X_test_full = data['x_test']   # Test images
y_test_full = data['y_test']  # Test labels

print("Train:", X_train_full.shape, "Test:", X_test_full.shape)

# Combine training and testing for unified splitting
X_full = np.concatenate([X_train_full, X_test_full], axis=0)
y_full = np.concatenate([y_train_full, y_test_full], axis=0)

# Normalize and reshape
X_full = X_full.astype('float32') / 255.0  # MNIST pixel range 0–255
X_full = np.expand_dims(X_full, -1)        # (n, 28, 28, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_full_oh = encoder.fit_transform(y_full.reshape(-1, 1))

# Split 70/15/15
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_full, y_full_oh, test_size=0.15, random_state=42, stratify=y_full
)
val_fraction = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_fraction, random_state=42,
    stratify=np.argmax(y_train_val, axis=1)
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
