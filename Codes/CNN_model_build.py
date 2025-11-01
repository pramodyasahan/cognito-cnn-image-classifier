# Simple CNN model builder
def build_simple_cnn(input_shape=(8,8,1), num_classes=10, dropout_rate=0.3):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),  # Second convolutional layer: extracts deeper features with 64 filters
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Fully connected (dense) layer with 128 neurons to learn complex relationships
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')  # softmax converts scores into probabilities across 10 classes
    ])
    return model

# Ensure num_classes is defined before training
num_classes = y_train.shape[1]   # automatically detects number of categories
print("Number of classes:", num_classes)

layers.Dense(num_classes, activation='softmax')  # # Final layer converting outputs into probability distribution across classes
