# Test different momentum values
momentum_values = [0.0, 0.5, 0.9, 0.99]
momentum_results = {}

# Train and evaluate the model for each momentum value
for momentum in momentum_values:
    print(f"\n=== Training with SGD Momentum={momentum} ===")
    model = build_simple_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
    opt = optimizers.SGD(learning_rate=0.01, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the current optimizer configuration
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    momentum_results[momentum] = {
        'history': history.history,
        'test_acc': test_acc
    }

# Plot momentum comparison
plt.figure(figsize=(10, 6))
for momentum, result in momentum_results.items():
    plt.plot(result['history']['val_accuracy'], label=f'Momentum={momentum}')
plt.title('Validation Accuracy for Different Momentum Values')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
