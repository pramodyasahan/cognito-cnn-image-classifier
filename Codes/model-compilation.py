# Utility to compile with chosen optimizer
def compile_model(optimizer_name, lr):
    model = build_simple_cnn(input_shape=X_train.shape[1:], num_classes=num_classes, dropout_rate=0.3)

    # Select optimizer based on the given name
    if optimizer_name == 'adam':
        opt = optimizers.Adam(learning_rate=lr)
    elif optimizer_name == 'sgd':
        opt = optimizers.SGD(learning_rate=lr)
    elif optimizer_name == 'sgd_mom':
        opt = optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError("Unknown optimizer")

    # Compile the model with categorical cross-entropy loss (for multi-class classification) and accuracy as the evaluation metric
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

EPOCHS = 20   # Number of training passes over the entire dataset
BATCH_SIZE = 32  # Number of samples per gradient update (mini-batch size)

experiments = [('adam', 1e-3), ('sgd', 1e-2), ('sgd_mom', 1e-2)]
results = {}

# Training loop for each optimizer experiment

for opt_name, lr in experiments:
    print(f"\n=== Training with {opt_name.upper()} (lr={lr}) ===")
    model = compile_model(opt_name, lr)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )

    # Evaluate trained model on test data (unseen data)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Generate predictions on test data
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    results[opt_name] = {
        'history': history.history,    # Training & validation accuracy/loss per epoch
        'test_acc': test_acc,   # Final test accuracy
        'conf_mat': confusion_matrix(y_true, y_pred),  # confusion matrix
        'report': classification_report(y_true, y_pred, output_dict=True)   # precision/recall report
    }
