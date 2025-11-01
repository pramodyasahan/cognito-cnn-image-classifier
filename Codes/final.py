import seaborn as sns  # Add this import

# Your existing code continues here...
# Train final model with best optimizer
final_model = compile_model('adam', 1e-3)
final_history = final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Final evaluation
final_test_loss, final_test_acc = final_model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(final_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"Final Test Accuracy: {final_test_acc:.4f}")
print(f"Final Test Loss: {final_test_loss:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix - NOW FIXED
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # sns is now defined
plt.title('Confusion Matrix - Final Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
