# Plot training & validation loss
plt.figure(figsize=(8,6))
for name, r in results.items():
    plt.plot(r['history']['loss'], label=f'{name} train')
    plt.plot(r['history']['val_loss'], '--', label=f'{name} val')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(8,6))
for name, r in results.items():
    plt.plot(r['history']['accuracy'], label=f'{name} train')
    plt.plot(r['history']['val_accuracy'], '--', label=f'{name} val')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.show()

