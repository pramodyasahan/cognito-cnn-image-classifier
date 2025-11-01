# Create a summary DataFrame
summary = pd.DataFrame([
    {'Optimizer': k, 'Test Accuracy': v['test_acc']} for k, v in results.items()
])
display(summary)

# Show confusion matrices and precision/recall
for name, r in results.items():
    print(f"\n--- {name.upper()} ---")
    print("Confusion Matrix:\n", r['conf_mat'])
    cr = pd.DataFrame(r['report']).transpose().round(3)
    display(cr)
