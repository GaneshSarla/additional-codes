import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Load both CSV files
data1 = pd.read_csv("values.csv")
data2 = pd.read_csv("values2.csv")

# Standardize column names
data1.rename(columns=lambda x: x.strip().lower(), inplace=True)
data2.rename(columns=lambda x: x.strip().lower(), inplace=True)

# Ensure the required columns are present
if 'ground truth' not in data1.columns or 'ground truth' not in data2.columns:
    raise ValueError("Column 'ground truth' not found in one of the datasets.")

columns_to_evaluate_1 = ['huggingface']
columns_to_evaluate_2 = ['bert', 'textblob']
all_models = columns_to_evaluate_1 + columns_to_evaluate_2

# Check column existence
for col in all_models:
    if col not in data1.columns and col not in data2.columns:
        raise ValueError(f"Column '{col}' is missing from both datasets.")

# Initialize results dictionary
results = {}

def evaluate_model_2(data, model_name):
    y_true = data['ground truth'].str.lower()
    y_pred = data[model_name].str.lower()
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }

def evaluate_model_1(data, model_name):
    y_true = data['ground truth'].str.lower()
    y_pred = data[model_name].str.lower()
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, pos_label='positive', average='binary'),
        'Recall': recall_score(y_true, y_pred, pos_label='positive', average='binary'),
        'F1 Score': f1_score(y_true, y_pred, pos_label='positive', average='binary')
    }

# Evaluate models
for model in columns_to_evaluate_2:
    results[model] = evaluate_model_2(data2, model)

for model in columns_to_evaluate_1:
    results[model] = evaluate_model_1(data1, model)

# Convert results dictionary into DataFrame (for table representation)
df_results = pd.DataFrame(results)

# Visualization
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
y = np.arange(len(metrics))  # Label positions
width = 0.25  # Width of bars

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['skyblue', 'orange', 'green']

for i, model in enumerate(all_models):
    scores = [results[model][metric] for metric in metrics]
    ax.barh(y + i * width, scores, width, label=model.capitalize(), color=colors[i])

    # Add value annotations
    for j, score in enumerate(scores):
        ax.text(score + 0.01, y[j] + i * width, f"{score:.2f}", va='center', ha='left', fontsize=10, color='black')

# Add labels and title
ax.set_ylabel('Metrics')
ax.set_xlabel('Scores')
ax.set_title('Performance Metrics Comparison')
ax.set_yticks(y + width)
ax.set_yticklabels(metrics)
ax.legend()

# Show table representation
print("\nPerformance Metrics Table:")
print(df_results.round(2))  # Round to 2 decimal places for readability

# Display the plot
plt.show()
