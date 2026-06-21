import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load test results
with open("results_test_nlu/intent_report.json") as f:
    report = json.load(f)

with open("results_test_nlu/intent_errors.json") as f:
    errors = json.load(f)

intents = [k for k in report if isinstance(report[k], dict) and 'f1-score' in report[k] and 'avg' not in k]
intents_sorted = sorted(intents, key=lambda x: report[x]['f1-score'])

# --- Plot 1: Per-intent F1 bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
f1s = [report[i]['f1-score'] for i in intents_sorted]
colors = ['#e74c3c' if f < 0.8 else '#f39c12' if f < 1.0 else '#2ecc71' for f in f1s]
bars = ax.barh(intents_sorted, f1s, color=colors, edgecolor='white', height=0.7)

ax.set_xlabel('F1-score', fontsize=12)
ax.set_xlim(0, 1.1)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.4)
ax.axvline(x=0.8, color='orange', linestyle='--', alpha=0.3)

for bar, f1 in zip(bars, f1s):
    ax.text(f1 + 0.01, bar.get_y() + bar.get_height()/2,
            f'{f1:.2f}', va='center', fontsize=9)

red_patch = mpatches.Patch(color='#e74c3c', label='F1 < 0.80')
orange_patch = mpatches.Patch(color='#f39c12', label='0.80 ≤ F1 < 1.00')
green_patch = mpatches.Patch(color='#2ecc71', label='F1 = 1.00')
ax.legend(handles=[green_patch, orange_patch, red_patch], loc='lower right', fontsize=9)

ax.set_title(f'NLU Intent F1-score (hold-out test set, n=63, accuracy=87.3%)', fontsize=12, pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('images/intent_f1.pdf', bbox_inches='tight')
plt.savefig('images/intent_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved intent_f1.pdf and intent_f1.png")

# --- Plot 2: Confusion matrix (14x14) ---
# Build from errors + support
labels = sorted(intents)
n = len(labels)
label_idx = {l: i for i, l in enumerate(labels)}

# Initialize with diagonal (correct predictions)
cm = np.zeros((n, n), dtype=int)
for intent in labels:
    sup = report[intent]['support']
    cm[label_idx[intent], label_idx[intent]] = sup

# Apply errors
for e in errors:
    true_label = e['intent']
    pred_label = e['intent_prediction']['name']
    if true_label in label_idx and pred_label in label_idx:
        cm[label_idx[true_label], label_idx[pred_label]] += 1
        cm[label_idx[true_label], label_idx[true_label]] -= 1

short_labels = [l.replace('ask_', '').replace('_', '\n') for l in labels]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set(xticks=np.arange(n), yticks=np.arange(n),
       xticklabels=short_labels, yticklabels=short_labels)
ax.set_ylabel('True label', fontsize=11)
ax.set_xlabel('Predicted label', fontsize=11)
ax.set_title('Intent Confusion Matrix (hold-out test set)', fontsize=12, pad=10)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=8)

thresh = cm.max() / 2.
for i in range(n):
    for j in range(n):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=8)

plt.tight_layout()
plt.savefig('images/intent_confusion_matrix.pdf', bbox_inches='tight')
plt.savefig('images/intent_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved intent_confusion_matrix.pdf and intent_confusion_matrix.png")
