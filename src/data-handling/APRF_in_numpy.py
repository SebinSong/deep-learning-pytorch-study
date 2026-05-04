import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

num_experi = 10000
num_trials = 50 # number of trials in each experiment

accuracies = np.zeros(num_experi)
precisions = np.zeros(num_experi)
recalls = np.zeros(num_experi)
F1_scores = np.zeros(num_experi)

for experi_i in range(num_experi):
  # Generate data
  TP = np.random.randint(1, num_trials) # true_positives, aka hits
  FN = num_trials - TP
  TN = np.random.randint(1, num_trials)
  FP = num_trials - TN

  total_trials = 2 * num_trials
  accuracies[experi_i] = (TP + TN) / total_trials
  precisions[experi_i] = TP / (TP + FP)
  recalls[experi_i] = TP / (TP + FN)
  F1_scores[experi_i] = TP / (TP + (FP + FN)/2)

  if experi_i % 1000 == 0:
    print(f'{experi_i} epochs done!')

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# ax1.scatter(accuracies, F1_scores, s=5, c=precisions)
# ax1.plot([0, 1], [.5, .5], 'k--', linewidth=.5)
# ax1.plot([.5, .5], [0, 1], 'k--', linewidth=.5)
# ax1.set_xlabel('Accuracy')
# ax1.set_ylabel('F1 score')
# ax1.set_title('F1-Accuracy by precision')

# ax2.scatter(accuracies, F1_scores, s=5, c=recalls)
# ax2.plot([0, 1], [.5, .5], 'k--', linewidth=.5)
# ax2.plot([.5, .5], [0, 1], 'k--', linewidth=.5)
# ax2.set_xlabel('Accuracy')
# ax2.set_ylabel('F1 score')
# ax2.set_title('F1-Accuracy by recall')

# plt.show()

df = pd.DataFrame({
  'Accuracy': accuracies,
  'Percision': precisions,
  'Recall': recalls,
  'F1 Score': F1_scores
})

corr_matrix = df.corr(method='pearson')
upper_mat = np.triu(corr_matrix.values, k=1)
print(corr_matrix)
print(upper_mat)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.3f', vmin=0, vmax=1)
plt.title('Pearson Correlation coefficient of 4 performance metrics')
plt.show()
