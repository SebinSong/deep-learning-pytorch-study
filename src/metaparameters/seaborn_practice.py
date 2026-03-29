import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

sns.set_theme()
tips = sns.load_dataset('tips')
penguins = sns.load_dataset('penguins')
fmri = sns.load_dataset("fmri")

def kde_with_histogram():
  wine_df = fetch_ucirepo(id=186) # wine-quality dataset
  wine_features = wine_df.data.features

  wine_features = (wine_features - wine_features.mean()) / wine_features.std()

  sns.histplot(wine_features, kde=True, color='skyblue', stat='density')
  plt.title('Histogram + KDE (Density) for Wine Alcohol Content')
  plt.xlabel('Alcohol Percentage')
  plt.ylabel('Density')
  plt.show()

print(fmri)
# kde_with_histogram()