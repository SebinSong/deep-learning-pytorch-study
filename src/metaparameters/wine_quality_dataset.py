from ucimlrepo import fetch_ucirepo
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# helpers
n_uniq = lambda l: len(np.unique(l))

# fetch wine quality dataset - reference: https://archive.ics.uci.edu/dataset/186/wine+quality
wine_quality = fetch_ucirepo(id=186)

# data
X = wine_quality.data.features
y = wine_quality.data.targets
data = X.assign(quality=y['quality'])
mean_data = data.mean()
std_data = data.std()
print(mean_data, type(mean_data))
print(std_data, type(std_data))

# # non_outliers = data['total_sulfur_dioxide'] < 200
# # data = data[non_outliers] # filter out the outliers

# # z-scoring the data : Normalizing the samples
# cols2zscore = data.keys().drop('quality')

# data[cols2zscore] = data[cols2zscore].apply(stats.zscore)
# sns.set_theme()
# sns.violinplot(data=data)
# plt.xticks(rotation=45)
# plt.show()

# # # Let's look into the quality column:
# # fig = plt.figure(figsize=(10, 7))
# # plt.rcParams.update({ 'font.size': 22 })

# # quality_col = data['quality']
# # quality_val_counts = quality_col.value_counts() # pandas Series

# # # drawing histogram for quality column
# # plt.bar(
# #   list(quality_val_counts.index),
# #   quality_val_counts.values
# # )
# # plt.xlabel('Quality rating')
# # plt.ylabel('Count')
# # plt.show()

# # # adding another column named 'bool_quality' which has two possible values: 0 for bad and 1 for good.
# # data['bool_quality'] = 0 # init to 0
# # data['bool_quality'][data['quality'] > 5] = 1

# # print(data[['quality', 'bool_quality']])
# # # using .keys() method of pandas dataframe, also getting the number of unique values from pandas Series.
# # # for k in X.keys():
# # #   print(f'{k} has {n_uniq(X[k])} number of unique values.')

# # # print(f'quality has {n_uniq(quality_col)} number of unique values.')

# # # pairwise plots 
# # # cols2plot = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'quality']
# # # sns.pairplot(data[cols2plot], kind='reg', hue='quality')
# # # plt.show()

# # # fig, ax = plt.subplots(1, figsize=(17, 4))
# # # ax = sns.boxplot(data=data)
# # # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# # # plt.show()
