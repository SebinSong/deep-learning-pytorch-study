import seaborn as sns
import matplotlib.pyplot as plt

def draw_box_plot (df):
  plt.figure(figsize=(10, 5))
  sns.boxplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

def to_numpy (t):
  if t.requires_grad:
    # detach(): Stops tracking gradients
    return t.detach().cpu().numpy()
  return t.cpu().numpy()
