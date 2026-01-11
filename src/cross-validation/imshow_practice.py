from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

dir_path = Path(__file__).parent
resolve_path = lambda rel_path: str((dir_path / rel_path).resolve().as_posix())

img_arr = plt.imread(resolve_path('logo.png'))

plt.figure(figsize=(7,5))
plt.imshow(img_arr, aspect='auto', origin='upper')
plt.show()