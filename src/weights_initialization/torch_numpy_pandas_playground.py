import numpy as np

if False:
  data = [1, 2, 1, 5, 6, 6]
  counts, edges = np.histogram(data, 3)
  middles = (edges[:-1] + edges[1:]) / 2
  print(counts)
  print(edges)
  print(middles)

if True:
  # np.logspace() research
  range1 = np.logspace(np.log10(1), np.log10(10**5), 6, base=10)
  # np.logspace(a, b, (b-a)+1, base=c):
  # - This pretty much means I want to create an ndarray that ranges from c**a to c**b in log-c space.
  range2 = np.logspace(1, 5, 5, base=2)

  print(range2)
  nd_arr1 = 2**np.arange(0, 10)
  nd_arr2 = np.arange(0, 10)**2
  print(nd_arr1)
  print(nd_arr2)