import torch
import matplotlib.pyplot as plt

plt.rcParams.update({ 'font.size': 18 })

# variable to evaluate over
x = torch.linspace(-3, 3, 101)
dashlinecol = [.7, .7, .7] # RGB channel values

def section_1 ():
  global dashlinecol

  def NNoutputx(act_func):
    act_func = getattr(torch, act_func)
    return act_func(x)

  activation_funcs = ['relu', 'sigmoid', 'tanh']

  fig = plt.figure(figsize=(10,8))

  for act_func_name in activation_funcs:
    plt.plot(
      x,
      NNoutputx(act_func_name),
      label=act_func_name,
      linewidth=3
    )

  plt.plot(x[[0,-1]], [0,0], '--', color=dashlinecol)
  plt.plot(x[[0,-1]], [1,1], '--', color=dashlinecol)
  plt.plot([0, 0], [-1, 3], '--', color=dashlinecol)

  plt.legend()
  plt.xlabel('x')
  plt.ylabel('$\sigma(x)$')
  plt.title('Various activation functions')
  plt.xlim(x[[0, -1]])
  plt.ylim([-1, 3])
  plt.show()

def section_2 ():
  global dashlinecol

  def NNoutput(act_func_name):
    act_func = getattr(torch.nn, act_func_name)
    # returns the instance of torch.nn.<act_func_class_name>
    return act_func()

  act_func_names = ['ReLU6', 'Hardshrink', 'LeakyReLU']
  n_types = len(act_func_names)

  fig, ax = plt.subplots(1, n_types, figsize=(8*n_types - 1, 7))

  for i, act_name in enumerate(act_func_names):
    ax[i].plot(x, NNoutput(act_name)(x), label=act_name, linewidth=3)
    ax[i].plot(x[[0,-1]], [0,0], '--', color=dashlinecol)
    ax[i].plot(x[[0,-1]], [1,1], '--', color=dashlinecol)
    ax[i].plot([0,0], [-1,3], '--', color=dashlinecol)
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('$\sigma(x)$')
    ax[i].set_title(act_name)
    ax[i].set_xlim(x[[0, -1]])
    ax[i].set_ylim([-.1, .1])
    # ax[i].set_ylim([-1, 3])

  plt.show()

def section_3 ():
  # x = torch.linspace(-3, 9, 101)
  # relu6 = torch.nn.ReLU6()

  # plt.plot(x, relu6(x))
  # plt.show()

  x = torch.linspace(-3, 3, 21)
  y1 = torch.relu(x)

  y2 = torch.nn.ReLU()(x)

  plt.plot(x, y1, 'ro', label='torch.relu')
  plt.plot(x, y2, 'bx', label='torch.nn.ReLU')
  plt.legend()
  plt.xlabel('Input')
  plt.ylabel('Output')
  plt.show()

def section_4 ():
  x1 = torch.linspace(-1, 1, 20)
  x2 = 2*x1

  # weights
  w1 = -0.3
  w2 = 0.5

  linpart = x1 * w1 + x2 * w2

  # non-linear output
  y = torch.relu(linpart)
  print(x1)
  print(linpart)

  plt.plot(x1, linpart, 'bo-', label='Linear input')
  plt.plot(x1, y, 'rs', label='Nonlinear output')
  plt.xlabel('x1 variable')
  plt.ylabel('$\\hat{y}$')
  plt.legend()
  plt.show()

# section_1()
# section_2()
# section_3()
section_4()
