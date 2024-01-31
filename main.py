from arm import Gaussian, Uniform, Rayleigh, Exponential
from solver import BanditSolver
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Palatino",
})

# create some arms for simulation
arm1 = Gaussian(0.5, 0.1)
arm2 = Gaussian(0.7, 0.1)
arm3 = Uniform(0, 1)
arm4 = Exponential(0.5)
arm5 = Rayleigh(2)
arms = [arm1, arm2, arm3, arm4, arm5]

# create a solver
sl = BanditSolver(arms)
# set number of iterations
T = 1000

# exp3 simulation
plt.figure(dpi=120)
epsilons = [0.01, 0.1, 0.5, 1]
for ep in epsilons:
    x, loss = sl.exp3(0.5, T)
    print(np.round(x, 3))
    plt.plot(loss, label=f"$\epsilon={ep}$")
plt.title("EXP3 Algorithm")
plt.xlabel("Iterations")
plt.ylabel("Mean Loss")
plt.legend()
plt.grid()
plt.show()
