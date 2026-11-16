from schemes import colocated_update as step
from schemes import initialise_colocated as initialise
from schemes import initial_condition as init
import matplotlib.pyplot as plt

'''
	Parameters
'''

dx = 2e5
dt = 600
g = 10
L = 8e6
H = 1e4

model_params = (dx, dt, g, L, H, init)
model, x_grid = initialise(model_params)

nt = 5

for _ in range(nt):

	step(model)

(levels, _) = model

plt.plot(x_grid, levels[0][0], label='Initial condition for u')
plt.plot(x_grid, levels[-1][0], label='u at final time')
plt.legend()
plt.show()