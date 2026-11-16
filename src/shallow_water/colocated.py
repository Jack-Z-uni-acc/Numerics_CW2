from schemes import colocated_update as step
from schemes import initialise_colocated as initialise
from schemes import initial_condition as init
import matplotlib.pyplot as plt

def solve_colocated(nt, dx=2e5, dt=6*60**2, g=10, L=8e6, H=1e4):

	model_params = (dx, dt, g, L, H, init)
	model, x_grid = initialise(model_params)

	nt = 5

	(levels, _) = model
	plt.plot(x_grid, levels[-1][0], label='u at timestep 0')

	for t in range(nt):

		step(model)
		(levels, _) = model
		plt.plot(x_grid, levels[-1][0], label=f'u at timestep {t + 1}')

	plt.yscale('log')
	plt.legend()
	plt.show()

def main():

	solve_colocated(5)

if __name__ == '__main__':
	
	main()