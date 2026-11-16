from schemes import update
from schemes import initialise
from schemes import initial_condition as init
from schemes import spurious_h as h
import matplotlib.pyplot as plt
import os
import numpy as np
from helpers import l2_error

def main():

	def solve(grid_type, filename, filepath='plots', step=update, initial_condition=init, nt=None, T=5*6*60**2, dx=2e5, dt=6*60**2, g=10, L=8e6, H=1e4, num_snapshots=5, full_revolution=False):

		if not os.path.exists(filepath):
			os.makedirs(filepath)

		if grid_type == 'colocated':
			idx = 2
		elif grid_type == 'staggered':
			idx = 1
		else:
			raise InputError('Invalid grid_type. grid_type must be colocated or staggered.')

		model_params = (dx, dt, g, L, H, initial_condition, idx)
		model, (u_grid, _), stable_dt = initialise(model_params)

		if full_revolution:
			if nt is not None:
				raise ValueError('Cannot set nt while ensuring a full revolution of the domain. Avoid setting nt or set full_revolution to False.')
			else:
				nt = int(np.floor(2 * L / dx))
		else:
			if nt is None:
				nt = int(np.floor(T / dt))

		(levels, _) = model

		save_step = int(np.floor(nt / num_snapshots))

		for t in range(nt):

			update(model)
			(levels, _) = model

			if t % save_step == 0:
				plt.plot(u_grid, levels[-1][0], label=f'u at timestep {t + 1}')

		if full_revolution:
			plt.clf()

		plt.plot(u_grid, levels[0][0], label='u at initial condition')
		plt.plot(u_grid, levels[-1][0], label='u at final time')
		plt.legend()
		plt.savefig(filepath + '/' + filename + '_' + grid_type + '.pdf')
		plt.clf()

		if full_revolution:
			l2_error_u = l2_error(levels[-1][0], levels[0][0])
			print(f'Error after 2 revolutions from initial condition: {l2_error_u}\n')

			return l2_error_u

		else:
			return stable_dt


	def plot_tests(grid_type):

		stable_dt = solve(grid_type, 'unstable')
		solve(grid_type, 'stable', dt=stable_dt)
		error_initial = solve(grid_type, 'full_revolution', dt=stable_dt, full_revolution=True)
		error_new = solve(grid_type, 'full_revolution_hi_res', dt=stable_dt/2, dx=1e5, full_revolution=True)

		print(f'Ratio of error reduction when doubling resolution: {error_new / error_initial}\n')

		solve(grid_type, 'spurious', dt=stable_dt, initial_condition=h)

	plot_tests('colocated')
	plot_tests('staggered')

if __name__ == '__main__':

	main()