import numpy as np

'''
	Colocated schemes
'''

def colocated_step(dependent_var, independent_var, const):

	dependent_var_update = dependent_var.copy()
	dependent_var_update -= const * (np.roll(independent_var, -1) - np.roll(independent_var, 1))

	return dependent_var_update

def colocated_update(model):

	(levels, params) = model
	(c1, c2) = params
	[u_current, h_current] = levels[-1]
	h = colocated_step(h_current, u_current, c1)
	u = colocated_step(u_current, h, c2)
	levels.append([u, h])

def initial_condition(x, L):

	in_domain = lambda x: (x > L / 3) & (x < 2 * L / 3)
	arg_fn = lambda x: (1 - np.cos(6 * np.pi * (x - L / 3) / L)) / 2

	return in_domain(x) * arg_fn(x)

def initialise_colocated(model_params):

	(dx, dt, g, L, H, init) = model_params
	x_grid = np.arange(0, L, dx)
	initial_condition = init(x_grid, L)
	levels = [[initial_condition.copy(), initial_condition.copy()]]
	constants = (H * dt / dx / 2, g * dt / dx / 2)

	return (levels, constants), x_grid