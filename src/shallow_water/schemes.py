import numpy as np

def step(dependent_var, independent_var, const, idx):

	dependent_var_update = dependent_var.copy()
	dependent_var_update -= const * (np.roll(independent_var, int(1 - idx)) - np.roll(independent_var, 1)) / idx

	return dependent_var_update

def update(model):

	(levels, params) = model
	(c1, c2, idx) = params
	[u_current, h_current] = levels[-1]
	h = step(h_current, u_current, c1, idx)
	u = step(u_current, h, c2, idx)
	levels.append([u, h])

def initial_condition(x, L):

	in_domain = lambda x: (x > L / 3) & (x < 2 * L / 3)
	arg_fn = lambda x: (1 - np.cos(6 * np.pi * (x - L / 3) / L)) / 2

	return in_domain(x) * arg_fn(x)

def initialise(model_params, display_values=True):

	(dx, dt, g, L, H, init, idx) = model_params
	u_grid = np.arange(1 - idx / 2, L, dx)
	h_grid = np.arange(0, L, dx)
	u_initial_condition = init(u_grid, L)
	h_initial_condition = init(h_grid, L)
	levels = [[u_initial_condition.copy(), h_initial_condition.copy()]]
	constants = (H * dt / dx, g * dt / dx, 2)

	if display_values:
		courant_number = np.sqrt(g * H) * dt / dx
		print(f'Courant number for gravity waves: {courant_number}')
		required_dt = dx / np.sqrt(g * H)
		print(f'Required timestep for stability: {int(np.floor(required_dt))}s\n')

	return (levels, constants), (u_grid, h_grid), required_dt

def spurious_h(x, _):

	out = list(np.ones(x.shape))
	out = [-1 if i % 2 == 1 else elem for i, elem in enumerate(out)]

	return np.array(out)