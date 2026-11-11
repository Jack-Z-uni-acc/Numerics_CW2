import numpy as np
import matplotlib.pyplot as plt

'''
	Code for the linearised shallow water equations based on the 
	numerical method described in textbook equations 2.24, 2.25.
'''

'''
	Setup and helper functions
'''

# Parameters

dx = 2e5
g = 10
L = 8e6
H = 1e4
dt = 6

# Grid setup

x_lim = [0, L]
x_grid = np.arange(x_lim[0], x_lim[-1], dx)

# Initial conditions

def initial_condition(x):

	in_domain = lambda x: (x > L / 3) & (x < 2 * L / 3)
	arg_fn = lambda x: (1 - np.cos(6 * np.pi * (x - L / 3) / L)) / 2

	return in_domain(x) * arg_fn(x)

# Timestep for h

def step_h(u_current, h_current):

	h = h_current.copy()
	h -= H * dt * (u_current - np.roll(u_current, 1)) / dx

	return h

# Timestep for u

def step_u(u_current, h):

	u = u_current.copy()
	u -= g * dt * (np.roll(h, -1) - h) / dx

	return u

# System timestep

def step(current_values):

	[u_current, h_current] = current_values
	h = step_h(u_current, h_current)
	u = step_u(u_current, h)

	return [u, h]

'''
	Main function to run experiment and analysis
'''

def main():

	init = initial_condition(x_grid)

if __name__ == '__main__':

	main()




















































