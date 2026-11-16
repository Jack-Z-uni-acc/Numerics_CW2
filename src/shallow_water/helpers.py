import numpy as np

def l2_error(numerical_solution, actual_solution):

	normalised_difference = (numerical_solution - actual_solution) / actual_solution
	mean_squared_error = np.sum(normalised_difference**2)

	return np.sqrt(mean_squared_error)
