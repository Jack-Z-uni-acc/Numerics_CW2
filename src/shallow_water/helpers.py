import numpy as np

def l2_error(numerical_solution, actual_solution):

	normalised_difference = (numerical_solution - actual_solution)**2 / np.sum(actual_solution**2)
	mean_squared_error = np.sum(normalised_difference)

	return np.sqrt(mean_squared_error)