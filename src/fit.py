import numpy as np
import pandas as pd
import scipy 
from matplotlib import pyplot as plt


def read_simulation_csv(file_path: str) -> np.ndarray:
	"""Read a simulation CSV and return its data as an array.

	The returned columns are temperature, absolute magnetisation, and heat
	capacity, in that order. # comment lines are skipped automatically.
	"""
	data_frame = pd.read_csv(file_path, comment="#")
	return data_frame.to_numpy(dtype=float)

# form of the function we are trying to fit
def objective(x, a, b, c, d):
	return a*np.arcsinh(b*(x-c))+d

def fit (data):
    # choose the input and output variables
    x, y = data[:, 0], data[:, 1]
    # curve fit
    popt, _ = scipy.optimize.curve_fit(objective, x, y)
    # summarize the parameter values
    a, b, c, d = popt
    # plot input vs output
    plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.linspace(np.min(x), np.max(x), 500)
    # calculate the output for the range
    y_line = objective(x_line, a, b, c, d)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '--', color='red')
    plt.title(f'Calculated Value {c}')
    plt.show()

fit(read_simulation_csv('data\\2026-09-18_11-58-03.csv'))