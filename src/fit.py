import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt


def read_simulation_csv(file_path: str) -> np.ndarray:
	"""Read a simulation CSV and return its data as an array.

	The returned columns are temperature, absolute magnetisation, and heat
	capacity, in that order. # comment lines in the data are skipped automatically.
	"""
	data_frame = pd.read_csv(file_path, comment="#")
	return data_frame.to_numpy(dtype=float)

# objective functions that we tried to use
def objectiveArcsin(x, a, b, c, d):
	return a*np.arcsinh(b*(x-c))+d

def objectiveSigmoid(x,a,b):
	return(1/(1+np.exp(-a*(x-b))))

def objectiveArctan(x,a,b):
	return(-1*np.arctan(a*(x-b))/np.pi+0.5)
    # Divide by pi and add 0.5 to have the function limited by 0 and 1


def fit2 (data, fitfunction):
	# Function fits the data with a fitfunction with two variables
    # choose the input and output variables
    x, y = data[:, 0], data[:, 1]
    # curve fit
    popt, _ = scipy.optimize.curve_fit(fitfunction, x, y)
    # summarize the parameter values
    a, b = popt
    # plot input vs output
    plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.linspace(np.min(x), np.max(x), 500)
    # calculate the output for the range
    y_line = fitfunction(x_line, a, b)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '--', color='red')
    plt.title(f'Calculated Value {b}')
    plt.show()
	
def fit4 (data, fitfunction):
	# Function fits the data with a fitfunction with four variables
    # choose the input and output variables
    x, y = data[:, 0], data[:, 1]
    # curve fit
    popt, _ = scipy.optimize.curve_fit(fitfunction, x, y)
    # summarize the parameter values
    a, b, c, d = popt
    # plot input vs output
    plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.linspace(np.min(x), np.max(x), 500)
    # calculate the output for the range
    y_line = fitfunction(x_line, a, b, c, d)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '--', color='red')
    plt.title(f'Calculated Value {c}')
    plt.show()

fit2(read_simulation_csv('data\\2026-09-19_15-02-21.csv'), objectiveArctan)