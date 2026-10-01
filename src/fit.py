import os

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

########### objective functions that we tried to use ####################
def objectiveArcsin(x, a, b, c, d):
	return a*np.arcsinh(b*(x-c))+d

def objectiveSigmoid(x,a,b):
	return(1/(1+np.exp(-a*(x-b))))

def objectiveArctan(x,a,b):
	return(-1*np.arctan(a*(x-b))/np.pi+0.5)
    # Divide by pi and add 0.5 to have the function limited between 0 and 1


def fitArctan (source, path):
	# Function fits the data with a fitfunction with two variables
	#Plotting values
    timestamp = os.path.splitext(os.path.basename(source))[0] #takes the timestamp out of the file name
    figure_path = os.path.join(path, f'Fit-{timestamp}.svg') #creates the new figure with the title MagCap- in front of the timestamp
    data=read_simulation_csv(source)
    # choose the input and output variables
    x, y = data[:, 0], data[:, 1]
    # curve fit
    popt, _ = scipy.optimize.curve_fit(objectiveArctan, x, y)
    # summarize the parameter values
    a, b = popt
    # plot input vs output
    plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.linspace(np.min(x), np.max(x), 500)
    # calculate the output for the range
    y_line = objectiveArctan(x_line, a, b)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '--', color='red')
    plt.title(f'Calculated Value {b}')
    plt.savefig(figure_path)
    plt.close()
	
def plotMagCap (source, path):
    timestamp = os.path.splitext(os.path.basename(source))[0] #takes the timestamp out of the file name
    figure_path = os.path.join(path, f'MagCap-{timestamp}.svg') #creates the new figure with the title MagCap- in front of the timestamp
    data=read_simulation_csv(source)
    temps=data[:,0]
    m_data=data[:,1]
    C_data=data[:,2]
    fig, axs = plt.subplots(2, figsize=(6.4, 8))
    axs[0].plot(temps, m_data)
    axs[1].plot(temps, C_data)
    axs[0].set_title("Absolute magnetisation over reduced temperature")
    axs[1].set_title("Heat capacity over reduced temperature")
    axs[0].set_xlabel("Reduced temperature")
    axs[1].set_xlabel("Reduced temperature")
    axs[0].set_ylabel("Absolute magnetization")
    axs[1].set_ylabel("Heat capacity")
    fig.tight_layout()
    plt.savefig(figure_path)
    plt.close()
	

#fitArctan('data/2026-09-19_15-02-21.csv', './figures')
#plotMagCap('data/2026-09-19_15-02-21.csv', './figures')