import csv
import datetime
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pytz
from numba import jit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# Multithreading initialisations
stop_event = threading.Event()
worker = None

DATA_PATH: str = "./data"                           # Default: ""       # Relative path to folder for data saving
FIGURE_PATH: str = "./figures"                      # Default: ""       # Relative path to foler for figure saving
SAVE_DATA: bool = True                              # Default: False    # Whether to save data at the end
SAVE_FIGURE: bool = True                            # Default: False    # Whether to save the figure at the end
DEBUG_MODE: bool = False                            # Default: False    # Whether intermediate values are printed into the terminal

RANDOMNESS_SEED: int | None = 17                    # Default: 5        # Seed for randomness to get reproducable results

LATTICE_SIDE_SIZE: int = 50                         # Default: 50       # In the instructions refered to as n, the spin lattice is of size n*n
INITIAL_DOWN_PROBABILITY: float = 0.5               # Default: 0.5      # Probability that any given spin in the initial configuration is spin down

START_TEMPERATURE: float = 1.5                      # Default: 2.0      # Lower temperature limit for the sweep over temperatures
END_TEMPERATURE: float = 3                          # Default: 2.5      # Upper temperature limit for the sweep over temperatures
TEMPERATURE_STEPS: int = 76                         # Default: 111      # Number of temperature values to simualte

# Default: 1        # Distribution of temperature values
TEMPERATURE_DISTRIBUTION = lambda x: 10*np.exp(-((x - ((START_TEMPERATURE + END_TEMPERATURE)/2))**2)/(0.25))

SIMULATION_BATCH_COUNT: int = 6                     # Default: 4        # Number of simulation batches to perform per temperature
BATCH_CPU_CORE_LIMIT: int = 7                       # Default: None     # Limit the number of CPU cores to a specified number

SIMULATION_SWEEP_COUNT: int = 1500                  # Default: 1300     # Number of sweeps to perform in all simulations. Values are collected at the end of every sweep
EQUILIBRATION_SWEEP_COUNT: int = 1000               # Default: 1000     # Number of sweeps during which data is not collected to give the system time to reach equilibrium
ITERATIONS_PER_SWEEP: int = 27000                   # Default: 10000    # Number of spin-flip-attempts per sweep

simulation_parameters = [
    TEMPERATURE_STEPS,                    
    SIMULATION_BATCH_COUNT,                     
    BATCH_CPU_CORE_LIMIT,                 
    INITIAL_DOWN_PROBABILITY,
    SIMULATION_SWEEP_COUNT,
    EQUILIBRATION_SWEEP_COUNT,
    np.asarray([START_TEMPERATURE, END_TEMPERATURE]),
    TEMPERATURE_DISTRIBUTION,
    LATTICE_SIDE_SIZE,
    ITERATIONS_PER_SWEEP,
    RANDOMNESS_SEED,
    DATA_PATH,
    FIGURE_PATH,
    SAVE_DATA,
    SAVE_FIGURE,
    DEBUG_MODE
]
# Added the names as strings so that we can put the parameters in the csv files
simulation_parameter_names = [
    "TEMPERATURE_STEPS",
    "SIMULATION_BATCH_COUNT",
    "BATCH_CPU_CORE_LIMIT",
    "INITIAL_DOWN_PROBABILITY",
    "SIMULATION_SWEEP_COUNT",
    "EQUILIBRATION_SWEEP_COUNT",
    "TEMPERATURE_RANGE",
    "LATTICE_SIDE_SIZE",
    "ITERATIONS_PER_SWEEP",
    "RANDOMNESS_SEED",
    "DATA_PATH",
    "FIGURE_PATH",
    "SAVE_DATA",
    "SAVE_FIGURE",
]

def generate_temperature_grid(pdf_func, low, high, num_points):
    x_eval = np.linspace(low, high, 10000, endpoint=True)
    y_eval = pdf_func(x_eval)
    
    cdf = cumulative_trapezoid(y_eval, x_eval, initial=0)
    cdf_normalized = cdf / cdf[-1]
    
    inverse_cdf = interp1d(cdf_normalized, x_eval, kind='linear')
    
    u_uniform = np.linspace(0, 1, num_points)
    
    return inverse_cdf(u_uniform)

def show_spins(spin_matrix: np.ndarray):
    '''Function to display the spin matrix as a rectangular field with colours corresponding to spin values'''
    plt.imshow(spin_matrix)
    plt.show()

@jit(nopython=True)
def generate_initial_spin_orientations(down_probability: float, n: int, seed: int = 0):
    '''Function to initialise the n by n initial matrix with spins.
    Each spin has down_probability to be a spin down (-1) and 1-down_probability to be a spin up (1).
    It then returns the matrix as np.ndarray with size n x n and dtype = int.'''
    if seed != 0:
        np.random.seed(seed)
    spins = np.ones((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if np.random.random() < down_probability:
                spins[i, j] = -1
    return spins

@jit(nopython=True)
def metropolis_algorithm_step(spin_matrix: np.ndarray, T: float, n: int, exp_4: float, exp_8: float):
    '''Function that performs 1 step in the metropolis algorithm.
    Optimized to use inlined energy calculation and precomputed exponentials.'''
    x: int = np.random.randint(0, n)
    y: int = np.random.randint(0, n)
    s = spin_matrix[x, y]
    
    flip_energy = 2 * s * (spin_matrix[(x+1)%n, y] + spin_matrix[(x-1)%n, y] + spin_matrix[x, (y+1)%n] + spin_matrix[x, (y-1)%n])

    # We use the pre-computed exponentails to determine wheter flip happens
    if flip_energy <= 0:
        spin_matrix[x, y] = -s
    else:
        if flip_energy == 4:
            if np.random.random() < exp_4:
                spin_matrix[x, y] = -s
        else:
            if np.random.random() < exp_8:
                spin_matrix[x, y] = -s

@jit(nopython=True)
def update_aggregate_quantities(spin_matrix: np.ndarray, agg_m: float, agg_E: float, agg_E2: float):
    '''Function that appends the running total of the magnetisation per spin, the system energy and squared system energy.'''
    n = spin_matrix.shape[0]
    vert_sum = 0
    horiz_sum = 0
    for i in range(n):
        for j in range(n):
            s = spin_matrix[i, j]
            vert_sum -= s * spin_matrix[(i + 1) % n, j]
            horiz_sum -= s * spin_matrix[i, (j + 1) % n]
    E = vert_sum + horiz_sum
    return agg_m + np.sum(spin_matrix) / spin_matrix.size, agg_E + E, agg_E2 + E**2

@jit(nopython=True)
def get_average_quantities(agg_m: float, agg_E: float, agg_E2: float, iterations: int, T: float, N: int):
    '''Function that calculates the average values of the magnetisation per spin, the system energy and squared system energy after a number of iterations.'''
    if iterations <= 0:
        return 0.0, 0.0, 0.0
    avg_m = agg_m / iterations
    avg_E = agg_E / iterations
    avg_E2 = agg_E2 / iterations
    heat_capacity = (avg_E2 - avg_E**2) / (T**2)

    return abs(avg_m), avg_E / N, heat_capacity

@jit(nopython=True)
def simulation(random_seed: int, down_probability: float, sweeps: int, burn_in_sweeps: int, T: float, n: int, iterations: int):
    '''Function that performs a simulation with a seed and initial down_probability for the spin matrix.'''
    np.random.seed(random_seed)
        
    agg_m = 0.0
    agg_E = 0.0
    agg_E2 = 0.0
    spin_matrix: np.ndarray = generate_initial_spin_orientations(down_probability, n, 0) # Generate initial spin matrix

    effective_samples = max(1, sweeps - burn_in_sweeps) # Get number of sweeps with data collection

    exp_4 = np.exp(-4.0 / T)
    exp_8 = np.exp(-8.0 / T)

    # Perform sweeps
    for i in range(sweeps):
        for _ in range(iterations):
            metropolis_algorithm_step(spin_matrix, T, n, exp_4, exp_8)
        if i >= burn_in_sweeps:
            agg_m, agg_E, agg_E2 = update_aggregate_quantities(spin_matrix, agg_m, agg_E, agg_E2)

    # Return average quantities from simulation
    return get_average_quantities(agg_m, agg_E, agg_E2, effective_samples, T, n**2)

def run_simulation_task(args):
    """Worker function top-level wrapper for multiprocessing."""
    return simulation(*args)

def meta_meta_simulation(value_count: int, batch_count: int,  core_limit: int | None, down_probability: float, sweeps: int, burn_in_sweeps: int, temp_range: np.ndarray, temp_function, n: int, iterations: int, rand_seed: int | None, DATA_PATH: str, FIGURE_PATH: str, SAVE_DATA: bool, SAVE_FIGURE: bool, DEBUG_MODE: bool):
    '''Function that sweeps across temperature range given by temp_range and performs a metasimulation with batch_count * # available cores simulations.
    It then displayes the values of averagre absolute magnetisation and heat capacity for the reduced temperature values.'''
    start_time = time.perf_counter()
    m_data: np.ndarray = np.zeros(value_count)
    C_data: np.ndarray = np.zeros(value_count)

    # Get temperatures
    temps: np.ndarray = generate_temperature_grid(temp_function, temp_range[0], temp_range[1], value_count)
    
    # Get number of usable cores per batch
    num_cores = min(max(1, (os.cpu_count() or 1) - 1), core_limit) if core_limit else max(1, (os.cpu_count() or 1) - 1)
    total_sims = batch_count * num_cores # Total number of simualtions to perform per temperature

    # Pre-generate deterministic simulation seeds to ensure physics remain identical regardless of DEBUG_MODE branch
    simulation_seeds = np.zeros((value_count, total_sims), dtype=np.int64)
    for i in range(value_count):
        simulation_seeds[i] = np.random.default_rng(rand_seed + i if rand_seed else None).integers(1, 2**31 - 1, size=total_sims)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        if DEBUG_MODE:
            # Sequential batch execution per temperature
            for i in range(value_count):
                print(f'Starting temperature {i+1}/{value_count}')
                
                meta_agg_m = 0.0
                meta_agg_E = 0.0
                meta_agg_C = 0.0

                temp_start_time = time.perf_counter()
                
                # generate simualtion tasks for multithreadding for this temperature
                tasks = [
                    (int(simulation_seeds[i, j]), down_probability, sweeps, burn_in_sweeps, temps[i], n, iterations)
                    for j in range(total_sims)
                ]

                futures = [executor.submit(run_simulation_task, task) for task in tasks]
                for sim_counter, future in enumerate(futures, 1):
                    if stop_event.is_set():
                        break
                    m, E, C = future.result()
                    meta_agg_m += m
                    meta_agg_E += E
                    meta_agg_C += C
                    print(f'Completed simulation {sim_counter}/{total_sims}')

                print(f'Temperature simulations took {time.perf_counter() - temp_start_time} s')

                m_data[i] = meta_agg_m / total_sims
                C_data[i] = meta_agg_C / total_sims

                print(f' reduced temperature is: {temps[i]}')
                print(f' average magnetisation is: {m_data[i]}')
                print(f' average heat capacity is: {C_data[i]}')
        else:
            # Flattened parallelization
            tasks = [
                (int(simulation_seeds[i, j]), down_probability, sweeps, burn_in_sweeps, temps[i], n, iterations)
                for i in range(value_count)
                for j in range(total_sims)
            ]
            
            futures = {executor.submit(run_simulation_task, task): idx for idx, task in enumerate(tasks)}
            for future in futures:
                if stop_event.is_set():
                    break
                task_idx = futures[future]
                temp_idx = task_idx // total_sims
                m, _, C = future.result()
                
                # Aggregate directly into the data arrays without blocking the rest of the queue
                m_data[temp_idx] += m / total_sims
                C_data[temp_idx] += C / total_sims

    # Plot relevant quantities and save to a figure if this is set to true at the top
    timestamp = datetime.datetime.now(pytz.timezone('Europe/Amsterdam')).strftime('%Y-%m-%d_%H-%M-%S')
    if SAVE_FIGURE:
        os.makedirs(FIGURE_PATH, exist_ok=True)
        figure_path = os.path.join(FIGURE_PATH, f'{timestamp}.png')

        plt.subplot(121)
        plt.plot(temps, m_data)
        plt.title("Absolute magnetisation over reduced temperature")
        plt.xlabel("Reduced temperature")
        plt.ylabel("Absolute magnetisation")

        plt.subplot(122)
        plt.plot(temps, C_data)
        plt.title("Heat capacity over reduced temperature")
        plt.xlabel("Reduced temperature")
        plt.ylabel("Heat capacity")
        plt.savefig(figure_path)
        if DEBUG_MODE: plt.show()
        plt.close()

    minute_sim_time: float = (time.perf_counter() - start_time) / 60
    if DEBUG_MODE: print(f'The whole simulation took {minute_sim_time} minutes.')
    
    # Saves the data to a csv file if the responding value at the top is set to True. Name of the file is a timestamp
    if SAVE_DATA:
        os.makedirs(DATA_PATH, exist_ok=True)
        file_path = os.path.join(DATA_PATH, f'{timestamp}.csv')

        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['temps', 'abs_magnetisation', 'Heat Capacity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csvfile.write("# ================================================================================================\n")
            csvfile.write("# This file contains the simulated magnetisation and heat capacity per reduced temperature.\n")
            csvfile.write(f"# The simualtion ran for {minute_sim_time} minutes.\n")
            csvfile.write("# Simulation parameters:\n")
            csvfile.writelines(f"# {name} = {value!r}\n" for name, value in zip(simulation_parameter_names, simulation_parameters))
            csvfile.write("# ================================================================================================\n")
            csvfile.write("#\n")
            writer.writeheader()
            for i in range(len(temps)):
                writer.writerow({
                    'temps': temps[i],
                    'abs_magnetisation': m_data[i],
                    'Heat Capacity': C_data[i],
                })

def main():
    '''Main function that runs the simulations.'''
    print("Simulation started")
    meta_meta_simulation(*simulation_parameters)
    print("Simulation ended")

if __name__=="__main__":
    '''Execution helper.'''
    main()