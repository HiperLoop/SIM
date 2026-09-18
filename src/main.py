import datetime
import os
import threading
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pytz
import datetime
import csv
from numba import jit

# Multithreading initialisations
stop_event = threading.Event()
worker = None

DATA_PATH: str = "./data"               # Default: ""       # Relative path to folder for data saving
FIGURE_PATH: str = "./figures"          # Default: ""       # Relative path to foler for figure saving
SAVE_DATA: bool = True                  # Default: False    # Whether to save data at the end
SAVE_FIGURE: bool = True                # Default: False    # Whether to save the figure at the end

RANDOMNESS_SEED: int | None = 5         # Default: 5        # Seed for randomness to get reproducable results

LATTICE_SIDE_SIZE: int = 50             # Default: 50       # In the instructions refered to as n, the spin lattice is of size n*n
INITIAL_DOWN_PROBABILITY: float = 0.5   # Default: 0.5      # Probability that any given spin in the initial configuration is spin down

START_TEMPERATURE: float = 2.0          # Default: 2.0      # Lower temperature limit for the sweep over temperatures
END_TEMPERATURE: float = 2.5            # Default: 2.5      # Upper temperature limit for the sweep over temperatures
TEMPERATURE_STEPS: int = 15             # Default: 111      # Number of temperature values to simualte

SIMULATION_BATCH_COUNT: int = 8         # Default: 4        # Number of simulation batches to perform per temperature
BATCH_CPU_CORE_LIMIT: int = 7           # Default: None     # Limit the number of CPU cores to a specified number

SIMULATION_SWEEP_COUNT: int = 1300      # Default: 1300     # Number of sweeps to perform in all simulations. Values are collected at the end of every sweep
EQUILIBRATION_SWEEP_COUNT: int = 1000   # Default: 1000     # Number of sweeps during which data is not collected to give the system time to reach equilibrium
ITERATIONS_PER_SWEEP: int = 10000       # Default: 10000    # Number of spin-flip-attempts per sweep

simulation_parameters = [
    TEMPERATURE_STEPS,                    
    SIMULATION_BATCH_COUNT,                      
    BATCH_CPU_CORE_LIMIT,                   
    INITIAL_DOWN_PROBABILITY,
    SIMULATION_SWEEP_COUNT,
    EQUILIBRATION_SWEEP_COUNT,
    np.asarray([START_TEMPERATURE, END_TEMPERATURE]),
    LATTICE_SIDE_SIZE,
    ITERATIONS_PER_SWEEP,
    RANDOMNESS_SEED,
    DATA_PATH,
    FIGURE_PATH,
    SAVE_DATA,
    SAVE_FIGURE
]

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
def get_reduced_energy_difference_at_position(position, spin_matrix: np.ndarray, n: int):
    '''Function to return the reduced energy difference that would result from a flip of a spin at position position.
    It calculates the interaction energy of the spin with its 4 nearest neighbours before and after the flip.
    The output is the actual energy E over the coupling constant J as J is not needed for the next step.'''
    x = position[0]
    y = position[1]
    nearest_neighbour_sum = spin_matrix[(x+1)%n, y] + spin_matrix[(x-1)%n, y] + spin_matrix[x, (y+1)%n] + spin_matrix[x, (y-1)%n]
    return 2 * spin_matrix[x, y] * nearest_neighbour_sum

@jit(nopython=True)
def metropolis_algorithm_step(spin_matrix: np.ndarray, T: float, n: int):
    '''Function that performs 1 step in the metropolis algorithm.
    It picks a random position and calculates the energy needed to flip the spin at that position.
    It then checks whether the energy is negative, at which point it flips the spin,
    or whether the energy is positive, then it uses the Boltzmann distribution probability to determine whether the flip happens.'''
    random_position = np.array([np.random.randint(0, n), np.random.randint(0, n)])
    flip_energy = get_reduced_energy_difference_at_position(random_position, spin_matrix, n)
    if flip_energy <= 0 or np.random.random() < np.exp(-flip_energy / T):
        spin_matrix[random_position[0], random_position[1]] *= -1

@jit(nopython=True)
def get_reduced_system_energy(spin_matrix: np.ndarray):
    '''Function to return the reduced energy for the whole spin system using Numba-compatible loops.'''
    n = spin_matrix.shape[0]
    vert_sum = 0
    horiz_sum = 0
    for i in range(n):
        for j in range(n):
            s = spin_matrix[i, j]
            vert_sum += s * spin_matrix[(i + 1) % n, j]
            horiz_sum += s * spin_matrix[i, (j + 1) % n]
    return -(vert_sum + horiz_sum)

@jit(nopython=True)
def get_magnetisation_per_spin(spin_matrix: np.ndarray):
    '''Function that calculates the magnetisation per spin by summing all spins and dividing by the number of spins.'''
    return np.sum(spin_matrix) / spin_matrix.size

@jit(nopython=True)
def get_iteration_quantities(spin_matrix: np.ndarray):
    '''Function that aggregates the magnetisation per spin, the system energy and squared system energy and outputs them.'''
    mag_per_spin = get_magnetisation_per_spin(spin_matrix)
    E = get_reduced_system_energy(spin_matrix)
    return mag_per_spin, E, E**2

@jit(nopython=True)
def update_aggregate_quantities(spin_matrix: np.ndarray, agg_m: float, agg_E: float, agg_E2: float):
    '''Function that appends the running total of the magnetisation per spin, the system energy and squared system energy.'''
    m, E, E2 = get_iteration_quantities(spin_matrix)
    return agg_m + m, agg_E + E, agg_E2 + E2

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
    '''Function that performs a simulation with a seed and initial down_probability for the spin matrix.
    It runs the simulation for iterations times sweeps metropolis algorithm steps.
    The average quantities are calculated from the burn_in_iterations iteration forwards at interval of sweeps.
    It returns the average magnetisation, average energy per spin and a heat capacity estimate.'''
    np.random.seed(random_seed)
        
    agg_m = 0.0
    agg_E = 0.0
    agg_E2 = 0.0
    spin_matrix: np.ndarray = generate_initial_spin_orientations(down_probability, n, 0) # Generate initial spin matrix

    effective_samples = max(1, sweeps - burn_in_sweeps) # Get number of sweeps with data collection

    # Perform sweeps
    for i in range(sweeps):
        for _ in range(iterations):
            metropolis_algorithm_step(spin_matrix, T, n)
        if i >= burn_in_sweeps:
            agg_m, agg_E, agg_E2 = update_aggregate_quantities(spin_matrix, agg_m, agg_E, agg_E2)

    # Return average quantities from simulation
    return get_average_quantities(agg_m, agg_E, agg_E2, effective_samples, T, n**2)

def run_simulation_task(args):
    """Worker function top-level wrapper for multiprocessing."""
    return simulation(*args)

def meta_simulation(batch_count: int, down_probability: float, sweeps: int, burn_in_sweeps: int, T: float, n: int, iterations: int, stop_event: threading.Event, core_limit: int | None, rng_seed: int | None = None):
    '''Function that performs multiple simulations and aggregates meta averages from the simulation averages.
    This function is optimized for multithreadding so that we can run an adequate amount of simulations in the 20 minutes.'''
    # Get number of usable cores per batch, either user defined maximum or maximum available cores
    num_cores = min(max(1, (os.cpu_count() or 1) - 1), core_limit) if core_limit else max(1, (os.cpu_count() or 1) - 1)
    total_sims = batch_count * num_cores # Total number of simualtions to perform

    # generate seed for simulations based on set seed or randomly
    simulation_seeds = np.random.default_rng(rng_seed).integers(1, None, size=total_sims)
        
    meta_agg_m = 0.0
    meta_agg_E = 0.0
    meta_agg_C = 0.0

    # generate simualtion tasks for multithreadding
    tasks = [
        (int(simulation_seeds[i]), down_probability, sweeps, burn_in_sweeps, T, n, iterations)
        for i in range(total_sims)
    ]

    # Execute all tasks where the amount of simulations ran is equal to the number of cores of the machine it is running on.
    # IMPORTANT: We currently limit this to 7 to reflefct the TA's machine. 
    # In case this is ran on a machine with less cores, the simulation results will differ.
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(run_simulation_task, task) for task in tasks]
        for sim_counter, future in enumerate(futures, 1):
            if stop_event.is_set():
                break
            m, E, C = future.result()
            meta_agg_m += m
            meta_agg_E += E
            meta_agg_C += C
            print(f'Completed simulation {sim_counter}/{total_sims}')
        
    return meta_agg_m / total_sims, meta_agg_E / total_sims, meta_agg_C / total_sims

def meta_meta_simulation(value_count: int, batch_count: int,  core_limit: int | None, down_probability: float, sweeps: int, burn_in_sweeps: int, temp_range: np.ndarray, n: int, iterations: int, rand_seed: int | None, DATA_PATH: str, FIGURE_PATH: str, SAVE_DATA: bool, SAVE_FIGURE: bool):
    '''Function that sweeps across temperature range given by temp_range and performs a metasimulation with batch_count * # available cores simulations.
    It then displayes the values of averagre absolute magnetisation and heat capacity for the reduced temperature values.'''
    m_data: np.ndarray = np.zeros(value_count)
    C_data: np.ndarray = np.zeros(value_count)

    # Get temperatures
    temps: np.ndarray = np.linspace(temp_range[0], temp_range[1], value_count, endpoint=True)
    
    # Get relevant values from metasimulation per temperature
    for i in range(value_count):
        print(f'Starting temperature {i+1}/{value_count}')
        m_data[i], _, C_data[i] = meta_simulation(batch_count, down_probability, sweeps, burn_in_sweeps, temps[i], n, iterations, stop_event, core_limit, rand_seed)
        print(f' reduced temperature is: {temps[i]}')
        print(f' average magnetisation is: {m_data[i]}')
        print(f' average heat capacity is: {C_data[i]}')

    # Plot relevant quantities and save to a figure if this is set to true at the top
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
        plt.close()
    # Saves the data to a csv file if the responding value at the top is set to True. Name of the file is a timestamp
    if SAVE_DATA:
        os.makedirs(DATA_PATH, exist_ok=True)
        file_path = os.path.join(DATA_PATH, f'{timestamp}.csv')

        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['temps', 'abs_magnetisation', 'Heat Capacity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(temps)):
                writer.writerow({
                    'temps': temps[i],
                    'abs_magnetisation': m_data[i],
                    'Heat Capacity': C_data[i],
                })

def main():
    '''Main function that runs the simulations.'''
    meta_meta_simulation(*simulation_parameters)

if __name__=="__main__":
    '''Execution helper.'''
    main()