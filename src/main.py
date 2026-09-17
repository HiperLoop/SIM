import os
import threading
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

stop_event = threading.Event()
worker = None

rng = np.random.default_rng()   # randomiser initialisation

T_red = 1                       # Reduced temperature
n:int = 20                      # Sqrt of number of spins
N:int = n*n                     # Number of spins

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
def simulation(random_seed: int, down_probability: float, iterations: int, burn_in_iterations: int, T: float, n: int, sweeps: int):
    '''Function that performs a simulation with a seed and initial down_probability for the spin matrix.
    It runs the simulation for iterations times sweeps metropolis algorithm steps.
    The average quantities are calculated from the burn_in_iterations iteration forwards at interval of sweeps.
    It returns the average magnetisation, average energy per spin and a heat capacity estimate.'''
    if random_seed != 0:
        np.random.seed(random_seed)
        
    agg_m = 0.0
    agg_E = 0.0
    agg_E2 = 0.0
    spin_matrix: np.ndarray = generate_initial_spin_orientations(down_probability, n, 0)
    
    effective_samples = max(1, iterations - burn_in_iterations)
    
    for i in range(iterations):
        for _ in range(sweeps):
            metropolis_algorithm_step(spin_matrix, T, n)
        if i >= burn_in_iterations:
            agg_m, agg_E, agg_E2 = update_aggregate_quantities(spin_matrix, agg_m, agg_E, agg_E2)

    return get_average_quantities(agg_m, agg_E, agg_E2, effective_samples, T, n**2)

def run_simulation_task(args):
    """Worker function top-level wrapper for multiprocessing."""
    return simulation(*args)

def meta_simulation(batch_count: int, down_probability: float, iterations: int, burn_in_iterations: int, T: float, n: int, sweeps: int, stop_event: threading.Event, simulation_seeds: np.ndarray | None = None):
    '''Function that performs multiple simulations and aggregates meta averages from the simulation averages.'''
    num_cores = max(1, (os.cpu_count() or 1) - 1)
    total_sims = batch_count * num_cores
    if simulation_seeds is None: 
        simulation_seeds = rng.integers(0, 1000000, size=total_sims)
        
    meta_agg_m = 0.0
    meta_agg_E = 0.0
    meta_agg_C = 0.0
    
    tasks = [
        (int(simulation_seeds[i]), down_probability, iterations, burn_in_iterations, T, n, sweeps)
        for i in range(total_sims)
    ]
    
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

def meta_meta_simulation(value_count: int, batch_count: int, down_probability: float, iterations: int, burn_in_iterations: int, temp_range: np.ndarray, n: int, sweeps: int):
    m_data = np.zeros(value_count)
    C_data = np.zeros(value_count)
    temps: np.ndarray = np.linspace(temp_range[0], temp_range[1], value_count, endpoint=True)
    for i in range(value_count):
        print(f'Starting temperature {i+1}/{value_count}')
        m_data[i], _, C_data[i] = meta_simulation(batch_count, down_probability, iterations, burn_in_iterations, temps[i], n, sweeps, stop_event)
        print(f' reduced temperature is: {temps[i]}')
        print(f' average magnetisation is: {m_data[i]}')
        print(f' average heat capacity is: {C_data[i]}')

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
    plt.show()

def main():
    '''Main function that runs the simulation or meta_simulation.'''
    meta_meta_simulation(61, 2, 0.5, 1300, 1000, np.asarray([2, 2.5]), 50, 10000)    

if __name__=="__main__":
    '''Execution helper.'''
    main()