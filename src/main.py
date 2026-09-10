import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()   # randomiser initialisation

T_red = 1                       # Reduced temperature
n:int = 20                      # Sqrt of number of spins
N:int = n*n                     # Number of spins

def show_spins(spin_matrix: np.ndarray):
    '''Function to display the spin matrix as a rectangular field with colours corresponding to spin values'''
    plt.imshow(spin_matrix)
    plt.show()

def generate_initial_spin_orientations(down_probability: float):
    '''Function to initialise the n by n initial matrix with spins.
    Each spin has down_probability to be a spin down (-1) and 1-down_probability to be a spin up (1).
    It then returns the matrix as np.ndarray with size n x n and dytpe = int.'''
    random_valued_matrix = rng.random((n, n))
    random_valued_matrix[random_valued_matrix < down_probability] = -1
    random_valued_matrix[random_valued_matrix >= down_probability] = 1
    return random_valued_matrix.astype(int)

def get_reduced_energy_difference_at_position(position: np.ndarray, spin_matrix: np.ndarray):
    '''Function to return the reduced energy difference that would result from a flip of a spin at position position.
    It calculates the interaction energy of the spin with its 4 nearest neighbours before and after the flip.
    The output is the actual energy E over the coupling constant J as J is not needed for the next step.'''
    x, y = position
    nearest_neighbour_sum = spin_matrix[(x+1)%n][y] + spin_matrix[(x-1)%n][y] + spin_matrix[x][(y+1)%n] + spin_matrix[x][(y-1)%n]
    return 2 * spin_matrix[x][y] * nearest_neighbour_sum

def metropolis_algorithm_step(spin_matrix: np.ndarray):
    '''Function that performs 1 step in the metropolis algorithm.
    It picks a random position and calculates the energy needed to flip the spin at that position.
    It then checks whether the energy is negative, at which point it flips the spin,
    or whether the energy is positive, then it uses the Boltzmann distribution probability to determine whether the flip happens.'''
    random_position: np.ndarray = rng.integers(0, n, size = 2)
    flip_energy = get_reduced_energy_difference_at_position(random_position, spin_matrix)
    if flip_energy <= 0 or rng.random() < np.exp(-flip_energy / T_red):
        spin_matrix[random_position[0]][random_position[1]]*=-1

def get_reduced_system_energy(spin_matrix: np.ndarray):
    '''Function to return the reduced energy for the whole spin sytem.
        It calculates the interaction energy of all spin pairs,
        it then outputs the actual energy E over the coupling constant J as J is not needed for the next step.'''
    vertical_pairs = spin_matrix * np.roll(spin_matrix, -1, axis=0)
    horizontal_pairs = spin_matrix * np.roll(spin_matrix, -1, axis=1)
    return -(np.sum(vertical_pairs) + np.sum(horizontal_pairs))

def get_magnetisation_per_spin(spin_matrix: np.ndarray):
    '''Function that calculates the magnetisation per spin by summing all spins and dividing by the number of spins.'''
    return np.sum(spin_matrix) / N

def get_iteration_quantities(spin_matrix: np.ndarray):
    '''Function that aggregates the magnetisation per spin, the system energy and squared system energy and outputs them.'''
    mag_per_spin = get_magnetisation_per_spin(spin_matrix)
    E = get_reduced_system_energy(spin_matrix)
    return mag_per_spin, E, E**2

def update_aggregate_quantities(spin_matrix: np.ndarray, agg_m: float, agg_E: float, agg_E2: float):
    '''Function that appends the running total of the magnetisation per spin, the system energy and squared system energy.'''
    m, E, E2 = get_iteration_quantities(spin_matrix)
    return agg_m + m, agg_E + E, agg_E2 + E2

def get_average_quantities(agg_m: float, agg_E: float, agg_E2: float, iterations: int):
    '''Function that calculates the average values of the magnetisation per spin, the system energy and squared system energy after a number of iterations.'''
    avg_m = agg_m / iterations
    avg_E = agg_E / iterations
    avg_E2 = agg_E2 / iterations
    heat_capacity = (avg_E2 - avg_E**2) / (T_red**2)

    return avg_m, avg_E / N, heat_capacity

def simulation(random_seed: int, down_probability: float, iterations: int, burn_in_iterations):
    '''Function that performs a simulation with a seed and initial down_probability for the spin amtrix.
    It runs the simulation for iterations metropolis algorithm steps.
    The average quantities are calculated from the burn_in_iterations iteration forwards.
    It returns the average magnetisation, average enrgy per spin and a heat capacity estimate.'''
    if random_seed != 0: np.random.seed(random_seed)
    agg_m = 0
    agg_E = 0
    agg_E2 = 0
    spin_matrix: np.ndarray = generate_initial_spin_orientations(down_probability)
    for i in range(iterations):
        if i % 100 == 0: print(f'Iteration {i}/{iterations}')
        metropolis_algorithm_step(spin_matrix)
        if i > burn_in_iterations:
            agg_m, agg_E, agg_E2 = update_aggregate_quantities(spin_matrix, agg_m, agg_E, agg_E2)

    return get_average_quantities(agg_m, agg_E, agg_E2, iterations - burn_in_iterations)

def meta_simulation(simulation_count: int, down_probability: float, iterations: int, burn_in_iterations: int, simulation_seeds: np.ndarray = None):
    '''Function that performs multiple simulations and aggregates meta averages from the simulation averages.
    It does simulation_count simulations with initial spin matrix down probability being down_probability.
    The simulations run for iterations iterations and the avergae simulation values are recorded freom burn_in_terations forwards.
    The simualtions are initialised with simulation_seed random seed.
    The function returns simulation averaged magnetisation, energy per spin and heat capacity estimate.'''
    if not simulation_seeds: simulation_seeds = np.zeros(simulation_count)
    agg_m = 0
    agg_E = 0
    agg_C = 0
    for i in range(simulation_count):
        print(f'Simulation {i}/{simulation_count}')
        m, E, C = simulation(simulation_seeds[i], down_probability, iterations, burn_in_iterations)
        agg_m += m
        agg_E += E
        agg_C += C
    return agg_m/simulation_count, agg_E/simulation_count, agg_C/simulation_count

def main():
    '''Main function that runs the simulation or meta_simualtion.'''
    print(meta_simulation(5, 0.5, 10000))

if __name__=="__main__":
    '''Execution helper.'''
    main()