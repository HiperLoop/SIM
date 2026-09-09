import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

T_red = 1 #Reduced temperature
n:int = 20 #Sqrt of number of spins
N:int = n*n #Number of spins

def show_spins(spin_matrix: np.ndarray):
    plt.imshow(spin_matrix)
    plt.show()

def generate_initial_spin_orientations(down_probability: float):
    random_valued_matrix = rng.random((n, n))
    random_valued_matrix[random_valued_matrix < down_probability] = -1
    random_valued_matrix[random_valued_matrix >= down_probability] = 1
    return random_valued_matrix.astype(int)

def get_reduced_energy_difference_at_position(position: int[2], spin_matrix: np.ndarray):
    x, y = position
    nearest_neighbour_sum = spin_matrix[(x+1)%n][y] + spin_matrix[(x-1)%n][y] + spin_matrix[x][(y+1)%n] + spin_matrix[x][(y-1)%n]
    return 2 * spin_matrix[x][y] * nearest_neighbour_sum

def metropolis_algorithm_step(spin_matrix: np.ndarray):
    random_position: int[2] = rng.integers(0, n, size = 2)
    flip_energy = get_reduced_energy_difference_at_position(random_position, spin_matrix)
    if flip_energy <= 0 or rng.random() < np.exp(-flip_energy / T_red):
        spin_matrix[random_position[0]][random_position[1]]*=-1

def get_reduced_system_energy(spin_matrix: np.ndarray):
    vertical_pairs = spin_matrix * np.roll(spin_matrix, -1, axis=0)
    horizontal_pairs = spin_matrix * np.roll(spin_matrix, -1, axis=1)
    return -(np.sum(vertical_pairs) + np.sum(horizontal_pairs))

def get_magnetisation_per_spin(spin_matrix: np.ndarray):
    return np.sum(spin_matrix) / N

def get_iteration_quantities(spin_matrix: np.ndarray):
    mag_per_spin = get_magnetisation_per_spin(spin_matrix)
    E = get_reduced_system_energy(spin_matrix)
    return mag_per_spin, E, E**2

def update_aggregate_quantities(spin_matrix: np.ndarray, agg_m: float, agg_E: float, agg_E2: float):
    m, E, E2 = get_iteration_quantities(spin_matrix)
    return agg_m + m, agg_E + E, agg_E2 + E2

def get_average_quantities(agg_m: float, agg_E: float, agg_E2: float, iterations: int):
    avg_m = agg_m / iterations
    avg_E = agg_E / iterations
    avg_E2 = agg_E2 / iterations
    heat_capacity = (avg_E2 - avg_E**2) / (T_red**2)

    return avg_m, avg_E / N, heat_capacity

def simulation(random_seed: int, down_probability: float, iterations: int):
    if random_seed == 0: np.random.seed(random_seed)
    agg_m = 0, agg_E = 0, agg_E2 = 0
    spin_matrix: np.ndarray = generate_initial_spin_orientations(down_probability)
    for _ in range(iterations):
        metropolis_algorithm_step(spin_matrix)
        agg_m, agg_E, agg_E2 = update_aggregate_quantities(spin_matrix, agg_m, agg_E, agg_E2)

    return get_average_quantities(agg_m, agg_E, agg_E2, iterations)

def meta_simulation(simulation_count: int, down_probability: float, iterations: int, simulation_seeds: np.ndarray = None):
    if not simulation_seeds: simulation_seeds = np.zeros(simulation_count)
    agg_m = 0, agg_E = 0, agg_C = 0
    for i in range(simulation_count):
        m, E, C = simulation(simulation_seeds[i], down_probability, iterations)
        agg_m += m
        agg_E += E
        agg_C += C
    return agg_m/simulation_count, agg_E/simulation_count, agg_C/simulation_count

def main():
    simulation(1, 0.5, 10)

if __name__=="__main__":
    main()