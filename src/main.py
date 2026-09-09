import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

T_red = 1 #Reduced temperature
n:int = 20 #Sqrt of number of spins
N:int = n*n #Number of spins

def show_spins(spin_matrix: int[n][n]):
    plt.imshow(spin_matrix)
    plt.show()

def generate_initial_spin_orientations(n: int, down_probability: float):
    random_valued_matrix = rng.random((n, n))
    random_valued_matrix[random_valued_matrix < down_probability] = -1
    random_valued_matrix[random_valued_matrix >= down_probability] = 1
    return random_valued_matrix.astype(int)

def get_reduced_energy_difference_at_position(n: int, position: int[2], spin_matrix: int[n][n]):
    x, y = position
    nearest_neighbour_sum = spin_matrix[(x+1)%n][y] + spin_matrix[(x-1)%n][y] + spin_matrix[x][(y+1)%n] + spin_matrix[x][(y-1)%n]
    return 2 * spin_matrix[x][y] * nearest_neighbour_sum

def metropolis_algorithm_step(n: int, spin_matrix: int[n][n]):
    random_position: int[2] = rng.integers(0, n, size = 2)
    flip_energy = get_reduced_energy_difference_at_position(n, random_position, spin_matrix)
    if flip_energy <= 0 or rng.random() < np.exp(-flip_energy / T_red):
        spin_matrix[random_position[0]][random_position[1]]*=-1

def simulation(random_seed: int, n: int, down_probability: float, iterations: int):
    np.random.seed(random_seed)
    spin_matrix: int[n][n] = generate_initial_spin_orientations(n, down_probability)
    for _ in range(iterations):
        metropolis_algorithm_step(n, spin_matrix)

def main():
    simulation(1, n, 0.5, 10)

if __name__=="__main__":
    main()