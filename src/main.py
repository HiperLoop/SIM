import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

J:float = 1 #Coupling constant
T:float = 1 #Temperature
k_B:float = const.Boltzmann #Boltzman constant
T_red:float = (k_B * T) / J #Reduced temperature
min_BT: float = -1/(k_B * T) #useful value for calculating flip probabilities
n:int = 4 #Sqrt of number of spins
N:int = n*n #Number of spins

def show_spins(spin_matrix: int[n][n]):
    plt.imshow(spin_matrix)
    plt.show()

def generate_initial_spin_orientations(n: int, down_probability: float):
    random_valued_matrix = rng.random((n, n))
    random_valued_matrix[random_valued_matrix < down_probability] = -1
    random_valued_matrix[random_valued_matrix >= down_probability] = 1
    return random_valued_matrix.astype(int)

def get_energy_at_position( J: float, n: int, position: int[2], spin_matrix: int[n][n]):
    x, y = position
    print(x)
    print(y)
    nearest_neighbour_sum = spin_matrix[(x+1)%n][y] + spin_matrix[(x-1)%n][y] + spin_matrix[x][(y+1)%n] + spin_matrix[x][(y-1)%n]
    return -J * spin_matrix[x][y] * nearest_neighbour_sum

def metropolis_algorithm(J: float, n: int, iterations: int, spin_matrix: int[n][n]):
    for _ in range(iterations):
        random_position: int[2] = rng.integers(0, n-1, size = 2)
        flip_energy = get_energy_at_position(J, n, random_position, spin_matrix)
        if flip_energy <= 0 or rng.random() < np.exp(flip_energy * min_BT):
            spin_matrix[random_position[0]][random_position[1]]*=-1

def simulation(random_seed: int, J: float, n: int, down_probability: float, iterations: int):
    np.random.seed(random_seed)
    spin_matrix: int[n][n] = generate_initial_spin_orientations(n, down_probability)
    metropolis_algorithm(J, n, iterations, spin_matrix)
    

def main():
    simulation(1, J, n, 0.5, 1)

if __name__=="__main__":
    main()