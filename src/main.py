import scipy.constants as const
import numpy as np

J:float = 1 #Coupling constant
T:float = 0 #Temperature
k_B:float = const.Boltzmann #Boltzman constant
T_red:float = (k_B * T) / J #Reduced temperature
n:int = 2 #Sqrt of number of spins
N:int = n*n #Number of spins

def generate_initial_spin_orientations(n: int, down_probability: float):
    random_valued_matrix = np.random.random((n, n))
    random_valued_matrix[random_valued_matrix < down_probability] = -1
    random_valued_matrix[random_valued_matrix >= down_probability] = 1
    return random_valued_matrix.astype(int)

def main():
    print(generate_initial_spin_orientations(n, 0.5))

if __name__=="__main__":
    main()