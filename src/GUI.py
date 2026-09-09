from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import PySimpleGUI as sg
import matplotlib
import threading
import time
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os

rng = np.random.default_rng()

# --- Single Simulation Global Variables ---
T_red: float = 1                    # Reduced temperature
n: int = 20                         # Sqrt of number of spins
N: int = n*n                        # Number of spins
down_probability = 0.5

update_sleep: float = 0.01          # Update sleep duration in seconds
sweep_steps: int = N                # Steps per sweep (defaults to N)
burn_in_iteration_count: int = 0    # Number of iterations before average values are collected

iteration_count = 0

# --- Multi Simulation Global Variables ---
multi_T_red: float = 1
multi_n: int = 20
multi_N: int = multi_n * multi_n
multi_down_probability = 0.5

multi_update_sleep: float = 0.01
multi_sweep_steps: int = multi_N
multi_burn_in_iteration_count: int = 0

steps_per_simulation: int = 1000
simulation_count = 0

def generate_initial_spin_orientations(down_probability: float, n: int, local_rng=None):
    if local_rng is None:
        local_rng = rng
    random_valued_matrix = local_rng.random((n, n))
    spins = np.ones((n, n), dtype=int)
    spins[random_valued_matrix < down_probability] = -1
    return spins

def get_reduced_energy_difference_at_position(position, spin_matrix: np.ndarray, n: int):
    x, y = position
    nearest_neighbour_sum = spin_matrix[(x+1)%n][y] + spin_matrix[(x-1)%n][y] + spin_matrix[x][(y+1)%n] + spin_matrix[x][(y-1)%n]
    return 2 * spin_matrix[x][y] * nearest_neighbour_sum

def metropolis_algorithm_step(spin_matrix: np.ndarray, T: float, n: int, local_rng=None):
    if local_rng is None:
        local_rng = rng
    random_position = local_rng.integers(0, n, size=2)
    flip_energy = get_reduced_energy_difference_at_position(random_position, spin_matrix, n)
    if flip_energy <= 0 or local_rng.random() < np.exp(-flip_energy / T):
        spin_matrix[random_position[0]][random_position[1]] *= -1

def get_reduced_system_energy(spin_matrix: np.ndarray):
    vertical_pairs = spin_matrix * np.roll(spin_matrix, -1, axis=0)
    horizontal_pairs = spin_matrix * np.roll(spin_matrix, -1, axis=1)
    return -(np.sum(vertical_pairs) + np.sum(horizontal_pairs))

def get_magnetisation_per_spin(spin_matrix: np.ndarray):
    return np.sum(spin_matrix) / spin_matrix.size

def get_iteration_quantities(spin_matrix: np.ndarray):
    mag_per_spin = get_magnetisation_per_spin(spin_matrix)
    E = get_reduced_system_energy(spin_matrix)
    return mag_per_spin, E, E**2

def update_aggregate_quantities(spin_matrix: np.ndarray, agg_m: float, agg_E: float, agg_E2: float):
    m, E, E2 = get_iteration_quantities(spin_matrix)
    return agg_m + m, agg_E + E, agg_E2 + E2

def get_average_quantities(agg_m: float, agg_E: float, agg_E2: float, iterations: int, T: float, N: int):
    if iterations <= 0:
        return 0.0, 0.0, 0.0
    avg_m = agg_m / iterations
    avg_E = agg_E / iterations
    avg_E2 = agg_E2 / iterations
    heat_capacity = (avg_E2 - avg_E**2) / (T**2)

    return avg_m, avg_E / N, heat_capacity

def simulation(random_seed: int, down_probability: float, iterations: int, burn_in_iterations: int, T: float, n: int, sweeps: int):
    local_rng = np.random.default_rng(random_seed if random_seed != 0 else None)
    agg_m = 0.0
    agg_E = 0.0
    agg_E2 = 0.0
    spin_matrix: np.ndarray = generate_initial_spin_orientations(down_probability, n, local_rng)
    
    effective_samples = max(1, iterations - burn_in_iterations)
    
    for i in range(iterations):
        for _ in range(sweeps):
            metropolis_algorithm_step(spin_matrix, T, n, local_rng)
        if i >= burn_in_iterations:
            agg_m, agg_E, agg_E2 = update_aggregate_quantities(spin_matrix, agg_m, agg_E, agg_E2)

    return get_average_quantities(agg_m, agg_E, agg_E2, effective_samples, T, n**2)

def run_simulation_task(args):
    """Worker function top-level wrapper for multiprocessing."""
    return simulation(*args)

spin_matrix: np.ndarray = generate_initial_spin_orientations(down_probability, n)
single_sim_agg_m = 0
single_sim_agg_E = 0
single_sim_agg_E2 = 0
single_sim_heat_capacity = 0

# Lists to keep track of iterations and computed quantities over time
iterations_history = []
m_history = []
E_history = []
C_history = []

matplotlib.use('TkAgg')

def reset_single_simulation():
    global spin_matrix, iteration_count, single_sim_agg_m, single_sim_agg_E, single_sim_agg_E2, single_sim_heat_capacity
    global iterations_history, m_history, E_history, C_history
    spin_matrix = generate_initial_spin_orientations(down_probability, n)
    iteration_count = 0
    single_sim_agg_m = 0
    single_sim_agg_E = 0
    single_sim_agg_E2 = 0
    single_sim_heat_capacity = 0.0
    iterations_history.clear()
    m_history.clear()
    E_history.clear()
    C_history.clear()

def acquisition_thread(window, stop_event):
    """Generate/acquire data continuously in the background."""

    global iteration_count, single_sim_agg_m, single_sim_agg_E, single_sim_agg_E2, single_sim_heat_capacity

    while not stop_event.is_set():

        for _ in range(sweep_steps):
            metropolis_algorithm_step(spin_matrix, T_red, n)
        iteration_count += 1
        current_single_sim_mag, current_single_sim_E, _ = get_iteration_quantities(spin_matrix)
        if iteration_count > burn_in_iteration_count:
            single_sim_agg_m, single_sim_agg_E, single_sim_agg_E2 = update_aggregate_quantities(spin_matrix, single_sim_agg_m, single_sim_agg_E, single_sim_agg_E2)
            _, _, single_sim_heat_capacity = get_average_quantities(single_sim_agg_m, single_sim_agg_E, single_sim_agg_E2, iteration_count - burn_in_iteration_count, T_red, N)

        iterations_history.append(iteration_count)
        m_history.append(current_single_sim_mag)
        E_history.append(current_single_sim_E / N)
        C_history.append(single_sim_heat_capacity)

        window.write_event_value(
            '-DATA-',
            (spin_matrix.copy(), list(iterations_history), list(m_history), list(E_history), list(C_history))
        )

        time.sleep(update_sleep)

multi_sim_agg_m = 0
multi_sim_agg_E = 0
multi_sim_agg_C = 0

# Lists to keep track of iterations and computed quantities over time
simulation_history = []
multi_m_history = []
multi_E_history = []
multi_C_history = []

def reset_multi_simulation():
    global simulation_count, multi_sim_agg_m, multi_sim_agg_E, multi_sim_agg_C
    global simulation_history, multi_m_history, multi_E_history, multi_C_history
    simulation_count = 0
    multi_sim_agg_m = 0
    multi_sim_agg_E = 0
    multi_sim_agg_C = 0
    simulation_history.clear()
    multi_m_history.clear()
    multi_E_history.clear()
    multi_C_history.clear()

def multi_acquisition_thread(window, stop_event):
    """Generate/acquire data continuously in the background using multi-core processing."""

    global simulation_count, multi_sim_agg_m, multi_sim_agg_E, multi_sim_agg_C

    multi_sim_agg_m = 0
    multi_sim_agg_E = 0
    multi_sim_agg_C = 0

    num_cores = max(1, (os.cpu_count() or 1) - 1)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        while not stop_event.is_set():

            tasks = [
                (0, multi_down_probability, steps_per_simulation, multi_burn_in_iteration_count, multi_T_red, multi_n, multi_sweep_steps)
                for _ in range(num_cores)
            ]

            futures = [executor.submit(run_simulation_task, task) for task in tasks]

            for future in futures:
                if stop_event.is_set():
                    break
                current_sim_agg_m, current_sim_agg_E, current_sim_agg_C = future.result()

                multi_sim_agg_m += current_sim_agg_m
                multi_sim_agg_E += current_sim_agg_E
                multi_sim_agg_C += current_sim_agg_C
                simulation_count += 1

                simulation_history.append(simulation_count)
                multi_m_history.append(multi_sim_agg_m / simulation_count)
                multi_E_history.append(multi_sim_agg_E / simulation_count)
                multi_C_history.append(multi_sim_agg_C / simulation_count)

            window.write_event_value(
                '-MULTI-DATA-',
                (list(simulation_history), list(multi_m_history), list(multi_E_history), list(multi_C_history))
            )

            time.sleep(multi_update_sleep)

def create_figure(canvas):

    fig = Figure(figsize=(8, 6), dpi=100)
    
    # 2x2 subplot layout
    ax_spin = fig.add_subplot(221)
    ax_m = fig.add_subplot(222)
    ax_E = fig.add_subplot(223)
    ax_C = fig.add_subplot(224)

    image = ax_spin.imshow(
        spin_matrix,
        cmap='gray',
        vmin=-1,
        vmax=1
    )

    ax_spin.set_title("Spin visualization")
    ax_spin.set_xticks([])
    ax_spin.set_yticks([])

    # Configure line plots
    line_m, = ax_m.plot([], [], 'r-')
    ax_m.set_title("Magnetisation / Spin")
    ax_m.set_xlabel("Iteration")

    line_E, = ax_E.plot([], [], 'b-')
    ax_E.set_title("Energy / Spin")
    ax_E.set_xlabel("Iteration")

    line_C, = ax_C.plot([], [], 'g-')
    ax_C.set_title("Heat Capacity Estimate")
    ax_C.set_xlabel("Iteration")

    fig.tight_layout()

    figure_canvas_agg = FigureCanvasTkAgg(
        fig,
        canvas
    )

    figure_canvas_agg.draw()

    figure_canvas_agg.get_tk_widget().pack(
        side='top',
        fill='both',
        expand=1
    )

    lines = (line_m, line_E, line_C)
    axes = (ax_m, ax_E, ax_C)

    return image, figure_canvas_agg, lines, axes

def create_multi_figure(canvas):

    fig = Figure(figsize=(8, 6), dpi=100)

    # 1x3 subplot layout for multi-simulation averages
    ax_m = fig.add_subplot(131)
    ax_E = fig.add_subplot(132)
    ax_C = fig.add_subplot(133)

    # Configure line plots
    line_m, = ax_m.plot([], [], 'r-')
    ax_m.set_title("Avg Magnetisation")
    ax_m.set_xlabel("Simulation")

    line_E, = ax_E.plot([], [], 'b-')
    ax_E.set_title("Avg Energy / Spin")
    ax_E.set_xlabel("Simulation")

    line_C, = ax_C.plot([], [], 'g-')
    ax_C.set_title("Avg Heat Capacity")
    ax_C.set_xlabel("Simulation")

    fig.tight_layout()

    figure_canvas_agg = FigureCanvasTkAgg(
        fig,
        canvas
    )

    figure_canvas_agg.draw()

    figure_canvas_agg.get_tk_widget().pack(
        side='top',
        fill='both',
        expand=1
    )

    lines = (line_m, line_E, line_C)
    axes = (ax_m, ax_E, ax_C)

    return figure_canvas_agg, lines, axes

single_simulation_tab_layout = [
    # Row 1
    [
        sg.Text('Reduced T:', size=(18, 1)),
        sg.Input(str(T_red), key='-T-', size=(10, 1)),

        sg.Text('Steps per sweep:', size=(18, 1)),
        sg.Input(str(sweep_steps), key='-STEPS-', size=(10, 1)),

        sg.Text('Update sleep (s):', size=(18, 1)),
        sg.Input(str(update_sleep), key='-SLEEP-', size=(10, 1))
    ],

    # Row 2
    [
        sg.Text('Down probability:', size=(18, 1)),
        sg.Input(str(down_probability), key='-DOWN-', size=(10, 1)),

        sg.Text('Sweep Burn-in count:', size=(18, 1)),
        sg.Input(str(burn_in_iteration_count), key='-BURNIN-', size=(10, 1))
    ],

    # Row 3
    [
        sg.Text('Lattice size n:', size=(18, 1)),
        sg.Input(str(n), key='-N-', size=(10, 1))
    ],

    # Row 4: Buttons
    [
        sg.Button('Start', key='-START-'),
        sg.Button('Stop', key='-STOP-'),
        sg.Button('Reset', key='-RESET-'),
        sg.Button('Exit', key='-EXIT-')
    ],

    # Row 5: Iteration display
    [
        sg.Text('Iteration:', font=('Any', 12)),
        sg.Text('0', key='-ITERATION-', size=(15, 1), font=('Any', 12))
    ],

    # Row 6: Canvas
    [
        sg.Canvas(size=(800, 600), key='-CANVAS-', expand_x=True, expand_y=True)
    ]
]

multi_simulation_tab_layout = [
    # Row 1
    [
        sg.Text('Reduced T:', size=(18, 1)),
        sg.Input(str(multi_T_red), key='-MULTI-T-', size=(10, 1)),

        sg.Text('Steps per sweep:', size=(18, 1)),
        sg.Input(str(multi_sweep_steps), key='-MULTI-STEPS-', size=(10, 1)),

        sg.Text('Update sleep (s):', size=(18, 1)),
        sg.Input(str(multi_update_sleep), key='-MULTI-SLEEP-', size=(10, 1))
    ],

    # Row 2
    [
        sg.Text('Down probability:', size=(18, 1)),
        sg.Input(str(multi_down_probability), key='-MULTI-DOWN-', size=(10, 1)),

        sg.Text('Sweep Burn-in count:', size=(18, 1)),
        sg.Input(str(multi_burn_in_iteration_count), key='-MULTI-BURNIN-', size=(10, 1))
    ],

    # Row 3
    [
        sg.Text('Lattice size n:', size=(18, 1)),
        sg.Input(str(multi_n), key='-MULTI-N-', size=(10, 1)),

        sg.Text('Sweeps per simulation:', size=(18, 1)),
        sg.Input(str(steps_per_simulation), key='-MULTI-STEPS-PER-SIM-', size=(10, 1))
    ],

    # Row 4: Buttons
    [
        sg.Button('Start', key='-MULTI-START-'),
        sg.Button('Stop', key='-MULTI-STOP-'),
        sg.Button('Reset', key='-MULTI-RESET-'),
        sg.Button('Exit', key='-MULTI-EXIT-')
    ],

    # Row 5: Simulation count display
    [
        sg.Text('Simulation count:', font=('Any', 12)),
        sg.Text('0', key='-MULTI-SIM-COUNT-', size=(15, 1), font=('Any', 12))
    ],

    # Row 6: Canvas
    [
        sg.Canvas(size=(800, 600), key='-MULTI-CANVAS-', expand_x=True, expand_y=True)
    ]
]

layout = [
    [
        sg.TabGroup(
            [
                [
                    sg.Tab('Single simulation', single_simulation_tab_layout),
                    sg.Tab('Multi-simulation', multi_simulation_tab_layout)
                ]
            ],
            expand_x=True,
            expand_y=True
        )
    ]
]

if __name__ == '__main__':
    window = sg.Window(
        'Ising Model Simulation',
        layout,
        finalize=True,
        resizable=True
    )

    image, fig_agg, (line_m, line_E, line_C), (ax_m, ax_E, ax_C) = create_figure(
        window['-CANVAS-'].TKCanvas
    )

    multi_fig_agg, (multi_line_m, multi_line_E, multi_line_C), (multi_ax_m, multi_ax_E, multi_ax_C) = create_multi_figure(
        window['-MULTI-CANVAS-'].TKCanvas
    )

    stop_event = threading.Event()
    worker = None

    multi_stop_event = threading.Event()
    multi_worker = None

    while True:

        event, values = window.read()

        if event in (sg.WIN_CLOSED, '-EXIT-', '-MULTI-EXIT-'):
            stop_event.set()
            multi_stop_event.set()
            break

        if event == '-START-':

            # Read parameters from GUI
            try:
                T_red = float(values['-T-'])
                n = int(values['-N-'])
                down_probability = float(values['-DOWN-'])
                update_sleep = float(values['-SLEEP-'])
                sweep_steps = int(values['-STEPS-'])
                burn_in_iteration_count = int(values['-BURNIN-'])

                if T_red <= 0:
                    raise ValueError("T must be greater than 0")

                if n <= 0:
                    raise ValueError("n must be greater than 0")

                if not 0 <= down_probability <= 1:
                    raise ValueError(
                        "Down probability must be between 0 and 1"
                    )

                if update_sleep < 0:
                    raise ValueError(
                        "Update sleep must be 0 or greater"
                    )

                if sweep_steps <= 0:
                    raise ValueError(
                        "Steps per sweep must be greater than 0"
                    )

                if burn_in_iteration_count < 0:
                    raise ValueError(
                        "Burn-in count must be 0 or greater"
                    )

            except ValueError as e:

                sg.popup_error(
                    "Invalid input",
                    str(e)
                )

                continue

            # Update global simulation values
            N = n * n

            # Start/restart only if no worker is running
            if worker is None or not worker.is_alive():

                reset_single_simulation()
                stop_event.clear()

                worker = threading.Thread(
                    target=acquisition_thread,
                    args=(window, stop_event),
                    daemon=True
                )

                worker.start()

        elif event == '-STOP-':

            stop_event.set()

        elif event == '-RESET-':

            # Stop current simulation
            stop_event.set()

            # Read new parameters
            try:
                T_red = float(values['-T-'])
                n = int(values['-N-'])
                down_probability = float(values['-DOWN-'])
                update_sleep = float(values['-SLEEP-'])
                sweep_steps = int(values['-STEPS-'])
                burn_in_iteration_count = int(values['-BURNIN-'])

                if T_red <= 0:
                    raise ValueError("T must be greater than 0")

                if n <= 0:
                    raise ValueError("n must be greater than 0")

                if not 0 <= down_probability <= 1:
                    raise ValueError(
                        "Down probability must be between 0 and 1"
                    )

                if update_sleep < 0:
                    raise ValueError(
                        "Update sleep must be 0 or greater"
                    )

                if sweep_steps <= 0:
                    raise ValueError(
                        "Steps per sweep must be greater than 0"
                    )

                if burn_in_iteration_count < 0:
                    raise ValueError(
                        "Burn-in count must be 0 or greater"
                    )

            except ValueError as e:

                sg.popup_error(
                    "Invalid input",
                    str(e)
                )

                continue

            N = n * n

            reset_single_simulation()

            window['-ITERATION-'].update('0')

            # Update image and reset lines
            image.set_data(spin_matrix)
            line_m.set_data([], [])
            line_E.set_data([], [])
            line_C.set_data([], [])

            fig_agg.draw_idle()

        elif event == '-MULTI-START-':

            # Read multi-simulation parameters from GUI
            try:
                multi_T_red = float(values['-MULTI-T-'])
                multi_n = int(values['-MULTI-N-'])
                multi_down_probability = float(values['-MULTI-DOWN-'])
                multi_update_sleep = float(values['-MULTI-SLEEP-'])
                multi_sweep_steps = int(values['-MULTI-STEPS-'])
                multi_burn_in_iteration_count = int(values['-MULTI-BURNIN-'])
                steps_per_simulation = int(values['-MULTI-STEPS-PER-SIM-'])

                if multi_T_red <= 0:
                    raise ValueError("T must be greater than 0")

                if multi_n <= 0:
                    raise ValueError("n must be greater than 0")

                if not 0 <= multi_down_probability <= 1:
                    raise ValueError("Down probability must be between 0 and 1")

                if multi_update_sleep < 0:
                    raise ValueError("Update sleep must be 0 or greater")

                if multi_sweep_steps <= 0:
                    raise ValueError("Steps per sweep must be greater than 0")

                if multi_burn_in_iteration_count < 0:
                    raise ValueError("Burn-in count must be 0 or greater")

                if steps_per_simulation <= 0:
                    raise ValueError("Steps per simulation must be greater than 0")

            except ValueError as e:

                sg.popup_error(
                    "Invalid input",
                    str(e)
                )

                continue

            multi_N = multi_n * multi_n

            if multi_worker is None or not multi_worker.is_alive():

                reset_multi_simulation()
                multi_stop_event.clear()

                multi_worker = threading.Thread(
                    target=multi_acquisition_thread,
                    args=(window, multi_stop_event),
                    daemon=True
                )

                multi_worker.start()

        elif event == '-MULTI-STOP-':

            multi_stop_event.set()

        elif event == '-MULTI-RESET-':

            multi_stop_event.set()

            try:
                multi_T_red = float(values['-MULTI-T-'])
                multi_n = int(values['-MULTI-N-'])
                multi_down_probability = float(values['-MULTI-DOWN-'])
                multi_update_sleep = float(values['-MULTI-SLEEP-'])
                multi_sweep_steps = int(values['-MULTI-STEPS-'])
                multi_burn_in_iteration_count = int(values['-MULTI-BURNIN-'])
                steps_per_simulation = int(values['-MULTI-STEPS-PER-SIM-'])

                if multi_T_red <= 0:
                    raise ValueError("T must be greater than 0")

                if multi_n <= 0:
                    raise ValueError("n must be greater than 0")

                if not 0 <= multi_down_probability <= 1:
                    raise ValueError("Down probability must be between 0 and 1")

                if multi_update_sleep < 0:
                    raise ValueError("Update sleep must be 0 or greater")

                if multi_sweep_steps <= 0:
                    raise ValueError("Steps per sweep must be greater than 0")

                if multi_burn_in_iteration_count < 0:
                    raise ValueError("Burn-in count must be 0 or greater")

                if steps_per_simulation <= 0:
                    raise ValueError("Sweeps per simulation must be greater than 0")

            except ValueError as e:

                sg.popup_error(
                    "Invalid input",
                    str(e)
                )

                continue

            multi_N = multi_n * multi_n
            reset_multi_simulation()

            window['-MULTI-SIM-COUNT-'].update('0')

            multi_line_m.set_data([], [])
            multi_line_E.set_data([], [])
            multi_line_C.set_data([], [])

            multi_fig_agg.draw_idle()

        elif event == '-DATA-':

            new_matrix, iters, m_vals, E_vals, C_vals = values['-DATA-']

            # Update spin matrix image
            image.set_data(new_matrix)

            # Update live trend lines
            line_m.set_data(iters, m_vals)
            ax_m.relim()
            ax_m.autoscale_view()

            line_E.set_data(iters, E_vals)
            ax_E.relim()
            ax_E.autoscale_view()

            line_C.set_data(iters, C_vals)
            ax_C.relim()
            ax_C.autoscale_view()

            # Update iteration counter
            window['-ITERATION-'].update(
                str(iteration_count)
            )

            # Redraw canvas
            fig_agg.draw_idle()

        elif event == '-MULTI-DATA-':

            sims, m_vals, E_vals, C_vals = values['-MULTI-DATA-']

            # Update multi-simulation trend lines
            multi_line_m.set_data(sims, m_vals)
            multi_ax_m.relim()
            multi_ax_m.autoscale_view()

            multi_line_E.set_data(sims, E_vals)
            multi_ax_E.relim()
            multi_ax_E.autoscale_view()

            multi_line_C.set_data(sims, C_vals)
            multi_ax_C.relim()
            multi_ax_C.autoscale_view()

            # Update simulation counter
            window['-MULTI-SIM-COUNT-'].update(
                str(simulation_count)
            )

            # Redraw canvas
            multi_fig_agg.draw_idle()

    window.close()