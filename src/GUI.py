from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import PySimpleGUI as sg
import matplotlib
import threading
import time
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

rng = np.random.default_rng()

T_red: float = 1                # Reduced temperature
n: int = 20                     # Sqrt of number of spins
N: int = n*n                    # Number of spins
down_probability = 0.5

# Update sleep duration in seconds
update_sleep: float = 0.01

iteration_count = 0

def generate_initial_spin_orientations(n: int, down_probability: float):
    random_valued_matrix = rng.random((n, n))
    random_valued_matrix[random_valued_matrix < down_probability] = -1
    random_valued_matrix[random_valued_matrix >= down_probability] = 1
    return random_valued_matrix.astype(int)

spin_matrix: int[n][n] = generate_initial_spin_orientations(n, down_probability)

def get_reduced_energy_difference_at_position(n: int, position: int[2], spin_matrix: int[n][n]):
    x, y = position
    nearest_neighbour_sum = spin_matrix[(x+1)%n][y] + spin_matrix[(x-1)%n][y] + spin_matrix[x][(y+1)%n] + spin_matrix[x][(y-1)%n]
    return 2 * spin_matrix[x][y] * nearest_neighbour_sum

def metropolis_algorithm_step(n: int, spin_matrix: int[n][n]):
    random_position: int[2] = rng.integers(0, n, size = 2)
    flip_energy = get_reduced_energy_difference_at_position(n, random_position, spin_matrix)
    if flip_energy <= 0 or rng.random() < np.exp(-flip_energy / T_red):
        spin_matrix[random_position[0]][random_position[1]]*=-1

matplotlib.use('TkAgg')

def acquisition_thread(window, stop_event):
    """Generate/acquire data continuously in the background."""

    global iteration_count

    while not stop_event.is_set():

        metropolis_algorithm_step(n, spin_matrix)

        iteration_count += 1

        window.write_event_value(
            '-DATA-',
            spin_matrix.copy()
        )

        time.sleep(update_sleep)

def create_figure(canvas):

    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)

    image = ax.imshow(
        spin_matrix,
        cmap='gray',
        vmin=-1,
        vmax=1
    )

    ax.set_title("Spin visualization")

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

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

    return image, figure_canvas_agg

layout = [
    [
        sg.Text('Reduced T:', size=(12, 1)),
        sg.Input(
            str(T_red),
            key='-T-',
            size=(10, 1)
        ),

        sg.Text('Lattice size n:', size=(12, 1)),
        sg.Input(
            str(n),
            key='-N-',
            size=(10, 1)
        )
    ],

    [
        sg.Text('Down probability:', size=(12, 1)),
        sg.Input(
            str(down_probability),
            key='-DOWN-',
            size=(10, 1)
        ),

        sg.Text('Update sleep (s):', size=(12, 1)),
        sg.Input(
            str(update_sleep),
            key='-SLEEP-',
            size=(10, 1)
        )
    ],

    [
        sg.Button('Start', key='-START-'),
        sg.Button('Stop', key='-STOP-'),
        sg.Button('Reset', key='-RESET-'),
        sg.Button('Exit', key='-EXIT-')
    ],

    [
        sg.Text(
            'Iteration:',
            font=('Any', 12)
        ),

        sg.Text(
            '0',
            key='-ITERATION-',
            size=(15, 1),
            font=('Any', 12)
        )
    ],

    [
        sg.Canvas(
            size=(500, 500),
            key='-CANVAS-'
        )
    ]
]


window = sg.Window(
    '2D Ising Model Simulation',
    layout,
    finalize=True
)

image, fig_agg = create_figure(
    window['-CANVAS-'].TKCanvas
)

stop_event = threading.Event()
worker = None

while True:

    event, values = window.read()

    if event in (sg.WIN_CLOSED, '-EXIT-'):
        stop_event.set()
        break

    if event == '-START-':

        # Read parameters from GUI
        try:
            T_red = float(values['-T-'])
            n = int(values['-N-'])
            down_probability = float(values['-DOWN-'])
            update_sleep = float(values['-SLEEP-'])

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

        except ValueError as e:

            sg.popup_error(
                "Invalid input",
                str(e)
            )

            continue

        N = n * n

        # Create new initial spin configuration
        spin_matrix = generate_initial_spin_orientations(
            n,
            down_probability
        )

        # Reset iteration counter
        iteration_count = 0

        window['-ITERATION-'].update('0')

        # Update image without recreating canvas
        image.set_data(spin_matrix)

        fig_agg.draw_idle()

    elif event == '-DATA-':

        new_matrix = values['-DATA-']

        # Update existing imshow
        image.set_data(new_matrix)

        # Update iteration counter
        window['-ITERATION-'].update(
            str(iteration_count)
        )

        # Redraw existing canvas
        fig_agg.draw_idle()

window.close()