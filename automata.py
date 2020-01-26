import numpy as np
from scipy.ndimage import convolve
from scipy.stats import entropy
from math import floor, ceil

# Default parameters as essentially GoL
DEFAULT_UNIVERSE_GRID_SIZE = (100, 100)
DEFAULT_NEIGHBORHOOD_GRID_SIZE = (3, 3)
DEFAULT_STEPS = 100
DEFAULT_THRESHOLDS = (1.9, 2.1, 2.9, 3.1)
ALIVE_T = 1.0
DEAD_T = 0.0

# NOTE: This could be fully vectorized
class AutomataOptions:
    def __init__(
        self,
        universe_grid_size=DEFAULT_UNIVERSE_GRID_SIZE,
        neighborhood_grid_size=DEFAULT_NEIGHBORHOOD_GRID_SIZE,
        thresholds_vector=DEFAULT_THRESHOLDS,
        alive_t=ALIVE_T,  # threshold of alive
        dead_t=DEAD_T,  # threshold of dead
    ):
        self.universe_size = universe_grid_size
        self.neighborhood_size = neighborhood_grid_size
        self.thresholds = thresholds_vector
        self.alive_t = alive_t
        self.dead_t = dead_t

    def __repr__(self):
        return """
            AutomataOptions:
                universe_size: {},
                neighborhood_grid_size: {},
                thresholds_vector: {},
                alive_t: {},
                dead_t: {}
        """.format(
            self.universe_size,
            self.neighborhood_size,
            self.thresholds,
            self.alive_t,
            self.dead_t,
        )


class Automata:
    def __init__(self, options=None):
        if not options:
            self.options = AutomataOptions()
        else:
            self.options = options

    def __repr__(self):
        return """
            Automata:
                AutomataOptions: {}
        """.format(
            self.options
        )

    def update_kernel(self):
        # Neighborhood size
        kernel_x = self.options.neighborhood_size[0]
        kernel_y = self.options.neighborhood_size[1]
        kernel_size = (kernel_x, kernel_y)

        self.kernel = np.full(kernel_size, self.ALIVE_T, dtype="float64")

        # Make center of kernel dead
        self.kernel[floor(kernel_y / 2), floor(kernel_x / 2)] = self.DEAD_T

    def create_universe(self):
        """
        Generate a universe using a normal, random distribution
        """
        size_grid_x = self.options.universe_size[0]
        size_grid_y = self.options.universe_size[1]
        size_tuple = (size_grid_x, size_grid_y)
        seed = np.random.rand(size_tuple[0], size_tuple[1])
        universe = seed.round()
        return np.copy(universe)  # return new instance

    def reset(self):
        self.scores = []
        self.states = []

        # Define constants for alive threshold and dead threshold
        self.ALIVE_T = self.options.alive_t
        self.DEAD_T = self.options.dead_t

        # Set kernel function based on parameters
        self.update_kernel()

        # Create universe
        self.universe = self.create_universe()

        # Set neighborhood thresholds
        self.neighbors_low = self.options.thresholds[0]
        self.neighbors_high = self.options.thresholds[1]
        self.neighbors_max_low = self.options.thresholds[2]
        self.neighbors_max_high = self.options.thresholds[3]

    def get_next_state(self) -> np.ndarray:
        universe = self.universe
        neighbors = convolve(universe, self.kernel, mode="constant")
        # This encodes the alive-ness as a logical operation using neighbors matrix
        alive = np.logical_or(
            np.logical_and(
                universe >= self.ALIVE_T,
                np.logical_and(
                    self.neighbors_low <= neighbors, neighbors < self.neighbors_high
                ),
            ),
            np.logical_and(
                self.neighbors_max_low <= neighbors, neighbors < self.neighbors_max_high
            ),
        )

        universe = np.where(alive, self.ALIVE_T, self.DEAD_T)
        return np.copy(universe)

    def run(self, steps=DEFAULT_STEPS) -> tuple:
        # Initialize all instance variables
        self.reset()

        # Run simulation for n steps
        for i in range(steps):
            # Save current state
            self.states.append(self.universe)
            # Evaluate current universe
            score = self.evaluate(self.universe)
            self.scores.append(score)
            # Generate next state
            self.universe = self.get_next_state()

        # Return a tuple of states, scores
        return self.states, self.scores

    def evaluate(self, state: np.ndarray) -> float:
        """
        Score the state using entropy function
        """
        return np.average(entropy(state, base=2))
