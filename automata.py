import numpy as np
import json
from scipy.ndimage import convolve
from scipy.stats import entropy
from math import floor, ceil, sqrt, log

# Default parameters as essentially GoL
DEFAULT_UNIVERSE_GRID_SIZE = (100, 100)
DEFAULT_NEIGHBORHOOD_GRID_SIZE = (3, 3)
DEFAULT_STEPS = 100
DEFAULT_THRESHOLDS = (-0.1, 0.1, -0.1, 0.1)
ALIVE_T = 0.99
DEAD_T = 0.0

# NOTE: This could be fully vectorized
class AutomataOptions:
    def __init__(
        self,
        steps=DEFAULT_STEPS,
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
        self.steps = steps

    def __repr__(self):
        return """
            AutomataOptions:
                steps: {},
                universe_size: {},
                neighborhood_grid_size: {},
                thresholds_vector: {},
                alive_t: {},
                dead_t: {}
        """.format(
            self.steps,
            self.universe_size,
            self.neighborhood_size,
            self.thresholds,
            self.alive_t,
            self.dead_t,
        )

    def to_dict(self):
        return {
            "universe_size": list(self.universe_size),
            "neighborhood_size": list(self.neighborhood_size),
            "thresholds": list(self.thresholds),
            "alive_t": self.alive_t,
            "dead_t": self.dead_t,
            "steps": self.steps,
        }


class Automata:
    def __init__(self, options=None, seed=[]):
        if not options:
            self.options = AutomataOptions()
        else:
            self.options = options
        self.seed = seed
        # Use this flag to label unscorable automata
        self.error_state = False

    def __repr__(self):
        return """
            Automata:
                {}
        """.format(
            self.options
        )

    def update_kernel(self):
        # Neighborhood size
        kernel_x = self.options.neighborhood_size[0]
        kernel_y = self.options.neighborhood_size[1]
        kernel_size = (kernel_x, kernel_y)

        # A fuzzy kernel function
        self.kernel = np.random.normal(
            self.options.alive_t, 0.001, size=(kernel_x, kernel_y)
        )

        # # Make center of kernel dead
        self.kernel[floor(kernel_y / 2), floor(kernel_x / 2)] = self.DEAD_T
        ##print("kernel: ", self.kernel)

    def get_seed(self):
        """
        Load a seed either from params or generate one at random
        """
        if self.seed:
            return self.seed
        else:
            # Random seed
            size_grid_x = self.options.universe_size[0]
            size_grid_y = self.options.universe_size[1]
            size_tuple = (size_grid_x, size_grid_y)
            seed = np.random.rand(size_tuple[0], size_tuple[1])
            return seed.round()

    def create_universe(self):
        """
        Generate a universe from seed
        """
        universe = self.get_seed()

        return np.copy(universe)  # return new instance

    def reset(self):
        self.scores = []
        self.states = []

        # Set number of steps
        self.steps = self.options.steps

        # Define constants for alive threshold and dead threshold
        self.ALIVE_T = self.options.alive_t
        self.DEAD_T = self.options.dead_t

        # Set kernel function based on parameters
        self.update_kernel()

        # Create universe
        self.universe = self.create_universe()

        # ceiling of 0.5 kernel width
        self.kernel_radius = ceil(self.options.neighborhood_size[0] * 0.5)

        # assumes square kernels, using X-dim
        self.kernel_span = self.options.neighborhood_size[0]

        # Set neighborhood thresholds
        self.neighbors_low = self.kernel_radius + self.options.thresholds[0]
        self.neighbors_high = self.kernel_radius + self.options.thresholds[1]
        self.neighbors_max_low = self.kernel_span + self.options.thresholds[2]
        self.neighbors_max_high = self.kernel_span + self.options.thresholds[3]

    def get_next_state(self) -> np.ndarray:
        universe = self.universe.round()
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

    def run(self, override_steps=None) -> tuple:
        # Initialize all instance variables
        self.reset()
        steps = self.steps
        # to run N-steps
        if override_steps:
            steps = override_steps

        # Run simulation for n steps
        for i in range(steps):
            # Save current state
            self.states.append(self.universe)
            # Evaluate current universe
            score = self.evaluate(self.universe)

            # Early abort
            if np.isnan(score):
                self.error_state = True
                self.steps_actual = i
                return self.states, self.scores

            self.scores.append(score)
            # Generate next state
            self.universe = self.get_next_state()

        self.steps_actual = steps
        # Return a tuple of states, scores
        return self.states, self.scores

    def evaluate(self, state: np.ndarray) -> float:
        """
        Score the state using entropy function
        """
        return np.average(entropy(state.flatten(), base=2))

    def stats(self):
        return {
            "score_max": max(self.scores),
            "score_min": min(self.scores),
            "score_sum": sum(self.scores),
            "score_delta": self.scores[-1] - self.scores[0],
            "score_avg": np.average(self.scores),
            "steps_actual": self.steps_actual,
        }

    def to_dict(self):
        return {
            "options": self.options.to_dict(),
            "scores": self.scores,
            "stats": self.stats(),
        }

    def save(self, filename):
        print("Saving to: ", filename)
        json_filename = filename + ".json"

        # Save states
        with open(json_filename, "w") as json_file:
            json.dump(self.to_dict(), json_file)

        # Save states
        i = 1
        zeros = int(log(self.steps, 10)) + 1
        for s in self.states:
            # padded filename
            f_name = filename + str(i).zfill(zeros)
            np.save(f_name, s)
            i = i + 1

    def load(self, files_array):
        """
        Load automata from directory
        """
        states = []
        scores = []
        for f in files_array:
            print("Loading: ", f)
            state = np.load(f)
            score = self.evaluate(state)
            scores.append(score)
            states.append(state)

        if len(scores) != len(states):
            print("Unable to load automata from files")
            exit(1)

        # This simulates a run
        self.seed = states[0]
        self.states = states
        self.scores = scores
        # Represents all scorable states
        self.steps_actual = len(scores)
