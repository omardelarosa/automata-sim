import numpy as np
from automata import Automata, AutomataOptions
from scipy.optimize import differential_evolution as de
import ctypes as c
from plotter import Plotter


class Optimizer:
    def __init__(self):
        self.universe_grid_size = (100, 100)
        self.neighborhood_grid_size = (5, 5)
        self.steps = 100
        self.DEAD_T = 0.0
        self.best_automata = None
        self.best_fitness = np.float64("inf")
        self.t = 0

    def vec_to_opts(self, vec):
        thresholds = tuple(vec[0:4])
        alive_t = vec[5]
        opts = AutomataOptions(
            self.steps,
            self.universe_grid_size,
            self.neighborhood_grid_size,
            thresholds,
            alive_t,
            self.DEAD_T,
        )
        return opts

    def objective_func(self, x):
        opts = self.vec_to_opts(x)
        print(opts)
        # run automata
        a = Automata(opts)
        a.run()
        stats = a.stats()
        fitness = np.float64(-stats["score_sum"])
        print("Fitness: ", fitness)
        # if fitness < self.best_fitness:
        #     self.best_fitness = fitness
        #     self.best_automata = a
        return fitness

    def results_callback(self, vec, dimension):

        # # Store results to python memory containers
        # # Store population
        # for i in range(0, population_size * problem_size):
        #     row = i // problem_size
        #     col = i % problem_size
        #     self.out_population[row][col] = np.float64(population[i])

        # # Store fitness values
        # for j in range(0, population_size):
        #     f = fitness_values[j]
        #     self.out_fitnesses[j] = np.float64(f)
        return


opt = Optimizer()

population_size = 30
optimization_steps = 5000
scaling_factor = 0.01
cross_over_rate = 0.5

problem_size = 4
# x = [-0.1, 0.1, -0.1, 1.0]
# y = [1000.0, 1000.0, 1000.0, 1000.0]
# x_c = x.ctypes.data_as(c.POINTER(c.c_double))
# y_c = x.ctypes.data_as(c.POINTER(c.c_double))

bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.7, 0.99)]
result = de(
    opt.objective_func,
    bounds,
    updating="deferred",
    workers=2,
    maxiter=optimization_steps,
)

x = result.x
opts = opt.vec_to_opts(x)
a = Automata(opts)
a.run()
print("BestAutomata", a)
print(a.stats())
p = Plotter(a)
p.plot(True)  # animate
