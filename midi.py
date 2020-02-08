import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt
from oned import run, metrics
from scipy.stats import entropy
from scipy.optimize import differential_evolution as de
import time

# Create a pianoroll matrix, where the first and second axes represent time
# and pitch, respectively, and assign a C major chord to the pianoroll

steps = 96

beat_duration = 16

pianoroll = np.zeros((steps * beat_duration, 128))

# C notes chord
seed_01 = np.array([
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # octave 0
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, # octave 1
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # octave 2
])

# Random
seed_02 = np.array([
    0., 0., 0., 0.,  1.,  1.,  1.,
    0., 0., 0., 0.,  1.,  1., 1.,
    1.,  1.,  1.,  1.,  1., 0.,  1.
])

# Checkers
seed_03 = np.array([
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, # octave 0
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, # octave 1
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, # octave 2
])


# kernel
kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
# kernel = np.array([0.5, 0.5, 0.0, 0.5, 0.5])
seed = seed_01

C_maj_scale = np.array([
    36, 38, 40, 41, 43, 45, 47, # oct 0
    48, 50, 52, 53, 55, 57, 59, # oct 1
    60, 62, 64, 65, 67, 69, 71, # oct 2
])

# Kernel bounds
bounds = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0), (0, 0), (-4.0, 4.0), (-3.0, 3.0), (-2.0, 2.0), (-1.0, 1.0)]

# Evolve min entropy kernel
def f_min_entropy(k):
    states = run(steps, seed, k)
    run_metrics = metrics(states)
    score = run_metrics["entropy_score"] * run_metrics["num_states"]
    print("Entropy: ", score)
    return score


def f_max_entropy(k):
    states = run(steps, seed, k)
    run_metrics = metrics(states)
    score = -(run_metrics["entropy_score"] * run_metrics["num_states"])
    print("Entropy: ", score)
    return score


optimization_steps = 250

result_k = de(
    f_max_entropy, bounds, updating="deferred", workers=2, maxiter=optimization_steps
)

k_arr = result_k.x
# k_arr = kernel

print("kernel: ", k_arr)
print("seed: ", seed)

states = run(steps, seed, k_arr)

run_metrics = metrics(states)

print("Metrics: ", run_metrics)

for t in range(steps):
    state = states[t]
    beat = C_maj_scale * np.array(state)
    beat_list = [int(x) for x in beat.tolist()]
    print("{} -> {}".format(state, beat_list))
    beat_idx = t * beat_duration
    pianoroll[beat_idx, beat_list] = 100

# Clear 0s
pianoroll[0 : (steps * beat_duration), 0] = 0

# Create a `pypianoroll.Track` instance
track = Track(pianoroll=pianoroll, program=0, is_drum=False, name="my awesome piano")

mt = Multitrack(tracks=[track])

ts = int(time.time())

mt.write("./renderings/midi/test_{ts}.mid".format(ts=ts))

# # Plot the pianoroll
# fig, ax = track.plot()
# plt.show()
