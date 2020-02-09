import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt
from oned import run, metrics, gol
from math import log
from scipy.stats import entropy
from scipy.optimize import differential_evolution as de
import json
import time

# Create a pianoroll matrix, where the first and second axes represent time
# and pitch, respectively, and assign a C major chord to the pianoroll

steps = 96

beat_duration = 16


# C notes chord
seed_01 = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # octave 0
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,  # octave 1
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # octave 2
    ]
)

# Random
seed_02 = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
    ]
)

# Checkers
seed_03 = np.array(
    [
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,  # octave 0
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,  # octave 1
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,  # octave 2
    ]
)


# kernel
kernel_01 = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
kernel_02 = np.array([0.5, 0.5, 0.0, 0.5, 0.5])

C_maj_scale = np.array(
    [
        36,
        38,
        40,
        41,
        43,
        45,
        47,  # oct 0
        48,
        50,
        52,
        53,
        55,
        57,
        59,  # oct 1
        60,
        62,
        64,
        65,
        67,
        69,
        71,  # oct 2
    ]
)

# Kernel bounds
bounds = [
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
]
# bounds = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0), (0, 0), (-4.0, 4.0), (-3.0, 3.0), (-2.0, 2.0), (-1.0, 1.0)]


def generate_pianoroll(states, steps=steps, beat_duration=beat_duration):
    pianoroll = np.zeros((steps * beat_duration, 128))

    for t in range(steps):
        state = states[t]
        beat = C_maj_scale * np.array(state)
        beat_list = [int(x) for x in beat.tolist()]
        # print("{} -> {}".format(state, beat_list))
        beat_idx = t * beat_duration
        pianoroll[beat_idx, beat_list] = 100

    # Clear 0s
    pianoroll[0 : (steps * beat_duration), 0] = 0

    return pianoroll


def write_files_from_states(
    states,
    metrics,
    seed,
    kernel,
    f_name="./renderings/midi/t",
    title="my awesome piano",
):
    # TODO: make it possible to alter parameters more easily
    pianoroll = generate_pianoroll(states)

    # Create a `pypianoroll.Track` instance
    track = Track(pianoroll=pianoroll, program=0, is_drum=False, name=title)

    mt = Multitrack(tracks=[track])

    mid_file = "{f_name}_.{ext}".format(f_name=f_name, ext="mid")
    json_file = "{f_name}_.{ext}".format(f_name=f_name, ext="json")

    # Write MIDI file
    mt.write(mid_file)

    # Write JSON file

    stats = {
        "metrics": metrics,
        "states": [list(s) for s in states],
        "seed": list(seed),
        "kernel": list(kernel),
    }

    print("Writing results: {}".format(json_file))
    # Save state info as json_file
    with open(json_file, "w") as json_file:
        json.dump(stats, json_file)


def plot_track(pianoroll, title="my awesome piano"):
    # Create a `pypianoroll.Track` instance
    track = Track(pianoroll=pianoroll, program=0, is_drum=False, name=title)

    # Plot the pianoroll
    fig, ax = track.plot()
    plt.show()


def optimize_kernel_using_de(steps, seed):

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

    def evolve_kernel(optimization_steps=3000, bounds=bounds, f=f_max_entropy):
        result_k = de(
            f, bounds, updating="deferred", workers=2, maxiter=optimization_steps
        )

        return result_k

    # Do stuff
    result_k = evolve_kernel()
    k_arr = result_k.x

    print("kernel: ", k_arr)
    print("seed: ", seed)

    states = run(steps, seed, k_arr)

    run_metrics = metrics(states)

    print("Metrics: ", run_metrics)


def fmt_idx(n, leading_zeroes=2):
    return str(n).zfill(leading_zeroes)


def run_and_save(steps, seeds, kernels, f_dir="./renderings/midi"):
    leading_zeroes = int(log(max(len(seeds), len(kernels)))) + 1
    ts = int(time.time())
    f_dir = "{}/t_{}_".format(f_dir, ts)
    for s_idx, seed in enumerate(seeds):
        for k_idx, kernel in enumerate(kernels):
            filename = "_s{}_k{}".format(
                fmt_idx(s_idx, leading_zeroes),  # seed idx
                fmt_idx(k_idx, leading_zeroes),  # kernel idx
            )
            f_name = f_dir + filename
            states = run(steps, seed, kernel)
            run_metrics = metrics(states)
            write_files_from_states(
                states, run_metrics, seed, kernel, f_name, title=f_name
            )


def kernel_space_3_1(span=1.0):
    """
    Create a space of 1x3 kernels (-1,1) in 0.1 steps
    """
    step = 0.2
    rng = np.arange(
        -1.0 * span, 1.0 * span, step  # lower bound  # higher bound  # step size
    )

    kernels = []

    for x in rng:
        for y in rng:
            for z in rng:
                kernel = [x, y, z]
                kernels.append(kernel)

    return kernels


kernel_space = kernel_space_3_1()
# print(kernel_space)
# Example when run as main
if __name__ == "__main__":
    """
    Run for kernel space 3x1 using 3 sample seeds
    """
    f_dir = "./renderings/midi/kernel_space_3x1"
    run_and_save(96, [seed_01, seed_02, seed_03], kernel_space, f_dir)

