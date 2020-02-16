import numpy as np
from pypianoroll import Multitrack, Track, load
from matplotlib import pyplot as plt
from oned import (
    run,
    metrics,
    gol,
    primes,
    generate_combined_products,
    wolfram,
    learn_rules_from_states,
    tens,
    generate_states_from_learned_rule,
    print_states,
)
from math import log, floor
from aggregate import aggregate_summary, plot_summary
from scipy.stats import entropy
from scipy.optimize import differential_evolution as de
from itertools import product
from constants import (
    C_maj_scale,
    C_maj_scale_ext,
    seed_01,
    seed_02,
    seed_03,
    MINOR_SCALE_MASK,
    MAJOR_SCALE_MASK,
)
import json
import time
import argparse
import os

# Create a pianoroll matrix, where the first and second axes represent time
# and pitch, respectively, and assign a C major chord to the pianoroll

steps = 96

beat_duration = 16


# kernel
kernel_01 = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
kernel_02 = np.array([0.5, 0.5, 0.0, 0.5, 0.5])


# bounds = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0), (0, 0), (-4.0, 4.0), (-3.0, 3.0), (-2.0, 2.0), (-1.0, 1.0)]


def get_full_scale(n=0, major=True, num_octaves=11):
    if not major:
        mask = MINOR_SCALE_MASK
    else:
        mask = MAJOR_SCALE_MASK

    shifted = np.roll(mask, n)
    return np.tile(shifted, num_octaves)[0:128]


def generate_pianoroll(
    states, steps=steps, beat_duration=beat_duration, scale=C_maj_scale
):
    pianoroll = np.zeros((steps * beat_duration, 128))

    for t in range(steps):
        state = states[t]
        beat = scale * np.array(state)
        beat_list = [int(x) for x in beat.tolist()]
        # print("{} -> {}".format(state, beat_list))
        beat_idx = t * beat_duration
        pianoroll[beat_idx, beat_list] = 100

    # Clear 0s
    pianoroll[0 : (steps * beat_duration), 0] = 0

    return pianoroll


def generate_pianoroll_chromatic(
    states, steps=steps, beat_duration=beat_duration, width=128
):
    pianoroll = np.zeros((steps * beat_duration, width))

    scale = np.array(range(0, width))

    for t in range(steps):
        state = states[t]
        beat = scale * np.array(state)
        beat_list = [int(x) for x in beat.tolist()]
        # print("{} -> {}".format(state, beat_list))
        beat_idx = t * beat_duration
        pianoroll[beat_idx, beat_list] = 100

    # Clear 0s
    pianoroll[0 : (steps * beat_duration), 0] = 0

    return pianoroll


def squash_state_to_scale(state, sc_mask):
    s_compressed = state[sc_mask]
    return s_compressed


def squash_piano_roll_to_chromatic_frames(states):
    width = len(states[0])
    state_slices = floor(width / 12)
    slices = []
    states_arr = np.array(states)
    for i in range(state_slices):
        l = i * 12
        r = l + 12
        slices.append(states_arr[:, l:r])
    s_compressed = np.concatenate(slices)
    return s_compressed


def get_states_from_file(
    f_name, track_num=None, scale_num=None, scale_type="maj", k_radius=1
):
    mt = load(f_name)

    # convert to binary representation
    mt.binarize()

    # ensure that the vector is 0,1 only
    track = mt.get_merged_pianoroll(mode="any").astype(int)

    if track_num:
        track = mt.tracks[track_num]

    sc_num = scale_num
    sc_type = scale_type

    if scale_num != None:
        scale_mask = get_full_scale(sc_num)
        n = np.array(range(0, 128))
        scale = n[scale_mask]
        print("sc_num: {}, sc_type: {}".format(scale_num, scale_type))

    # NOTE: these are the dimensions
    # rows = timestep, cols = keyboard
    # print(track.shape)
    states = []
    for s in track:
        # Skip rests by only taking tracks with non-zero sums
        sum_s = np.sum(s)
        if sum_s > 0:
            # compress to scale
            if sc_num != None:
                s_compressed = squash_state_to_scale(s, scale_mask)
                states.append(s_compressed)
            else:
                states.append(s)
    states = squash_piano_roll_to_chromatic_frames(states)
    print_states(states[0:5])
    # return
    # mets = metrics(states)
    print("States read from file: ", f_name)
    # print(mets, learn_rules_from_states)

    rule = learn_rules_from_states(states, k_radius)
    print(rule)
    write_rule_to_json(rule, f_name.replace(".", "_rule_"))

    # Option 1. ECA rule seeds
    width = 128
    # width = len(states[0])
    # seed is initial state
    seed = np.zeros((width,), dtype=int)
    seed[floor(width / 2)] = 1  # add 1 to middle

    # Option 0. Random seed
    # seed = np.random.rand(width).round()
    print("seed:", seed)
    # Option 1. Start from a single 1 value, ala ECA
    seeds = [seed]

    ## Option 2. Sample states from original
    # seeds = [states[0:100][3], states[0:100][20], states[0:100][40]]
    # if sc_num != None:
    #     maj_triad = np.tile(np.array([1, 0, 1, 0, 1, 0, 0]), reps=11)[
    #         0 : (len(states[0]))
    #     ]
    #     seeds.append(maj_triad)

    i = 0
    for seed in seeds:
        states = generate_states_from_learned_rule(96, np.array(seed), rule)
        # print_states(states)
        mets = metrics(states)
        # print(mets)
        f_name_out = f_name.replace(".", "_") + "_{}_".format(i)
        if sc_num != None:
            g = lambda x, y, z: generate_pianoroll(x, y, z, scale[0:width])
        else:
            g = lambda x, y, z: generate_pianoroll_chromatic(x, y, z, width)
        write_files_from_states(states, mets, seed, [], f_name_out, g=g)
        i += 1


def write_rule_to_json(rule, f_name):
    json_file = "{f_name}_.{ext}".format(f_name=f_name, ext="json")
    d = {}
    d["k"] = list(map(str, rule["k"]))
    d["rule"] = {}
    for key in rule["rule"]:
        d["rule"][str(key)] = rule["rule"][key]
    print(d)
    print("Writing Rule to: {}".format(json_file))
    # Save state info as json_file
    with open(json_file, "w") as json_file:
        json.dump(d, json_file)


def write_files_from_states(
    states,
    metrics,
    seed,
    kernel,
    f_name="./renderings/midi/t",
    title="my awesome piano",
    g=generate_pianoroll,
):
    # TODO: make it possible to alter parameters more easily
    pianoroll = g(states, steps, beat_duration)

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
        # "states": [list(s) for s in states],
        # "seed": list(seed),
        # "kernel": list(kernel),
    }

    print("Writing results: {}".format(json_file))
    # Save state info as json_file
    with open(json_file, "w") as json_file:
        json.dump(stats, json_file)


def write_files_from_eca(
    states,
    metrics,
    kernel,
    activation,
    f_name="./renderings/midi/t",
    title="eca piano",
):
    # TODO: make it possible to alter parameters more easily
    pianoroll = generate_pianoroll(states, scale=C_maj_scale_ext)

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
        "states": [s.astype(np.float32).tolist() for s in states],
        "activation": list(activation),
        "kernel": list(kernel),
    }
    # print("Stats:", stats)

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


def generate_all_wolfram_eca(f_dir):

    # kernel is 3 consequtive primes
    k = tens(3)

    # k_states is all the combinations of products of k
    k_states = np.array(generate_combined_products(k) + [0])

    width = 32
    # seed is initial state
    seed = np.zeros((width,), dtype=int)
    seed[floor(width / 2)] = 1  # add 1 to middle

    # generate cartesian product of all bit flags
    bits = [False, True]

    # the maximum length of the rule, aka 2^(len(k))
    rule_size = 8

    activations_arr = list(product(bits, repeat=rule_size))

    # convert to numpy
    activations = [np.array(a) for a in activations_arr]

    # print(activations)

    steps = 96  # note this is not square

    # a = np.array([True, False, False, False, True, True, True, False])

    activations_states_mets = []
    # Iterate over all rules
    for a in activations:
        r = lambda x, k: wolfram(x, k, a, k_states)
        states = run(steps, seed=seed, kernel=k, f=r)
        mets = metrics(states)
        # TODO: expand this for more than 8bits
        n = np.packbits(a)
        activations_states_mets.append((n, states, mets))

    # do something with results
    leading_zeroes = int(log(len(activations_states_mets), 3)) + 1
    ts = int(time.time())
    f_dir = "{}/t_{}_".format(f_dir, ts)
    i = 0
    pianorolls = []
    for n, states, mets in activations_states_mets:
        # stringify bits
        a = ",".join(list(map(str, list(n))))
        filename = "_i{}_a{}".format(
            fmt_idx(i, leading_zeroes),  # seed idx
            fmt_idx(a, leading_zeroes),  # kernel idx
        )
        f_name = f_dir + filename
        # states = run(steps, seed, k, f=f)
        # run_metrics = metrics(states)
        write_files_from_eca(states, mets, k, a, f_name, title=f_name)
        i += 1

        # TODO: make it possible to alter parameters more easily
        pianoroll = generate_pianoroll(states, scale=C_maj_scale_ext)
        pianorolls.append((pianoroll, "{}_{}".format(i, a)))

    tracks = []
    for pianoroll, title in pianorolls:
        # Create a `pypianoroll.Track` instance
        track = Track(pianoroll=pianoroll, program=0, is_drum=False, name=title)
        tracks.append(track)

    # Summary midi file
    mt = Multitrack(tracks=tracks)
    mid_file = "{f_dir}__summary.{ext}".format(f_dir=f_dir, ext="mid")
    mt.write(mid_file)

    # plot_filename = "{f_dir}__summary.{ext}".format(f_dir=f_dir, ext="pdf")
    # print("Generating plot: ", plot_filename)
    # mt.plot()

    # Save plot as PDF
    # plt.savefig(plot_filename, dpi=150)

    # plt.show()

    # Aggregate results
    aggregate_summary(f_dir)

    # if plot:
    #     # Plot aggregation
    #     plot_summary(f_dir + "_summary.json")


def evolve_kernel_using_de(steps, seed, mode="min", optimization_steps=None):
    # Activation function
    g = gol

    # Evolve min entropy kernel
    def f_min_entropy(k):
        states = run(steps, seed, k, f=g)
        run_metrics = metrics(states)
        score = run_metrics["entropy_score"] * run_metrics["num_states"]
        # print("Entropy: ", score)
        return score

    def f_max_entropy(k):
        states = run(steps, seed, k, f=g)
        run_metrics = metrics(states)
        score = -(run_metrics["entropy_score"] * run_metrics["num_states"])
        # print("Entropy: ", score)
        return score

    def f_max_diversity(k):
        states = run(steps, seed, k, f=g)
        run_metrics = metrics(states)
        score = -run_metrics["num_states"]
        # print("Num States: ", score)
        return score

    def f_min_diversity(k):
        states = run(steps, seed, k, f=g)
        run_metrics = metrics(states)
        score = run_metrics["num_states"]
        # print("Num States: ", score)
        return score

    f = f_max_entropy
    if mode == "min_entropy":
        f = f_min_entropy
    elif mode == "max_entropy":
        f = f_max_entropy
    elif mode == "min_diversity":
        f = f_min_diversity
    elif mode == "max_diversity":
        f = f_max_diversity
    else:
        print("Unsupported mode: {}".format(mode))
        exit(1)

    def evolve_kernel(optimization_steps=3000, bounds=bounds, f=f_max_entropy):
        result_k = de(f, bounds, updating="deferred", maxiter=optimization_steps)

        return result_k

    # Do stuff
    result_k = evolve_kernel(f=f, optimization_steps=optimization_steps)
    k_arr = result_k.x

    # print("kernel: ", k_arr)
    # print("seed: ", seed)

    states = run(steps, seed, k_arr)

    run_metrics = metrics(states)

    print("Metrics: ", run_metrics)

    return k_arr


def fmt_idx(n, leading_zeroes=2):
    return str(n).zfill(leading_zeroes)


def run_and_save(steps, seeds, kernels, f_dir="./renderings/midi", plot=True):
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

    # Aggregate results
    aggregate_summary(f_dir)

    if plot:
        # Plot aggregation
        plot_summary(f_dir + "_summary.json")


def kernel_space_3_1(span=1.0):
    """
    Create a space of 1x3 kernels (-1,1) in 0.1 steps
    """
    step = 0.1
    rng = np.arange(
        -1.0 * span, 1.0 * span, step  # lower bound  # higher bound  # step size
    )

    kernels = []

    CENTER = 0.0  # unweighted center value

    for x in rng:
        for z in rng:
            kernel = [x, CENTER, z]
            kernels.append(kernel)

    return kernels


# print(kernel_space)
# Example when run as main
# if __name__ == "__main__":
#     """
#     Run for kernel space 3x1 using 3 sample seeds
#     """
#     f_dir = "./renderings/midi/kernel_space_3x1"
#     run_and_save(96, [seed_01, seed_02, seed_03], kernel_space, f_dir)

DEFAULT_OUTDIR = "./renderings/midi/kernel_space_3x1"

parser = argparse.ArgumentParser(description="Run a 1-D cellular automata simulation.")

MODES = ["optimizer", "kernel_space", "plot", "wolfram"]

# Args
parser.add_argument(
    "--outdir",
    metavar="O",
    type=str,
    default=DEFAULT_OUTDIR,
    help="Output Directory: (default: {})".format(DEFAULT_OUTDIR),
)

parser.add_argument(
    "--plotSummary",
    metavar="P",
    type=str,
    default=DEFAULT_OUTDIR,
    help="Plot summary json: (default: {})".format(DEFAULT_OUTDIR),
)

parser.add_argument(
    "--mode",
    metavar="M",
    type=str,
    default=None,
    help="Optimizer or kernel space mode: (options: {})".format(MODES),
)

parser.add_argument(
    "--load", metavar="F", type=str, default=None, help="Load states from a midi file."
)

parser.add_argument(
    "--kernelRadius",
    metavar="R",
    type=int,
    default=1,
    help="The radius of the kernel around the cell.",
)

parser.add_argument(
    "--track",
    metavar="T",
    type=int,
    default=0,
    help="Select which track to read from in a midifile.",
)

parser.add_argument(
    "--scaleNum", metavar="SN", type=int, default=None, help="Select scale 0-12",
)

parser.add_argument(
    "--scaleType",
    metavar="ST",
    type=str,
    default="maj",
    help="Select scale type: [maj, min]",
)

args = parser.parse_args()

f_dir = DEFAULT_OUTDIR

if args.outdir:
    f_dir = args.outdir

if args.mode == MODES[3]:
    generate_all_wolfram_eca(f_dir)
    exit(0)

if args.load:
    get_states_from_file(
        args.load,
        track_num=args.track,
        scale_num=args.scaleNum,
        scale_type=args.scaleType,
        k_radius=args.kernelRadius,
    )
    exit(0)

# Mutually exclusive modes
if args.mode == MODES[1]:
    kernel_space = kernel_space_3_1()
    run_and_save(96, [seed_01, seed_02, seed_03], kernel_space, f_dir)
elif args.mode == MODES[0]:
    print("f_dir", f_dir)
    steps = 96
    optimization_steps = 1000
    seeds = [seed_01, seed_02, seed_03]
    kernels = []

    evolution_fitness_modes = [
        "min_entropy",
        "max_entropy",
        "min_diversity",
        "max_diversity",
    ]

    for m in evolution_fitness_modes:
        kernels = []
        print("Evolving mode: {}".format(m))
        f_dir_k = f_dir + "/{}".format(m)
        try:
            os.mkdir(f_dir_k)
        except FileExistsError:
            print("warning: directory already exists: {}".format(f_dir_k))

        for seed in seeds:
            k = evolve_kernel_using_de(steps, seed, m, optimization_steps)
            kernels.append(k)
        run_and_save(96, seeds, kernels, f_dir_k, plot=False)
elif args.mode == MODES[2] and args.plotSummary:
    # Plot aggregation
    plot_summary(args.plotSummary)
