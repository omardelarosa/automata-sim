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
    uint8_tuple_to_bin_arr,
    bin_arr_to_s,
    int_to_activation_set,
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


def generate_states_from_rule_and_seed(
    f_name=None, rule=None, seed=None, scale_num=None, scale_type="maj"
):
    sc_num = scale_num
    sc_type = scale_type

    if scale_num != None:
        scale_mask = get_full_scale(sc_num)
        n = np.array(range(0, 128))
        scale = n[scale_mask]
        print("sc_num: {}, sc_type: {}".format(scale_num, scale_type))

    # Option 1. ECA rule seeds
    if scale_num != None:
        width = 75
    else:
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

    # Option 2. Start from a random given state
    random_state_idx = int(np.random.uniform(0, len(states)))

    rand_state = states[random_state_idx]
    rand_state_tiled = np.tile(rand_state, 128)[0:width]
    ## Option 3. Sample states from original
    # seeds = [rand_state_tiled]
    # if sc_num != None:
    #     maj_triad = np.tile(np.array([1, 0, 1, 0, 1, 0, 0]), reps=11)[
    #         0 : (len(states[0]))
    #     ]
    #     seeds.append(maj_triad)

    steps = 96
    i = 0
    for seed in seeds:
        k = rule["k"]
        a = np.array(rule["rule"])
        k_states = np.array(list(map(np.int64, rule["k_states"])))
        # generate rule from k_states / mask
        r_set = k_states[a.astype(bool)]
        r = lambda x, k: wolfram(x, k, r_set)
        states = run(steps, seed=seed, kernel=k, f=r)

        print_states(states[0:10])
        mets = metrics(states)
        f_name_out = f_name.replace(".", "_") + "_{}_".format(i)
        if sc_num != None:
            g = lambda x, y, z: generate_pianoroll(x, y, z, scale[0:width])
        else:
            g = lambda x, y, z: generate_pianoroll_chromatic(x, y, z, width)
        write_files_from_states(states, mets, seed, [], f_name_out, g=g)
        i += 1
    return


def learn_rule_from_file(
    f_name, track_num=None, scale_num=None, scale_type="maj", k_radius=1
):
    is_midi = f_name.endswith(".mid") or f_name.endswith(".midi")
    is_json = f_name.endswith(".json")

    sc_num = scale_num
    sc_type = scale_type

    if scale_num != None:
        scale_mask = get_full_scale(sc_num)
        n = np.array(range(0, 128))
        scale = n[scale_mask]
        print("sc_num: {}, sc_type: {}".format(scale_num, scale_type))

    if is_midi:
        states = convert_midi_to_state(f_name, scale_num, scale_type)
    elif is_json:
        # handle json
        with open(f_name, "r") as json_file:
            d = json.load(json_file)
        states = d["states"]
    else:
        print("File extension not supported!")
        exit(1)
    print_states(states[0:5])
    # return
    # mets = metrics(states)
    print("States read from file: ", f_name)
    # print(mets, learn_rules_from_states)

    rule = learn_rules_from_states(states, k_radius)

    write_rule_to_json(rule, f_name.replace(".", "_rule_"))

    return rule


def write_rule_to_json(rule, f_name):
    json_file = "{f_name}_.{ext}".format(f_name=f_name, ext="json")
    d = {}
    d["k"] = rule["k"]
    d["k_states"] = list(map(str, rule["k_states"]))
    d["rule"] = rule["rule"]
    d["confidence_scores"] = rule["confidence_scores"]
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
    json_file = "{f_name}_metrics_.{ext}".format(f_name=f_name, ext="json")
    json_states_file = "{f_name}_states_.{ext}".format(f_name=f_name, ext="json")
    # Write MIDI file
    mt.write(mid_file)

    # Write JSON file

    stats = {
        "metrics": metrics,
        "activation": activation,
        "kernel": list(kernel),
    }

    states = {"states": [s.astype(np.float32).tolist() for s in states]}
    # print("Stats:", stats)

    print("Writing results: {}".format(json_file))
    # Save state info as json_file
    with open(json_file, "w") as json_file:
        json.dump(stats, json_file)

    with open(json_states_file, "w") as json_states_file:
        json.dump(states, json_states_file)


def plot_track(pianoroll, title="my awesome piano"):
    # Create a `pypianoroll.Track` instance
    track = Track(pianoroll=pianoroll, program=0, is_drum=False, name=title)

    # Plot the pianoroll
    fig, ax = track.plot()
    plt.show()


def convert_midi_to_state(f_name, scale_num=None, scale_type="maj"):
    is_midi = f_name.endswith(".mid") or f_name.endswith(".midi")

    sc_num = scale_num
    sc_type = scale_type

    if scale_num != None:
        scale_mask = get_full_scale(sc_num)
        n = np.array(range(0, 128))
        scale = n[scale_mask]
        print("sc_num: {}, sc_type: {}".format(scale_num, scale_type))

    if is_midi:
        mt = load(f_name)

        # convert to binary representation
        mt.binarize()

        # ensure that the vector is 0,1 only
        track = mt.get_merged_pianoroll(mode="any").astype(int)

        # NOTE: these are the dimensions
        # rows = timestep, cols = keyboard
        # print(track.shape)
        states = []
        for s in track:
            # compress to scale
            if sc_num != None:
                s_compressed = squash_state_to_scale(s, scale_mask)
                states.append(s_compressed)
            else:
                states.append(s)
        states = squash_piano_roll_to_chromatic_frames(states)
        if sc_num != None:
            # Squash to scale
            states = [squash_state_to_scale(s, scale_mask[0:12]) for s in states]
        states = list(filter(lambda x: np.sum(x) > 0, states))
        deduped_states = []
        for i, state in enumerate(states):
            if i == 0:
                deduped_states.append(state)
            else:
                # filter out silence
                s = np.sum(state)
                if s > 0 and not np.all(np.equal(state, states[i - 1])):
                    deduped_states.append(state)
        states = deduped_states
    else:
        print("Not midi file!")
        exit(1)

    json_states_file = "{f_name}_states_.{ext}".format(
        f_name=f_name.replace(".", "_"), ext="json"
    )

    states_dict = {"states": [s.astype(int).tolist() for s in states]}

    with open(json_states_file, "w") as json_file:
        json.dump(states_dict, json_file)

    print("Wrote file: ", json_states_file)

    return states


def generate_all_wolfram_eca(f_dir, neighborhood_radius=1):
    # ECA is 1, anything greater is ???
    max_search_space_size = 5000

    # kernel is 3 consequtive primes
    k = tens(neighborhood_radius * 2 + 1)

    num_bits_kernel = 2 ** (len(k))
    print("k_size: ", num_bits_kernel)

    k_states = [uint8_tuple_to_bin_arr((i,)) for i in range(0, num_bits_kernel)]
    k_states_trimmed = list(
        map(lambda x: x[-int(log(num_bits_kernel, 2)) :], [x for x in k_states])
    )
    k_states_trimmed.reverse()  # sort by sums
    print("k_space_size: ", len(k_states))

    k_states = list(map(lambda x: np.dot(k, x), k_states_trimmed))
    # the maximum length of the rule, aka 2^(len(k))
    activations_search_space_size = 2 ** num_bits_kernel

    print("rule_space_size: ", activations_search_space_size)

    # Width of state to test
    width = 32
    # seed is initial state
    seed = np.zeros((width,), dtype=int)
    seed[floor(width / 2)] = 1  # add 1 to middle

    actual_search_space_size = min(activations_search_space_size, max_search_space_size)
    print("actual_search_space_size (with limit applied): ", actual_search_space_size)
    # Steps to draw sequence
    steps = 1000  # note this is not square
    # steps = 96
    activations_states_mets = []

    # skip states that do not meet filtering condition
    filtering_condition = lambda m, s: (
        m["entropy_score"] < 1.05 and m["num_states"] > 60
    )
    # filtering_condition = lambda m, s: True # no-op filtering

    # Iterate over all rules
    for n in range(0, actual_search_space_size):
        # generate activation
        a = int_to_activation_set(n, activations_search_space_size)
        # apply activation
        r_set = np.array(k_states)[a.astype(bool)]
        # NOTE: a, k must be np.arrays
        r = lambda x, k: wolfram(x, k, r_set)
        states = run(steps, seed=seed, kernel=k, f=r)
        mets = metrics(states)
        # TODO: expand this for more than 8bits
        # n = np.packbits(a)
        # if filtering_condition(mets, states):
        #     print("activation rule found: ", n, a)
        activations_states_mets.append((n, a, states, mets))

    # do something with results
    leading_zeroes = int(log(activations_search_space_size, 10)) + 1
    ts = int(time.time())
    f_dir = "{}/t_{}_".format(f_dir, ts)
    i = 0
    pianorolls = []
    for n, a, states, mets in activations_states_mets:
        # stringify bits
        filename = "_rule_{}".format(fmt_idx(n, leading_zeroes),)  # activation idx
        f_name = f_dir + filename
        write_files_from_eca(states, mets, k, n, f_name, title=f_name)

        # TODO: make it possible to alter parameters more easily
        scale = C_maj_scale_ext
        pianoroll = generate_pianoroll(states, scale=scale)
        pianorolls.append((pianoroll, "rule_{}".format(n)))

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
    "--learn",
    metavar="F",
    type=str,
    default=None,
    help="Learn rule from a midi or json states file.",
)

parser.add_argument(
    "--convert",
    metavar="C",
    type=str,
    default=None,
    help="Convert midi file to states file.",
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
    generate_all_wolfram_eca(f_dir, args.kernelRadius)
    exit(0)

if args.convert:
    convert_midi_to_state(args.convert, args.scaleNum, args.scaleType)
    exit(0)

if args.learn or args.load:
    learn_rule_from_file(
        args.load or args.learn,
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
