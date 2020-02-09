import numpy as np
from scipy.ndimage import convolve
from scipy.stats import entropy
from math import floor, ceil, sqrt, log
from automata import Automata, AutomataOptions

seed = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])

print("seed: {}, kernel: {}".format(seed, kernel))
a_b = convolve(seed, kernel)

results = []
# print("results: ")


def encode_state(s):
    """
    Encode as a tuple
    """
    bool_arr = np.array(s, dtype=np.bool)
    x = np.packbits(bool_arr)
    x_t = tuple(x)
    # x_s = str(x_t)
    # Tuple and string representation
    return x_t


def decode_state(s_t):
    """
    Decode from tuple
    """
    x_t = np.unpackbits(np.array(s_t, dtype=np.uint8))
    return x_t


def metrics(results_arr):
    # size = len(results_arr)
    states_set = set()
    states_counts = {}
    states_distribution = {}
    for s in results_arr:
        s_t = encode_state(s)  # create tuple
        s_hash = str(s_t)
        states_set.add(s_t)
        if s_t in states_counts:
            states_counts[s_hash] = states_counts[s_hash] + 1
        else:
            states_counts[s_hash] = 1

    # Num states
    num_states = len(states_set)
    for key in states_counts:
        states_distribution[key] = states_counts[key] / num_states

    # Calculate entropy
    probs_of_state = []

    for s in results_arr:
        s_t = encode_state(s)  # create tuple
        s_hash = str(s_t)
        prob_of_state = states_distribution[s_hash]
        probs_of_state.append(prob_of_state)

    # print("Probs", probs_of_state)
    ent_score = entropy(probs_of_state, base=num_states)

    return {
        "states_counts": states_counts,
        "states_distribution": states_distribution,
        "entropy_score": ent_score,
        "num_states": num_states,
    }


# # Normalize
def f(x, k):
    x_next = convolve(x, k, mode="reflect")
    x_norm = np.linalg.norm(x_next)
    if x_norm == 0:
        return x
    return np.abs(np.round(x_next / x_norm))


def gol(x, k):
    neighbors = convolve(x, k, mode="constant")
    alive = np.logical_or(
        np.logical_and(x == 0.0, np.logical_and(neighbors > 3),),
        np.logical_and(x >= 1.0, np.logical_and(neighbors >= 2, neighbors <= 3)),
    )
    x = np.where(alive, 1.0, 0.0)
    return x


def run(steps, seed=seed, kernel=kernel, f=f):
    results = [seed]
    a_b = seed
    for i in range(steps):
        a_b = f(a_b, kernel)
        r = a_b.copy()
        # print("{}: {}".format(i, r))
        results.append(r)
    return results


# class OneD(Automata):


# Example when run as main
if __name__ == "__main__":
    # TODO: do something with results
    run(10)
