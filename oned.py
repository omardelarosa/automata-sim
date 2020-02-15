import numpy as np
from scipy.ndimage import convolve
from scipy.stats import entropy
from math import floor, ceil, sqrt, log
from automata import Automata, AutomataOptions

# For generating primes
import sympy
from itertools import combinations_with_replacement
from functools import reduce

seed = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
primes_kernel = np.array([2, 3, 5])

# seed = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

seed = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0, 0, 0])
ones = np.full((0, len(seed)), 1).squeeze()
zeros = np.full((0, len(seed)), 0).squeeze()

# Assuming k = [2,3,5]


# seed = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
# kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
# kernel = np.array([0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5])
# kernel = np.array([-0.5, -1, -1, 0, 1, 1, 0.5])
# kernel = np.array([1, 1, 1, 0, 1, 1, 1])

print("seed: {}, kernel: {}".format(seed, kernel))
a_b = convolve(seed, kernel)

results = []
# print("results: ")


def primes(n):
    """
    Return list of first n consequtive primes
    """
    ns = [2]
    if n == 1:
        return ns
    for i in range(n - 1):
        ns.append(sympy.nextprime(ns[-1]))
    return ns


def generate_combined_products(primes_list):
    l_primes = len(primes_list)
    c = list(combinations_with_replacement(primes_list, l_primes))
    # dedupe
    cc = list(map(lambda x: np.unique(x).tolist(), c))
    ccc = np.unique(cc).tolist()
    products = list(map(lambda x: reduce(lambda a, b: a * b, list(x)), ccc))
    return products


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
    x_next = convolve(x, k, mode="constant")
    # Activation Function 1: - binary
    # res = np.array(x_next * (x_next > 0) > 0, dtype=int)

    # Activation Function 2: based on population
    res = np.array(x_next * (x_next > 1) * (x_next < 3), dtype=bool).astype(int)

    # print("res: ", res)
    return res


# def gol(x, k):
#     neighbors = convolve(x, k, mode="constant")
#     alive = np.logical_or(
#         np.logical_and(x == 0.0, np.logical_and(neighbors > 1),),
#         np.logical_and(x >= 1.0, np.logical_and(neighbors >= 0, neighbors <= 1)),
#     )
#     x = np.where(alive, 1.0, 0.0)
#     return x


def rule_30():
    # States with 0 added

    # kernel is 3 consequtive primes
    k = primes(3)

    # k_states is all the combinations of products of k
    k_states = np.array(generate_combined_products(k) + [0])

    # seed is initial state
    seed = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    # print("K_states", k_states)
    # a is the "rule" bit array
    a = np.array([True, False, False, False, True, True, True, False])

    # this is the callback
    def r30(x, k):
        return wolfram(x, k, a, k_states)

    # TODO: do something with results
    states = run(10, seed=seed, kernel=primes_kernel, f=r30)

    mets = metrics(states)

    print("Metrics: ", mets)
    # # Pretty print for LaTex
    for s in states:
        print(" & ".join(map(str, s.tolist())) + "\\")


def wolfram(x, k, a, k_states=None):
    """
    k: kernel operator
    a: activation
    k_states: kernel combination space
    """
    x_next = convolve(x, k, mode="constant")

    states_arr = k_states[a]

    matches = np.isin(x_next, states_arr)
    x = np.where(matches, 1, 0)
    return x


def run(steps, seed=seed, kernel=kernel, f=f):
    results = [seed]
    a_b = seed
    for i in range(steps):
        a_b = f(a_b, kernel)
        r = a_b.copy()
        print("{}: {}".format(i, r))
        results.append(r)
    return results


# class OneD(Automata):


# Example when run as main
if __name__ == "__main__":
    rule_30()

