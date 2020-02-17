import numpy as np
from scipy.ndimage import convolve
from scipy.stats import entropy
from math import floor, ceil, sqrt, log
from automata import Automata, AutomataOptions
from bitarray import bitarray
from PIL import Image

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

# print("seed: {}, kernel: {}".format(seed, kernel))
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


def tens(n):
    ns = [1]
    if n == 1:
        return ns
    for i in range(1, n):
        ns.append(10 ** i)
    return ns


def generate_combined_products(primes_list):
    l_primes = len(primes_list)
    c = list(combinations_with_replacement(primes_list, l_primes))
    # dedupe
    cc = list(map(lambda x: np.unique(x).tolist(), c))
    ccc = np.unique(cc).tolist()
    products = list(map(lambda x: reduce(lambda a, b: a + b, list(x)), ccc))
    return products


def learn_rules_from_states(states, kernel_radius=1, debug=False):
    # generate a kernel based on radius
    k_len = (kernel_radius * 2) + 1
    # kernel = primes(k_len)
    k = tens(k_len)

    num_bits_kernel = 2 ** (len(k))

    k_states = [uint8_tuple_to_bin_arr((i,)) for i in range(0, num_bits_kernel)]
    k_states_trimmed = list(
        map(lambda x: x[-int(log(num_bits_kernel, 2)) :], [x for x in k_states])
    )
    k_states_trimmed.reverse()  # sort by sums
    if debug:
        print("k_space_size: ", len(k_states))

    k_states = list(map(lambda x: np.dot(k, x), k_states_trimmed))

    # the maximum length of the rule, aka 2^(len(k))
    activations_search_space_size = 2 ** num_bits_kernel

    if debug:
        print("rule_space_size: ", activations_search_space_size)

    # print("k_states: ", k_states)
    # only track non-zero
    counts_dict = {}
    # for key in k_states:
    # count number of next states, 0 = Falses, 1 = Trues
    # counts_dict[key] = [0, 0]

    # 1. iterate over all states learning transitions
    n = len(states)
    for i in range(n - 1):
        x = states[i]
        x_plus_1 = states[i + 1]
        # Apply kernel to x
        x_pattern = convolve(x, k, mode="constant")
        # compare patterns to next state value
        for j in range(len(x)):
            x_patt_i = x_pattern[j]  # pattern encoding
            # print("x_patt_i", x_patt_i)
            rule_str = str(int(x_patt_i))
            x_plus_1_j = int(x_plus_1[j])  # next transition
            # print("r({}) {} -> {}: ".format(rule_str, x_patt_i, x_plus_1_j))
            # return
            if rule_str in counts_dict:
                counts_dict[rule_str][x_plus_1_j] = (
                    counts_dict[rule_str][x_plus_1_j] + 1
                )
            else:
                counts = [0, 0]
                counts[x_plus_1_j] = 1
                counts_dict[rule_str] = counts

        # Find match for x-pattern

    # create a dictionary of likelihood value will be 1
    rule = {}
    targets = {}
    state_size = len(states[0])
    population = np.sum(np.array(states))  # i.e. number of alive cells
    occurences = n * len(states[0])
    if debug:
        print(counts_dict)

    # the minimum probability to mark rule
    prob_floor = 0.0000
    for n in counts_dict:
        v = counts_dict[n]
        # rule[n] = np.float64(v[1]) / np.sum(v, dtype=np.float64)
        prob_1 = np.float64(v[1]) / population
        prob_0 = np.float64(v[0]) / (occurences - population)
        target = 0
        if prob_1 > prob_0:
            prob = prob_1
            target = 1
        else:
            prob = prob_0
            target = 0
        # assign the prob
        if prob > prob_floor:
            rule[n] = prob
            targets[n] = target
    # Match with rule in rulespace
    a = []
    for ks in k_states:
        rule_str = str(ks)
        if rule_str in rule:
            t = targets[rule_str]
            a.append(t)
        else:
            a.append(0)
    if debug:
        print("a:", a)
    return {"k": k, "rule": a, "k_states": k_states, "confidence_scores": rule}


def generate_states_from_learned_rule(steps, seed, rule):
    """
    learned_rule:  { k, rules: counts_dict }
    """
    states = [seed]
    for i in range(1, steps):
        state = state_from_rule(states[-1], rule)
        states.append(state)
    return states


def state_from_rule(x, learned_rule):
    """
    """
    k = learned_rule["k"]
    activations_search_space_size = 2 ** (2 ** len(k))
    # print("K", k)
    rule = learned_rule["rule"]
    # print("rule:", rule)
    x_next = convolve(x, k, mode="wrap")
    # print("x_next: ", x, "->", x_next)
    result = np.zeros(x.shape)
    for i in range(len(result)):
        n = x_next[i]
        if n == 0:
            continue
        # n_s = str(n).split(".")[0]
        n_s = int_to_activation_set_hash(int(n), activations_search_space_size)
        # if n > 0:
        #     print("n: ", n, n_s)
        # TEST: generate purely random state
        # if np.random.rand() > 0.5:
        #     result[i] = 1
        # continue
        # print("n_s", n_s)
        if n_s in rule:
            result[i] = 1
            #     # print("n: ", n, n_s, rule[n_s])
            # ## Option 1. Random:
            random_value = np.random.rand()
            # ## Option 2. Deterministic
            # random_value = 0.5
            # # check against rule prob
            prob = rule[n_s]
            if random_value <= prob:
                result[i] = 1
            # not necessary, but being explicit
            else:
                result[i] = 0
        # # else:
        # #     print("n: ", n, n_s, None)
    return result


def int_to_activation_set(n, search_space_size=2 ** 32):
    """
    Decodes activation binary set from integer
    """
    width = int(log(search_space_size, 2))
    bitstring = np.binary_repr(n, width=width)
    bitarr = bitarray(bitstring)
    a = np.array(bitarr.tolist(), dtype=np.uint8)
    return a


def int_to_activation_set_hash(n, search_space_size=2 ** 32):
    """
    Decodes activation binary set from integer
    """
    width = int(log(search_space_size, 2))
    bitstring = np.binary_repr(n, width=width)
    return bitstring
    # bitarr = bitarray(bitstring)
    # a = np.array(bitarr.tolist(), dtype=np.uint8)
    # return a


def uint8_tuple_to_bin_arr(t):
    """
    Convert a uint8 tuple into an array of booleans.
    """
    return np.unpackbits(np.array(t, dtype=np.uint8))


def bin_arr_to_s(b):
    """
    Convert binary arr to string of 1,0
    """
    return "".join(map(str, b.tolist()))


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
    x_next = convolve(x, k, mode="wrap")
    x_norm = np.linalg.norm(x_next)
    if x_norm == 0:
        return x
    return np.abs(np.round(x_next / x_norm))


def gol(x, k):
    x_next = convolve(x, k, mode="wrap")
    # Activation Function 1: - binary
    # res = np.array(x_next * (x_next > 0) > 0, dtype=int)

    # Activation Function 2: based on population
    res = np.array(x_next * (x_next > 1) * (x_next < 3), dtype=bool).astype(int)

    # print("res: ", res)
    return res


def rule_30():
    # States with 0 added

    # kernel is 3 consequtive primes
    k = tens(3)

    # k_states is all the combinations of products of k
    k_states = np.array(generate_combined_products(k) + [0])

    width = 32
    # seed is initial state
    seed = np.zeros((width,), dtype=int)
    seed[floor(width / 2)] = 1  # add 1 to middle

    # k_states:  [1, 11, 111, 101, 10, 110, 100, 0]
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)

    # this is the callback
    def r30(x, k):
        return wolfram(x, k, a, k_states)

    # TODO: do something with results
    states = run(width, seed=seed, kernel=k, f=r30)

    mets = metrics(states)

    print("Metrics: ", mets)
    # # Pretty print for LaTex
    # for s in states:
    #     print(" & ".join(map(str, s.tolist())) + "\\")


def wolfram(x, k, r_set):
    """
    k: kernel operator
    a: activation
    k_states: kernel combination space
    """
    x_next = convolve(x, k, mode="constant", cval=0.0)
    matches = np.isin(x_next, r_set)
    result = np.where(matches, 1, 0)
    return result


def run(steps, seed=seed, kernel=kernel, f=f):
    results = [seed]
    a_b = np.copy(seed)
    # print("{}: {}".format(0, seed))
    for i in range(steps):
        a_b = f(a_b, kernel)
        r = a_b.copy()
        # print("{}: {}".format(i + 1, r))
        results.append(r)
    return results


def print_states(states):
    for i in range(len(states)):
        print("{}: {}".format(i, states[i]))


def image_from_states(states, f_name, max_height=64):
    mat = np.array(np.uint8(np.logical_not(states[0:max_height])) * 255)
    # for s in mat:
    #     print(s)
    # print("mat:", mat)
    im = Image.fromarray(mat, mode="L")
    print("Saving image: ", f_name)
    im.save(f_name)
    return im


def print_markdown_table(labels, rows):
    # labels
    print("| " + " | ".join(labels) + " |")

    # divider
    print("| " + " | ".join(list(map(lambda x: "----", labels))) + " |")
    for row in rows:
        # row
        print("| " + " | ".join(list(map(str, row))) + " |")


# TODO:
def print_latex_table(labels, rows):
    return


# class OneD(Automata):


# Example when run as main
if __name__ == "__main__":
    rule_30()

