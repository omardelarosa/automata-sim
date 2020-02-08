import numpy as np
from scipy.ndimage import convolve
from scipy.stats import entropy
from math import floor, ceil, sqrt, log

seed = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])

print("seed: {}, kernel: {}".format(seed, kernel))
a_b = convolve(seed, kernel)

results = []
print("results: ")

# Normalize
def f(x):
    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        return x
    return np.abs(np.round(x / x_norm))


def run(steps, seed=seed, kernel=kernel):
    results = []
    a_b = seed
    for i in range(steps):
        a_b = f(convolve(a_b, kernel))
        r = a_b.copy()
        print("{}: {}".format(i, r))
        results.append(r)
    return results


# Example when run as main
if __name__ == "__main__":
    # TODO: do something with results
    run(10)
