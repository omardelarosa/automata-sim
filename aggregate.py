import json
import numpy as np
import glob
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import cm

prefix = "./renderings/midi/kernel_space_3x1/t_1581277553_"


def aggregate_summary(f_name_prefix):
    f_patt = "{prefix}*.json".format(prefix=f_name_prefix)

    files = glob.glob(f_patt)

    seeds_kernels = {}
    keys = [
        "num_states",
        "entropy_score",
    ]
    for f in files:
        f_name = f.replace(f_name_prefix, "")
        f_parts = f_name.split("_")
        seed_key = f_parts[1]
        kernel_key = f_parts[2]
        if seed_key not in seeds_kernels:
            seeds_kernels[seed_key] = {}
        seed = seeds_kernels[seed_key]

        with open(f, "r") as f_json:
            data_dict = json.load(f_json)
            data = {}
            if "kernel" in data_dict:
                data["kernel"] = data_dict["kernel"]
                for k in keys:
                    data[k] = data_dict["metrics"][k]
            seed[kernel_key] = {"file_name": f_name, "data": data}
        # print("file: ", seed_key, kernel_key)

    print(seeds_kernels)

    summary_file = "{}_summary.json".format(f_name_prefix)

    # Write summary as json file
    with open(summary_file, "w") as json_file:
        json.dump(seeds_kernels, json_file)


def plot_summary(summary_file):
    print("summary_file: ", summary_file)
    d = {}
    with open(summary_file, "r") as f_json:
        d = json.load(f_json)

    # seeds
    for seed_key in d:
        kernels = d[seed_key]
        # print("kernels: ", kernels.keys())
        # kernels
        n = len(kernels.keys())
        kernel_points = np.zeros((n, 5), dtype=np.float64)
        i = 0
        for kernel_key in kernels:
            kernel = kernels[kernel_key]["data"]
            k = kernel["kernel"]
            entropy = kernel["entropy_score"]
            num_states = kernel["num_states"]
            kernel_points[i] = [k[0], k[1], k[2], entropy, num_states]
            i += 1
        # k_vec = np.array(kernel_points)
        print(kernel_points)

        # Take column slices
        x = kernel_points[:, 0]
        y = kernel_points[:, 2]
        Z = kernel_points[:, 3]
        D = kernel_points[:, 4]
        # D = kernel_points[:, ]
        # X, Y = np.meshgrid(x, y)
        # Z = np.outer(P, P)
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.set_title("Seed: {} - Entropy".format(seed_key))
        ax.plot_trisurf(x, y, Z, cmap=cm.coolwarm)

        ax.set_xlabel("x (left operand)")
        ax.set_ylabel("y (right operand)")
        ax.set_zlabel("entropy")

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.set_title("Seed: {} - Diversity of States".format(seed_key))
        ax.plot_trisurf(x, y, D, cmap=cm.coolwarm)

        ax.set_xlabel("x (left operand)")
        ax.set_ylabel("y (right operand)")
        ax.set_zlabel("num_states")

    plt.show()


# plot_summary(prefix + "_summary.json")
