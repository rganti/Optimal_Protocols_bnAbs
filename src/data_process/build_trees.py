import argparse
import pickle

import numpy as np
from ete3 import Tree

np.set_printoptions(suppress=True)
import multiprocessing as mp
import time
import sys

sys.setrecursionlimit(30000)


def recursive_function(node, indices, time, sequences, previous_indices):
    # print("previous indices = " + str(previous_new))
    # if time < sequences.shape[0]:

    # CHANGE TO WHILE Loop: while time < sequences.shape[0]:
    if time < sequences.shape[0]:
        if time == sequences.shape[0] - 1:
            print("t = " + str(time))
        # print("Node = " + str(node.name) + ": indices = " + str(indices))
        # new_indices = []
        for i in range(sequences.shape[1]):
            if (sequences[time][i, 0] in indices + 1) and (i not in previous_indices):
                indices = np.append(indices, i)
                previous_indices.append(i)

        all_indices = indices
        sum_eligible_seq = np.sum(sequences[time][all_indices][:, 1:], axis=0)

        for j in range(len(sum_eligible_seq)):
            if sum_eligible_seq[j] > 0:
                child_node = node.add_child(name="N{0}".format(j))
                child_node.add_feature("numbers", sum_eligible_seq[j])

                if j > 0:
                    child_indices = np.array([index for index in all_indices if sequences[time][:, 1:][index][j] == 1])
                    if len(child_indices) > 0:
                        # (node, indices, time) = (child_node, child_indices, time + 1)
                        # Need to define so it is iterative: redefine node = child_node, indices = child_indices, time = time +1
                        # try:
                        recursive_function(child_node, child_indices, time + 1, sequences, previous_indices)

                        # except RuntimeError:
                        #     print("recursion depth reached!")
                        #     sys.setrecursionlimit(sys.getrecursionlimit() + 5000)
                        #     recursive_function(child_node, child_indices, time + 1, sequences, previous_indices)


def compute_all_bnab_leaves(t):
    number_bnab_paths = 0
    for node in t.get_leaves():
        if node.name == "N4":
            number_bnab_paths += 1

    return number_bnab_paths


class OutputTreeFeatures(object):

    def __init__(self, tree, number):
        self.t = tree
        self.number = number
        self.num_odes = 16

    def compute_total_paths(self):
        total_paths = np.zeros(2)
        total_paths[0] = len(self.t.get_leaves())

        for node in self.t.traverse():
            if node.name == "N{0}".format(self.num_odes / 2):
                total_paths[1] += 1
        return total_paths

    def compute_bnab_leaf_distribution(self):
        path_array = np.zeros(self.num_odes)
        path_array[0] = compute_all_bnab_leaves(self.t)

        count = 0
        for node in self.t.get_children():
            for i in range(1, len(path_array)):
                if node.name == "N{0}".format(i):
                    path_array[i] += compute_all_bnab_leaves(node)

            count += 1

        return path_array

    def compute_bnab_number_distribution(self):
        bnab_histogram = []
        for node in self.t.get_leaves():
            if node.name == "N{0}".format(self.num_odes / 2):
                bnab_histogram.append(node.numbers)

        return bnab_histogram

    def check_leaves(self):
        seq_array = np.zeros(self.num_odes)
        for node in self.t.get_leaves():
            for i in range(len(seq_array)):
                if node.name == "N{0}".format(i):
                    seq_array[i] += node.numbers

        return seq_array

    def check_all_nodes(self):
        total_array = np.zeros(self.num_odes)
        for node in self.t.traverse():
            for i in range(len(total_array)):
                if node.name == "N{0}".format(i):
                    total_array[i] += node.numbers
        return total_array

    def run(self):
        tree_features = open("t_{0}_features".format(self.number), "w")
        tree_features.write("final seq array: " + str(self.check_leaves()) + "\n")
        tree_features.write("total seq array: " + str(self.check_all_nodes()) + "\n")
        tree_features.close()


class BuildTree(object):

    def __init__(self, trajectory_file):
        pickle_in = open(trajectory_file, "rb")
        self.full_sequences = pickle.load(pickle_in)
        pickle_in.close()
        self.t = call_recursive(self.full_sequences)


# def set_tree_style():
#     ts = TreeStyle()
#     ts.show_leaf_name = False
#     ts.show_branch_length = False
#     ts.show_branch_support = False
#     ts.layout_fn = my_layout
#
#     return ts


def call_recursive(full_sequences):
    t = Tree()

    t.add_feature("numbers", 0)
    previous_indices = []

    for i in range(len(full_sequences[0][:50][:, 1:])):
        seq_index = np.where(full_sequences[0][:50][:, 1:][i] == 1)[0][0]
        indices = np.array([i])
        seqi_node = t.add_child(name="N{0}".format(seq_index))
        seqi_node.add_feature("numbers", len(indices))

        previous_indices.append(i)
        time = 1
        recursive_function(seqi_node, indices, time, full_sequences, previous_indices)

    return t


def build(num):
    num = int(num)
    print("Worker: " + str(num))
    pickle_in = open("trajectory_{0}.pickle".format(num), "rb")
    full_sequences = pickle.load(pickle_in)
    pickle_in.close()

    t = call_recursive(full_sequences)

    # bnab_leaf_array = compute_bnab_leaves(t)
    # total_path_array = compute_total_paths(t)
    # bnab_histogram = compute_bnab_distribution(t)

    pickle_out = open("full_tree_{0}.pickle".format(num), "wb")
    pickle.dump(t, pickle_out)
    pickle_out.close()

    output = OutputTreeFeatures(t, num)
    output.run()

    # bnab_leaf_array = output.compute_bnab_leaf_distribution()
    # total_nodes = output.compute_total_paths()
    # bnab_number_distribution = output.compute_bnab_number_distribution()

    # return bnab_leaf_array, total_nodes, bnab_number_distribution


def output_features(num):
    pickle_in = open("full_tree_{0}.pickle".format(num), "rb")
    t = pickle.load(pickle_in)
    pickle_in.close()

    output = OutputTreeFeatures(t, num)
    output.run()


def tree_mp():
    t0 = time.time()

    success_exit = np.loadtxt("successful_exit")

    if len(success_exit.shape) == 1:
        success_exit = np.array([success_exit])

    pool = mp.Pool(processes=16)

    # Preserving order
    # gc_bnab_leaves, gc_total_paths, gc_bnab_histogram = zip(*pool.map(build, success_exit[:, 0]))
    pool.map(build, success_exit[:, 0])

    # # Output
    # np.savetxt("tree_bnab_leaves", gc_bnab_leaves, fmt='%f')
    # np.savetxt("tree_total_paths", gc_total_paths, fmt='%f')
    #
    # pickle_out = open("tree_bnab_histogram.pickle", "wb")
    # pickle.dump(gc_bnab_histogram, pickle_out)
    # pickle_out.close()

    pool.close()
    pool.join()

    t1 = time.time()
    print("Elapsed time: " + str(t1 - t0))
    np.savetxt("time", [t1 - t0], fmt='%f')
    # os.remove("trajectory_*.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submitting immune protocol calculations.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file', dest='file', type=int, action='store')
    args = parser.parse_args()

    print("Only running on tree " + str(args.file))
    build(args.file)
    # output_features(args.file)
