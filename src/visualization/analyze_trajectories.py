import os

import numpy as np

from src.data_process.compute_kld_time import compute_kl_distance
from src.visualization.visualize_fitness import InjectionKlDistance


def dict_to_array(dictionary, num_odes=16):
    array = np.zeros(num_odes)
    for i in range(1, num_odes):
        array[i] = dictionary['N{0}'.format(i)]

    return array


def compute_n_ave(path):
    n_ave_array = [np.loadtxt(path + "Trial_{0}/n_ave".format(t)) for t in range(30)]
    return np.mean(n_ave_array, axis=0)


class EntropyDistributions(object):

    def __init__(self, path=""):
        self.path = path
        self.p0 = np.loadtxt(self.path + "event_prob")
        self.sigma = np.loadtxt(self.path + "sigma")
        # self.h_array = self.compute_h_array()
        self.kl_distance = InjectionKlDistance(self.p0, self.sigma, num_odes=16)
        self.fitness = np.loadtxt(self.path + "fitness")
        self.kld_array = self.compute_kld_array()

    def compute_kld_array(self):
        kld_array = []
        success_exit = np.loadtxt(self.path + "successful_exit")
        for i in range(len(success_exit)):
            traj_index, exit_index = np.array(success_exit[i], int)
            traj = np.loadtxt(self.path + "hashed_traj_{0}".format(traj_index))
            p0 = traj[0][1:] / np.sum(traj[0][1:])
            kld_0 = compute_kl_distance(p0, self.fitness)

            p_final = traj[exit_index][1:] / np.sum(traj[exit_index][1:])
            kld_final = compute_kl_distance(p_final, self.fitness)

            kld_array.append(kld_final - kld_0)

        unsuccess_exit = np.loadtxt(self.path + "unsuccessful_exit")
        for j in range(len(unsuccess_exit)):
            traj_index, exit_index = np.array(unsuccess_exit[j], int)
            traj = np.loadtxt(self.path + "hashed_traj_{0}".format(traj_index))
            p0 = traj[0][1:] / np.sum(traj[0][1:])
            kld_0 = compute_kl_distance(p0, self.fitness)

            p_final = traj[exit_index - 1][1:] / np.sum(traj[exit_index - 1][1:])
            kld_final = compute_kl_distance(p_final, self.fitness)

            kld_array.append(kld_final - kld_0)

        return kld_array

    def compute_entropy(self, p):
        h = []
        for p_i in p:
            if p_i == 0:
                h.append(p_i)
            else:
                h.append(-p_i * np.log(p_i))

        return np.sum(h)

    def check_dimension(self, exit_array):
        if exit_array.shape[0] > 0:
            if len(exit_array.shape) == 1:
                exit_array = np.array([exit_array])

        return exit_array

    def compute_h_array(self):
        if os.path.exists(self.path + "h_array"):
            h_array = np.loadtxt(self.path + "h_array")

        else:
            h_array = []

            prime_success_exit = np.loadtxt(self.path + "successful_exit")
            prime_success_exit = self.check_dimension(prime_success_exit)

            for entry in prime_success_exit:
                t = int(entry[0])
                traj = np.loadtxt(self.path + "hashed_traj_{0}".format(t))
                exit = int(entry[1])

                p = traj[exit][1:] / np.sum(traj[exit][1:])
                h_array.append(self.compute_entropy(p) - self.compute_entropy(self.p0))

            prime_unsuccessful = np.loadtxt(self.path + "unsuccessful_exit")
            prime_unsuccessful = self.check_dimension(prime_unsuccessful)

            for entry in prime_unsuccessful:
                h_array.append(0.0 - self.compute_entropy(self.p0))

            np.savetxt(self.path + "h_array", h_array, fmt='%f')

        return h_array


# class PlotSuccessfulBins(object):
#     def __init__(self):
#         self.number_success_dict = self.load_dictionaries()
#
#     def load_dictionaries(self):
#         number_success_dict = pickle_in_data("num_success_dict")
#
#         return number_success_dict
#
#     def plot_bnabs_bin(self):
#         plt.plot(range(1, len(self.number_success_dict.keys())), self.number_success_dict.values(), linestyle='None', marker='o',
#                  label="Low: $\\sigma_{1} = $" + "{0}".format(low.sigma_1) + ", $\\sigma_{2} = $" +
#                        "{:01.2f}".format(low.sigma_2_range[3]) + ", Total bnAbs = {0}".format(
#                      np.sum(low.success_array[3])))




