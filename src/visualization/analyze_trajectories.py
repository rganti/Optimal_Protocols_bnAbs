import numpy as np
from src.visualization.visualize_fitness import InjectionKlDistance


class EntropyDistributions(object):

    def __init__(self, path=""):
        self.path = path  # "Sigma_{0}/Trial_0/".format(round(self.sigma, 2))
        self.p0 = np.loadtxt(self.path + "event_prob")
        self.sigma = np.loadtxt(self.path + "sigma")
        self.h_array = self.compute_h_array()
        self.kl_distance = InjectionKlDistance(self.p0, self.sigma, num_odes=16)

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
        p_i_array = []
        h_array = []

        prime_success_exit = np.loadtxt(self.path + "successful_exit")
        prime_success_exit = self.check_dimension(prime_success_exit)

        for entry in prime_success_exit:
            t = int(entry[0])
            traj = np.loadtxt(self.path + "hashed_traj_{0}".format(t))
            exit = int(entry[1])

            p = traj[exit][1:] / np.sum(traj[exit][1:])
            h_array.append(self.compute_entropy(p) - self.compute_entropy(self.p0))
            p_i_array.append(p)

        prime_unsuccessful = np.loadtxt(self.path + "unsuccessful_exit")
        prime_unsuccessful = self.check_dimension(prime_unsuccessful)

        for entry in prime_unsuccessful:
            h_array.append(0.0 - self.compute_entropy(self.p0))

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




