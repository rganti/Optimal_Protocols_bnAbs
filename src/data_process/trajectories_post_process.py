import pickle

import numpy as np
from src.general.io_handling import pickle_out_data, pickle_in_data


def initialize_dictionaries(num_odes):
    number_bin_dict = {}
    number_success_dict = {}

    for i in range(1, num_odes):
        number_bin_dict['N{0}'.format(i)] = 0
        number_success_dict['N{0}'.format(i)] = 0

    return number_bin_dict, number_success_dict


class ComputeTrajectorySuccessProbability(object):
    def __init__(self, num_odes=8, path=""):

        # self.sigma_1 = round(sigma_1, 2)
        # self.sigma_2 = round(sigma_2, 2)
        # self.num_gcs = num_gcs
        self.num_odes = num_odes
        self.start_index = 2
        self.path = path  # + "Sigma_{0}/injection_2_sig_{1}/".format(self.sigma_1, self.sigma_2)

    def compute_num_success_bin(self):

        number_bin_dict, number_success_dict = initialize_dictionaries(self.num_odes)
        boost_success = np.loadtxt(self.path + "successful_exit")

        if len(boost_success.shape) == 1:
            boost_success = np.array([boost_success])

        for entry in boost_success:
            t = int(entry[0])

            pickle_in = open(self.path + "trajectory_{0}.pickle".format(t), "rb")
            sequences = pickle.load(pickle_in)
            pickle_in.close()

            for i in range(1, self.num_odes):
                total = np.where(sequences[:, 1] == i)[0]
                success = np.where(sequences[total][:, self.start_index + self.num_odes/2] == 1.0)[0]

                number_bin_dict['N{0}'.format(i)] += len(total)
                number_success_dict['N{0}'.format(i)] += len(success)

        return number_success_dict, number_bin_dict

    def process_trajectories(self):
        number_success_dict, number_bin_dict = self.compute_num_success_bin()
        pickle_out_data(number_success_dict, self.path + "num_success_dict")
        pickle_out_data(number_bin_dict, self.path + "num_bin_dict")


def load_dictionaries(path):
    number_success_dict = pickle_in_data(path + "num_success_dict")
    return number_success_dict


class ComputeStatistics(object):

    def __init__(self, sigma_1, sigma_2):
        self.sigma_1 = round(sigma_1, 2)
        self.sigma_2 = round(sigma_2, 2)
        self.num_trials = 30
        self.num_odes = 16
        self.number_bin_dict, self.number_success_dict = initialize_dictionaries(self.num_odes)

    def process_trials(self):

        for t in range(self.num_trials):
            number_success_trial = pickle_in_data("Sigma_{0}/Trial_{1}/injection_2_sig_{2}/num_success_dict".format(self.sigma_1, t, self.sigma_2))

            for k in self.number_success_dict.keys():
                self.number_success_dict[k] += number_success_trial[k]

        pickle_out_data(self.number_success_dict, "Sigma_{0}/total_num_success_dict".format(self.sigma_1))


if __name__ == "__main__":
    optimal_pairs = np.loadtxt("../optimal_pairs")

    for sigma_1, sigma_2 in optimal_pairs[1:-1]:

        total_success = ComputeStatistics(sigma_1=sigma_1, sigma_2=sigma_2)
        total_success.process_trials()

