import numpy as np


class ProtocolStatistics(object):

    def __init__(self, path="", num_gcs=100.0):

        self.sigma_1_range = np.loadtxt(path + "sigma_1_range")
        self.path = path
        self.num_gcs = num_gcs

    def process_trial_statistics(self):
        for sigma1 in self.sigma_1_range:
            print("Processing Sigma_1 = {0}".format(sigma1))

            total_bnabs = []
            survival_fraction = []

            for t in range(30):
                path = self.path + "Sigma_{0}/Trial_{1}/".format(round(sigma1, 2), t)
                total_bnabs.append(np.loadtxt(path + "total_bnabs"))
                successful_exit = np.loadtxt(path + "successful_exit")
                if len(successful_exit) > 0:
                    survival_fraction.append(float(len(successful_exit))/self.num_gcs)
                else:
                    survival_fraction.append(0)

            sigma_path = self.path + "Sigma_{0}/".format(sigma1)
            np.savetxt(sigma_path + "mean_bnabs", np.mean(np.array(total_bnabs), axis=0), fmt='%f')
            np.savetxt(sigma_path + "error_bnabs", np.std(np.array(total_bnabs), axis=0), fmt='%f')
            np.savetxt(sigma_path + "survival_fraction", [np.mean(survival_fraction)], fmt='%f')
            np.savetxt(sigma_path + "sigma2_range",
                       np.loadtxt(self.path + "Sigma_{0}/Trial_0/sigma2_range".format(round(sigma1, 2))), fmt='%f')


if __name__ == "__main__":

    compute_protocol_statistics = ProtocolStatistics()
    compute_protocol_statistics.process_trial_statistics()