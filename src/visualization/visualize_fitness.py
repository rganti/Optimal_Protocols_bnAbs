import numpy as np


class Injection(object):

    def __init__(self, sigma=10.0, num_odes=8):
        self.num_odes = num_odes

        self.x_array = self.compute_x_array()
        self.sigma = sigma
        self.f = (self.compute_fitness(self.sigma) / np.sum(self.compute_fitness(self.sigma)))

        self.h_uniform = self.compute_h_uniform()
        self.h_f = self.compute_h_fitness()
        self.delta_h = self.compute_delta_h()  # -(self.h_f - self.h_uniform)

    def compute_h_uniform(self):
        h_max = -np.log(1.0/(self.num_odes - 1))
        return h_max

    def compute_h_fitness(self):
        h_fitness = -np.sum(self.f * np.log(self.f))
        return h_fitness

    def compute_delta_h(self):
        delta_h = -(self.compute_h_fitness() - self.compute_h_uniform())
        return delta_h

    def compute_x_array(self, edge=2.5):
        bins = np.linspace(-edge, edge, num=self.num_odes)
        x_array = []
        for i in range(len(bins) - 1):
            x_array.append(bins[i] + (bins[i + 1] - bins[i]) / 2.0)

        return x_array

    def compute_fitness(self, sigma):
        x_array = self.compute_x_array()
        p = (1 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(-(np.array(x_array)) ** (2) / (2 * (sigma ** 2)))
        return p


class InjectionKlDistance(Injection):

    def __init__(self, p_ini, sigma=10.0):
        Injection.__init__(self, sigma=sigma, num_odes=8)
        self.p_ini = p_ini

    def compute_kl_distance(self):
        kl_distance = []
        for i in range(len(self.p_ini)):
            if self.p_ini[i] == 0:
                kl_distance.append(0)
            else:
                kl_distance.append(self.p_ini[i] * np.log(self.p_ini[i] / self.f[i]))

        return np.sum(kl_distance)


class ProcessProtocolData(object):

    def __init__(self, path, num_gcs=100.0):
        self.sigma_1_range = np.sort(np.logspace(1.0, -1.0, num=15))[::-1]
        self.fitness_array = []
        self.kl1_array = []
        self.kl2_matrix = []
        self.mean_bnab_array = []
        self.error_bnab_array = []
        self.path = path
        self.num_gcs = num_gcs

    def load_arrays(self):
        for sigma1 in self.sigma_1_range[:-3]:
            print("Processing sigma_1 = {0}".format(sigma1))

            p0 = []
            total_bnabs = []

            path = self.path + "Sigma_{0}/".format(round(sigma1, 2))
            p0.append(np.loadtxt(path + "event_prob"))
            total_bnabs.append(np.loadtxt(path + "total_bnabs"))

            p0_mean = np.mean(p0, axis=0)
            kl1 = InjectionKlDistance(p0_mean, sigma=sigma1)
            self.kl1_array.append(kl1.compute_kl_distance())

            self.fitness_array.append(kl1.f)
            self.mean_bnab_array.append(np.array(total_bnabs) / self.num_gcs)

            sigma2_range = np.loadtxt(path + "sigma2_range")

            kl2_array = []
            for sigma2 in sigma2_range:
                p1 = []
                p1.append(np.loadtxt(path + "n_ave") / np.sum(np.loadtxt(path + "n_ave")))
                p1_mean = np.mean(p1, axis=0)
                kl2 = InjectionKlDistance(p1_mean, sigma=sigma2)
                kl2_array.append(kl2.compute_kl_distance())

            self.kl2_matrix.append(kl2_array)
