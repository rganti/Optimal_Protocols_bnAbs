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