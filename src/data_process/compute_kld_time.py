import matplotlib.pyplot as plt
import numpy as np


def compute_kl_distance(p_ini, f):
    kl_distance = []
    for i in range(len(p_ini)):
        if p_ini[i] == 0:
            kl_distance.append(0)
        else:
            kl_distance.append(p_ini[i] * np.log(p_ini[i] / f[i]))

    return np.sum(kl_distance)


class KLDTime(object):

    def __init__(self, path):
        self.path = path
        self.fitness = np.loadtxt(self.path + "fitness")
        self.p0 = np.loadtxt(self.path + "p0")
        self.kld_0 = compute_kl_distance(self.p0, self.fitness)

    def compute_p(self, i, n_stop, survive=True):
        if survive:
            successful_exit = np.loadtxt(self.path + "successful_exit")
            traj_index, exit_index = np.array(successful_exit[i], int)
            num_gc_success = np.loadtxt(self.path + "hashed_traj_{0}".format(traj_index))
            exit_index = np.where(np.sum(num_gc_success[:, 1:], axis=1) >= n_stop)[0][0]
            p_gc = num_gc_success[:, 1:][:exit_index + 1] / np.sum(num_gc_success[:exit_index + 1, 1:],
                                                                   axis=1)[:, np.newaxis]
            ntot = n_stop
        else:
            unsuccessful_exit = np.loadtxt(self.path + "unsuccessful_exit")
            traj_index, exit_index = np.array(unsuccessful_exit[i], int)
            num_gc_fail = np.loadtxt(self.path + "hashed_traj_{0}".format(traj_index))
            p_gc = num_gc_fail[:, 1:][:exit_index] / np.sum(num_gc_fail[:exit_index, 1:],
                                                            axis=1)[:, np.newaxis]
            ntot = 0

        return p_gc, exit_index, ntot

    # def compute_p_success(self, i, n_stop):
    #     traj_index, exit_index = np.array(self.successful_exit[i], int)
    #     num_gc_success = np.loadtxt(self.path + "hashed_traj_{0}".format(traj_index))
    #     exit_index = np.where(np.sum(num_gc_success[:, 1:], axis=1) >= n_stop)[0][0]
    #     p_gc_success = num_gc_success[:, 1:][:exit_index + 1] / np.sum(num_gc_success[:exit_index + 1, 1:],
    #                                                                    axis=1)[:, np.newaxis]
    #
    #     return p_gc_success

    # def compute_stop_time(self, i, n_stop):
    #     traj_index, exit_index = np.array(self.successful_exit[i], int)
    #     num_gc_success = np.loadtxt(self.path + "hashed_traj_{0}".format(traj_index))
    #     exit_index = np.where(np.sum(num_gc_success[:, 1:], axis=1) >= n_stop)[0][0]
    #     ntot = np.sum(num_gc_success[:, 1:][exit_index])
    #
    #     return exit_index, ntot

    # def compute_extinction_time(self, i):
    #     traj_index, exit_index = np.array(self.unsuccessful_exit[i], int)
    #
    #     return exit_index, 0

    # def compute_p_fail(self, i):
    #     traj_index, exit_index = np.array(self.unsuccessful_exit[i], int)
    #     num_gc_fail = np.loadtxt(self.path + "hashed_traj_{0}".format(traj_index))
    #     p_gc_fail = num_gc_fail[:, 1:][:exit_index] / np.sum(num_gc_fail[:exit_index, 1:],
    #                                                                    axis=1)[:, np.newaxis]
    #
    #     return p_gc_fail

    def plot(self, i, n_stop, survive=True):
        kld_time, exit_time, ntot = self.compute_kld_time(i, n_stop=n_stop, survive=survive)

        plt.plot(range(len(kld_time)), kld_time)

        # n_stop_array = [n_stop]
        # color_array = ['k']
        # for k in range(len(n_stop_array)):
        # if survive:
        #     t_1, ntot = self.compute_stop_time(i, n_stop)
        # else:
        #     t_1, ntot = self.compute_extinction_time(i)

        if i == 0:
            if survive:
                plt.axvline(x=exit_time, linestyle='--', color='k', label="$N_{max}= $" + str(int(ntot)))
            else:
                plt.axvline(x=exit_time, linestyle='--', color='k', label="$N_{exit}= $" + str(int(ntot)))
        else:
            plt.axvline(x=exit_time, linestyle='--', color='k')

    # def plot_fail(self, i, n_stop):
    #     kld_time = self.compute_kld_time(i, n_stop=n_stop, survive=False)
    #     plt.plot(range(len(kld_time)), kld_time)
    #
    #     t_1, ntot = self.compute

    def compute_kld_time(self, i, n_stop, survive=True):
        p_gc, exit_time, ntot = self.compute_p(i, n_stop, survive)
        # if survive == True:
        #     p_gc = self.compute_p_success(i, n_stop)
        # else:
        #     p_gc = self.compute_p_fail(i)

        kld_time = []

        for i in range(len(p_gc)):
            kld = compute_kl_distance(p_gc[i], self.fitness)
            kld_time.append(kld)

        return kld_time, exit_time, ntot
