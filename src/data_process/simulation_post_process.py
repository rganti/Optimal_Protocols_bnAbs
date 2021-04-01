import os

import numpy as np

from src.general.directory_handling import load, getcolumnames


class PostProcess(object):
    def __init__(self, trajectory_file):
        self.trajectory_file = trajectory_file
        self.data = load(self.trajectory_file)
        self.column_names = getcolumnames(self.data)

    def write_columns(self):
        f = open("column_names", "w")
        for item in self.column_names:
            f.write("{0} ".format(item))
        f.write("\n")
        f.close()

    def insert_hash(self):
        self.data[0] = "# " + self.data[0]

    def write_data(self):
        f = open("hashed_{0}".format(self.trajectory_file), "w")
        f.write("".join(map(lambda x: x, self.data)))
        f.close()
        # os.remove(self.trajectory_file)

    def main(self):
        self.insert_hash()
        self.write_data()


class BnabGcExit(object):

    def __init__(self, n_exit=200, num_odes=8, num_files=100):
        if num_files:
            self.num_files = num_files
        else:
            self.num_files = len([name for name in os.listdir(".") if "traj" in name])

        self.num_odes = num_odes
        self.n_exit = n_exit
        self.start_index = 1

        self.pn_dict = {}
        for i in range(1, self.num_odes):
            self.pn_dict['n{0}'.format(i)] = []

    def post_process_files(self):
        for i in range(self.start_index, self.num_files):
            post_process = PostProcess("traj_{0}".format(i))
            post_process.main()

    def populate_pn(self, index=2):
        successful_exit = []
        unsuccessful_exit = []
        no_exit = []

        for i in range(self.start_index, self.num_files):
            print("trajectory: " + str(i))
            traj = np.loadtxt("hashed_traj_{0}".format(i))
            ntot = np.sum(traj[:, index:], axis=1)
            survival_condition = np.where(ntot >= self.n_exit)[0]
            death_condition = np.where(ntot == 0)[0]

            if len(survival_condition) > 0:
                exit_index = survival_condition[0]

                print(str(traj[exit_index, index:]))
                for j in range(1, self.num_odes):
                    self.pn_dict['n{0}'.format(j)].append(traj[exit_index, index:][j-1])

                successful_exit.append((i, exit_index))

            elif len(death_condition) > 0:
                exit_index = death_condition[0]
                print(str(traj[exit_index, index:]))
                for j in range(1, self.num_odes):
                    self.pn_dict['n{0}'.format(j)].append(traj[exit_index, index:][j-1])

                unsuccessful_exit.append((i, exit_index))

            else:
                print("No GC exit for traj{0}".format(i))
                no_exit.append(i)

        np.savetxt("successful_exit", successful_exit, fmt='%f')
        np.savetxt("unsuccessful_exit", unsuccessful_exit, fmt='%f')
        np.savetxt("no_exit", no_exit, fmt='%f')

        n_ave = []
        for k in range(1, self.num_odes):
            np.savetxt("n{0}".format(k), self.pn_dict['n{0}'.format(k)], fmt='%f')
            n_ave.append(np.mean(self.pn_dict['n{0}'.format(k)]))

        np.savetxt("n_ave", n_ave, fmt='%f')

    def main(self):
        self.post_process_files()
        self.populate_pn()


class GillespieGCExit(BnabGcExit):

    def __init__(self, n_exit=200, num_odes=8, num_files=None):
        BnabGcExit.__init__(self, n_exit=n_exit, num_odes=num_odes, num_files=num_files)
        if num_files:
            self.num_files = num_files
        else:
            len([name for name in os.listdir(".") if "hashed_traj" in name])
        self.start_index = 0


def get_frequencies(directory):
    nave = np.loadtxt("{0}/n_ave".format(directory))
    n4 = np.loadtxt("{0}/n4".format(directory))
    # p_ini = np.loadtxt("{0}/event_prob".format(directory))
    return nave, n4


def frequencies_to_file(f, directory):
    nave, n4 = get_frequencies(directory)
    # f.write("p_ini: " + str(p_ini) + "\n")
    f.write("n_ave: " + str(nave) + "\n")
    f.write("n4: " + str(n4) + "\n\n")


def print_info(sigma):

    f = open("simulation_output", "w")

    f.write("Results: \n")
    f.write("Injection 1: sigma = {0} \n".format(sigma[0]))
    success_exit = np.loadtxt("successful_exit")
    f.write("Number of Successful Exits = {0} \n".format(len(success_exit)))

    fitness = np.loadtxt("fitness")
    f.write("Fitness 1: " + str(fitness) + "\n")
    frequencies_to_file(f, ".")

    for j in range(1, len(sigma)):
        directory = "injection_2_sig_{0}".format(round(sigma[j], 2))
        f.write("Injection 2: sigma = {0} \n".format(sigma[j]))
        success_exit = np.loadtxt("{0}/successful_exit".format(directory))
        f.write("Number of Successful Exits = {0} \n".format(len(success_exit)))
        fitness = np.loadtxt("{0}/fitness".format(directory))
        f.write("Fitness 2: " + str(fitness) + "\n")

        frequencies_to_file(f, directory)

    f.close()
