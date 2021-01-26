import pickle

import numpy as np
from numpy import zeros, log
from numpy.random import multinomial, random

from src.general.io_handling import write_results


def define_n_initial(num_bins, length=1000):
    n_initial = np.zeros(num_bins, dtype=float)

    n_b_cells = 25
    index = (num_bins + 1) / 2

    mu, sigma = 3., 1.

    s = np.random.lognormal(mu, sigma, n_b_cells)
    count, bins = np.histogram(s, bins=range(0, length, length / index))

    n_initial[:len(count)] = count

    s = np.random.lognormal(mu, sigma, n_b_cells)
    count, bins = np.histogram(s, bins=range(0, length, length / index))

    n_initial[-len(count):] = count[::-1]

    return n_initial


class Model(object):
    def __init__(self, p_ini, vnames, tmat, propensity, n_stop=200):
        '''
         * vnames: list of strings
         * inits: list of initial values of variables
         * propensity: list of lambda functions of the form:
            lambda ini: some function of inits.
        '''

        print("Calculations with Model class.")

        self.ntot = 50
        self.p_ini = p_ini
        self.vn = vnames
        self.tm = tmat
        self.pv = propensity  # [compile(eq,'errmsg','eval') for eq in propensity]
        self.pvl = len(self.pv)  # length of propensity vector
        self.nvars = len(self.p_ini) + 1  # number of variables
        self.time = None
        self.series = None
        self.steps = 0
        self.exit = n_stop

    def getStats(self):
        return self.time, self.series, self.steps

    def sample_initial_population(self):
        initial = define_n_initial(len(self.p_ini))
        # events = multinomial(self.ntot, self.p_ini)
        new_initial = np.insert(initial, 0, 0)
        ini = [i for i in new_initial]

        return ini

    def run(self, method='SSA', tmax=10, reps=1):
        # self.res = zeros((tmax, self.nvars, reps), dtype=float)
        tvec = range(1, tmax)

        if method == 'SSA':
            n0 = []
            for i in range(reps):
                res = zeros((tmax, self.nvars), dtype=float)
                steps, tim = self.GSSA(res, tmax=tmax, reps=i)
                write_results(res, i)
                n0.append(res[0, 1:])

            n_ave = np.mean(n0, axis=0)
            p0 = n_ave / np.sum(n_ave)

            np.savetxt("p0", p0)

            # np.savetxt("time_array_{0}".format(i), self.time_array, fmt='%f')
        elif method == 'SSAct':
            pass
        self.time = tvec
        np.savetxt("time", self.time, fmt='%f')

        self.steps = steps
        np.savetxt("steps", [steps], fmt='%f')

    def GSSA(self, res, tmax=50, reps=0):
        '''
        Gillespie Direct algorithm
        '''

        ini = self.sample_initial_population()

        pvi = self.pv
        l = self.pvl
        pv = zeros(l, dtype=float)
        tm = self.tm

        tc = 0
        steps = 0
        # self.res[0, :, reps] = ini
        res[0, :] = ini
        # self.time_array.append(0)
        a0 = 1

        print("rep = " + str(reps))
        print("ini = " + str(ini))
        print("------")
        for tim in range(1, tmax):
            while tc < tim:
                for i in range(l):
                    pv[i] = pvi[i](ini)
                # pv = abs(array([eq() for eq in pvi]))# #propensity vector
                a0 = pv.sum()  # sum of all transition probabilities
                if a0 == 0:
                    break
                if np.sum(ini[1:]) == self.exit:
                    break

                tau = (-1 / a0) * log(random())
                event = multinomial(1, pv / a0)  # event which will happen on this iteration
                ini += tm[:, event.nonzero()[0][0]]
                tc += tau
                steps += 1

            res[tim, :] = ini
            # self.time_array.append(tim)
            if a0 == 0:
                break
            if np.sum(ini[1:]) == self.exit:
                break

        # print(str(res))

        return steps, tim


class ModelIndividualGC(Model):

    def __init__(self, n_exit, p_ini, vnames, tmat, propensity, n_stop=200):
        Model.__init__(self, p_ini, vnames, tmat, propensity, n_stop=n_stop)
        self.n_exit = n_exit

    def sample_initial_population(self):

        ini = np.zeros(len(self.n_exit), dtype=int)

        for i in range(self.ntot):
            # print("n_i = " + str(self.n_exit))
            p_ini = self.n_exit / np.sum(self.n_exit)

            # print("p_ini = " + str(p_ini))
            events = multinomial(1, p_ini)

            bin_index = events.nonzero()[0][0]

            ini[bin_index] += 1
            self.n_exit[bin_index] -= 1

        new_initial = np.insert(ini, 0, 0)
        ini = [i for i in new_initial]

        return ini

    def run(self, method='SSA', tmax=10, reps=1):
        tvec = range(1, tmax)

        if method == 'SSA':
            res = zeros((tmax, self.nvars), dtype=float)
            steps = self.GSSA(res, tmax=tmax, reps=reps)
            write_results(res, reps)
        elif method == 'SSAct':
            pass

        self.time = tvec
        self.steps = steps


def pickle_trajectories(seq_array, i):
    pickle_out = open("trajectory_{0}.pickle".format(i), "wb")
    pickle.dump(seq_array, pickle_out)
    pickle_out.close()


def pickle_transitions(transitions, i):
    pickle_out = open("transitions_{0}.pickle".format(i), "wb")
    pickle.dump(transitions, pickle_out)
    pickle_out.close()


class ModelTrajectories(Model):

    def __init__(self, p_ini, vnames, tmat, propensity, n_exit=None):
        Model.__init__(self, p_ini, vnames, tmat, propensity)
        print("Calculations with ModelTrajectories class.")

        self.number_sequences = 1000
        self.n_exit = n_exit
        self.start_index = 1
        self.sequence_dimensions = self.nvars + self.start_index
        self.t_exit = None

    def sample_initial_population_nexit(self):
        ini = np.zeros(len(self.p_ini), dtype=int)

        for i in range(self.ntot):
            print("n_i = " + str(self.n_exit))
            p_ini = self.n_exit / np.sum(self.n_exit)

            print("p_ini = " + str(p_ini))
            events = multinomial(1, p_ini)

            bin_index = events.nonzero()[0][0]

            ini[bin_index] += 1
            self.n_exit[bin_index] -= 1

        new_initial = np.insert(ini, 0, 0)
        ini = [i for i in new_initial]

        return ini

    def set_ini(self):
        if self.n_exit is not None:
            ini = self.sample_initial_population_nexit()
        else:
            ini = self.sample_initial_population()

        new_ini = np.zeros(self.sequence_dimensions, dtype=float)
        new_ini[self.start_index:] = ini

        print("Initializing Sequence: ")
        print("new ini: " + str(new_ini))

        return new_ini

    def initialize_sequence_array(self):
        new_ini = self.set_ini()

        sequence_array = np.zeros((self.number_sequences, self.sequence_dimensions), dtype=float)
        index = 0
        for i in range(self.start_index + 1, len(new_ini)):
            next_index = int(new_ini[i]) + index
            sequence_array[index:next_index][:, i] = 1
            index += int(new_ini[i])

        return sequence_array

    def run_GSSA(self, tmax, i):
        res = zeros((tmax, self.nvars), dtype=float)
        tc, sequence_time = self.GSSA(res, tmax=tmax, reps=i)
        pickle_trajectories(sequence_time, i)
        write_results(res, i)
        return tc

    def run(self, method='SSA', tmax=10, reps=1):
        tvec = range(1, tmax)

        if self.n_exit is not None:
            tc = self.run_GSSA(tmax, reps)
            self.t_exit = tc
        else:
            tc = []
            for i in range(reps):
                tc_i = self.run_GSSA(tmax, i)
                tc.append(tc_i)
                np.savetxt("tc_gc_exit", tc, fmt='%f')

        self.time = tvec
        np.savetxt("time", self.time, fmt='%f')

        # np.savetxt("tc", [tc], fmt='%f')

    def extend_array(self, ini_sequence, sequence_time, tmax):
        self.number_sequences += 500

        new_ini_sequence = np.zeros((self.number_sequences, self.sequence_dimensions), dtype=float)
        new_ini_sequence[:ini_sequence.shape[0], :ini_sequence.shape[1]] = ini_sequence
        ini_sequence = new_ini_sequence

        new_sequence_time = np.zeros((tmax, self.number_sequences, self.sequence_dimensions), dtype=float)
        new_sequence_time[:sequence_time.shape[0], :sequence_time.shape[1], :sequence_time.shape[2]] = sequence_time
        sequence_time = new_sequence_time

        return ini_sequence, sequence_time

    def replication_event(self, ini_sequence, transition, tc):
        seq_bin = np.where(transition == 1)[0][0]
        seq_indices = np.nonzero(ini_sequence[:, self.start_index:][:, seq_bin])[0]
        chosen_seq = np.random.choice(seq_indices)

        new_seq_index = np.nonzero(ini_sequence)[0][-1] + 1

        # Have to add one when indexing to not confuse with 0th entry of original crop which have index 0 in bin 0
        ini_sequence[new_seq_index][0] = chosen_seq + 1
        ini_sequence[:, self.start_index:][new_seq_index] = transition

        return ini_sequence

    def mutation_event(self, ini_sequence, transition, tc):  # , time_death_step, time_mutation_step):
        seq_bin = np.where(transition == -1)[0][0]
        seq_indices = np.nonzero(ini_sequence[:, self.start_index:][:, seq_bin])[0]
        chosen_seq = np.random.choice(seq_indices)
        ini_sequence[:, self.start_index:][chosen_seq] += transition

        return ini_sequence

    def initialize_all_arrays(self, tmax):
        sequence_time = np.zeros((tmax, self.number_sequences, self.sequence_dimensions), dtype=float)
        ini_sequence = self.initialize_sequence_array()
        ini = np.sum(ini_sequence, axis=0)[self.start_index:]
        sequence_time[0, :] = ini_sequence
        return ini, ini_sequence, sequence_time

    def GSSA(self, res, tmax=50, reps=0):
        ''' Gillespie Direct algorithm '''

        # sequence_time = np.zeros((tmax, self.number_sequences, self.nvars + 1), dtype=int)
        # ini_sequence = self.initialize_sequence_array()
        # ini = np.sum(ini_sequence, axis=0)[1:]
        # sequence_time[0, :] = ini_sequence

        ini, ini_sequence, sequence_time = self.initialize_all_arrays(tmax)

        pvi = self.pv
        l = self.pvl
        pv = zeros(l, dtype=float)
        tm = self.tm

        tc = 0
        steps = 0
        res[0, :] = ini
        a0 = 1

        print("rep = " + str(reps))
        print("ini = " + str(ini))
        print("seq_ini = " + str(ini_sequence))
        print("------")
        for tim in range(1, tmax):
            # time_death_step = []
            # time_mutation_step = []

            while tc < tim:
                for i in range(l):
                    pv[i] = pvi[i](ini)
                # pv = abs(array([eq() for eq in pvi]))# #propensity vector
                a0 = pv.sum()  # sum of all transition probabilities
                if a0 == 0:
                    break
                if np.sum(ini[1:]) == self.exit:
                    break

                tau = (-1 / a0) * log(random())
                event = multinomial(1, pv / a0)  # event which will happen on this iteration
                transition = tm[:, event.nonzero()[0][0]]

                tc += tau
                steps += 1

                # Extend sequence array
                new_seq_index = np.nonzero(ini_sequence)[0][-1] + 1

                if (ini_sequence.shape[0] - new_seq_index) < 50:
                    ini_sequence, sequence_time = self.extend_array(ini_sequence, sequence_time, tmax)
                    # self.number_sequences += 500
                    #
                    # new_ini_sequence = np.zeros((self.number_sequences, self.nvars + 1), dtype=int)
                    # new_ini_sequence[:ini_sequence.shape[0], :ini_sequence.shape[1]] = ini_sequence
                    # ini_sequence = new_ini_sequence
                    #
                    # new_sequence_time = np.zeros((tmax, self.number_sequences, self.nvars + 1), dtype=int)
                    # new_sequence_time[:sequence_time.shape[0], :sequence_time.shape[1], :sequence_time.shape[2]] = sequence_time
                    # sequence_time = new_sequence_time

                # Replication event
                if np.sum(transition) > 0:
                    ini_sequence = self.replication_event(ini_sequence, transition, tc)
                    # seq_bin = np.where(transition == 1)[0][0]
                    # seq_indices = np.nonzero(ini_sequence[:, 1:][:, seq_bin])[0]
                    # chosen_seq = np.random.choice(seq_indices)
                    #
                    # new_seq_index = np.nonzero(ini_sequence)[0][-1] + 1
                    #
                    # # Have to add one when indexing to not confuse with 0th entry
                    # ini_sequence[new_seq_index][0] = chosen_seq + 1
                    # ini_sequence[:, 1:][new_seq_index] = transition

                # Mutation event
                else:
                    ini_sequence = self.mutation_event(ini_sequence, transition, tc)
                    # seq_bin = np.where(transition == -1)[0][0]
                    # seq_indices = np.nonzero(ini_sequence[:, 1:][:, seq_bin])[0]
                    # chosen_seq = np.random.choice(seq_indices)
                    # ini_sequence[:, 1:][chosen_seq] += transition

                ini = np.sum(ini_sequence, axis=0)[self.start_index:]

            # self.time_death.append(np.mean(time_death_step))
            # self.time_mutation.append(np.mean(time_mutation_step))

            sequence_time[tim, :] = ini_sequence
            res[tim, :] = ini
            if a0 == 0:
                break
            if np.sum(ini[1:]) == self.exit:
                break

        end = np.nonzero(np.sum(res, axis=1))[0][-1] + 1

        return tc, sequence_time[:end]


class ModelBoostTrajectories(ModelTrajectories):

    def __init__(self, prime_sequences, p_ini, vnames, tmat, propensity, n_exit=None):
        ModelTrajectories.__init__(self, p_ini, vnames, tmat, propensity, n_exit=n_exit)

        # Sequences generated from priming
        self.prime_sequences = prime_sequences
        self.number_sequences = self.prime_sequences.shape[1]

        print("Calculations with ModelBoostTrajectories class.")

    def run(self, method='SSA', tmax=10, reps=1):
        tvec = range(1, tmax)

        if method == 'SSA':
            res = zeros((tmax, self.nvars), dtype=float)
            tc, sequence_time = self.GSSA(res, tmax=tmax, reps=reps)

            if self.prime_sequences.shape[1] != sequence_time.shape[1]:
                new_prime_sequences = np.zeros((self.prime_sequences.shape[0], sequence_time.shape[1],
                                                self.prime_sequences.shape[2]), dtype=float)
                new_prime_sequences[:self.prime_sequences.shape[0], :self.prime_sequences.shape[1],
                :self.prime_sequences.shape[2]] = self.prime_sequences
                self.prime_sequences = new_prime_sequences

            prime_boost_sequence_time = np.vstack((self.prime_sequences, sequence_time))
            pickle_trajectories(prime_boost_sequence_time, reps)
            write_results(res, reps)
        elif method == 'SSAct':
            pass

        self.time = tvec
        np.savetxt("time", self.time, fmt='%f')

        # self.steps = steps
        # np.savetxt("steps", [steps], fmt='%f')

    def initialize_sequence_array(self):
        dynamic_prime_sequences = np.copy(self.prime_sequences[-1])
        new_indices = []
        for i in range(self.ntot):
            n_i_exit = np.sum(dynamic_prime_sequences, axis=0, dtype=np.float)[2:]
            print("n_i = " + str(n_i_exit))
            p_ini = n_i_exit / np.sum(n_i_exit)

            print("p_ini = " + str(p_ini))
            events = multinomial(1, p_ini)
            # print("events = " + str(events))
            bin_index = events.nonzero()[0][0]

            seq_indices = [j for j in range(len(dynamic_prime_sequences)) if
                           dynamic_prime_sequences[j][2:][bin_index] == 1]

            chosen_seq = np.random.choice(seq_indices)
            new_indices.append(chosen_seq)

            dynamic_prime_sequences[chosen_seq].fill(0)

        remaining_indices = np.array([])
        for j in range(1, dynamic_prime_sequences.shape[1] - 1):
            alive_indices = np.array([index for index in range(len(dynamic_prime_sequences)) if
                                      dynamic_prime_sequences[:, 1:][index][j] == 1])
            remaining_indices = np.append(remaining_indices, alive_indices)

        boost_sequence_array = np.copy(self.prime_sequences[-1])

        death_entry = np.zeros(self.nvars, dtype=int)
        death_entry[0] = 1
        for i in remaining_indices:
            boost_sequence_array[int(i)][1:] = death_entry

        return boost_sequence_array


# Need to remove time until access. Makes array way too huge.
class ModelMFPTTrajectories(ModelTrajectories):

    def __init__(self, p_ini, vnames, tmat, propensity, n_exit=None):
        ModelTrajectories.__init__(self, p_ini, vnames, tmat, propensity, n_exit=n_exit)
        self.start_index = 2
        # self.time_start_index = 2
        self.sequence_dimensions = self.nvars + self.start_index

    def run_GSSA(self, tmax, i):
        res = zeros((tmax, self.nvars), dtype=float)

        tc, sequence_time = self.GSSA(res, tmax=tmax, reps=i)
        pickle_trajectories(sequence_time[-1], i)
        write_results(res, i)

        return tc

    def initialize_sequence_array(self):

        new_ini = self.set_ini()
        sequence_array = np.zeros((self.number_sequences, self.sequence_dimensions), dtype=float)
        index = 0
        for i in range(self.start_index + 1, len(new_ini)):
            next_index = int(new_ini[i]) + index
            sequence_array[index:next_index][:, i] = 1

            # Setting 0th entry to index of sequence in seq_array
            for j in range(index, next_index):
                sequence_array[j][0] = j

            # Setting 1st entry to starting bin of initial sequence
            sequence_array[index:next_index][:, 1] = i - self.start_index

            # # Recording time of access to bin of birth
            # sequence_array[index:next_index][:, i - (self.start_index - self.time_start_index)] = 1e-07

            index += int(new_ini[i])

        return sequence_array

    def replication_event(self, ini_sequence, transition, tc):
        seq_bin = np.where(transition == 1)[0][0]
        seq_indices = np.nonzero(ini_sequence[:, self.start_index:][:, seq_bin])[0]
        chosen_seq = np.random.choice(seq_indices)
        new_seq_index = np.nonzero(ini_sequence)[0][-1] + 1

        # Need to propagate information about index of initial sequence to new sequence array[0]
        # Need to propagate information about bin of initial sequence to new sequence array[1]
        # Need to propagate all information regarding time of access to bins i

        ini_sequence[new_seq_index][:self.start_index] = ini_sequence[chosen_seq][:self.start_index]

        # Setting array for new sequence equal to transition event

        ini_sequence[:, self.start_index:][new_seq_index] = transition

        return ini_sequence

    def mutation_event(self, ini_sequence, transition, tc):

        '''Can easily compute tau_m(x[4] - x[i]) via ini_sequence[chosen_seq][4] - ini_sequence[chosen_seq][i].
        Can also compute tau_d(x[i]) via ini_sequence[chosen_seq][0] - ini_sequence[chosen_seq][i]
        assuming sequence has died.'''

        seq_bin = np.where(transition == -1)[0][0]
        next_seq_bin = np.where(transition == 1)[0][0]

        seq_indices = np.nonzero(ini_sequence[:, self.start_index:][:, seq_bin])[0]
        chosen_seq = np.random.choice(seq_indices)

        # # Lethal mutation: Recording t_{d} in 2nd entry = 0 (next_seq_bin) + 2 (time_start_index) of sequence array
        # # (indices 0 and 1 are for sequence and bin index)
        # if next_seq_bin == 0:
        #     ini_sequence[chosen_seq][next_seq_bin + self.time_start_index] = tc
        #
        # else:
        #     # Checking if time in the bin is 0.0; if not, bin has already been accessed by lineage
        #     # If next_seq_bin + self.time_start_index has already been accessed before, time is not recorded.
        #     if ini_sequence[chosen_seq][next_seq_bin + self.time_start_index] == 0.0:
        #         ini_sequence[chosen_seq][next_seq_bin + self.time_start_index] = tc

        # Adding transition event to sequence array

        ini_sequence[:, self.start_index:][chosen_seq] += transition

        return ini_sequence
