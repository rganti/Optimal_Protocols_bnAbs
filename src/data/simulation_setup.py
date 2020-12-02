import numpy as np

from src.data.ssc_setup import BnabModel


class BnabGillespie(object):

    def __init__(self, p_ini, parameters):

        self.p_ini = p_ini
        self.len_ini = len(self.p_ini) + 1
        self.vars = ['N{0}'.format(i) for i in range(self.len_ini)]

        self.bnab = BnabModel(p_ini, parameters)
        self.mu_ij = self.bnab.define_mu_ij()

        # First 7 are death reactions, next 2 are N1,
        self.prop = (lambda ini:self.mu_ij['mu{0}{1}'.format(1, 0)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 0)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 0)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 0)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 0)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 0)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(7, 0)] * ini[7],

                     lambda ini:self.mu_ij['f{0}'.format(1)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(1, 2)] * ini[1],

                     lambda ini:self.mu_ij['f{0}'.format(2)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 1)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 3)] * ini[2],

                     lambda ini:self.mu_ij['f{0}'.format(3)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 2)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 4)] * ini[3],

                     lambda ini:self.mu_ij['f{0}'.format(4)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 3)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 5)] * ini[4],

                     lambda ini:self.mu_ij['f{0}'.format(5)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 4)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 6)] * ini[5],

                     lambda ini:self.mu_ij['f{0}'.format(6)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 5)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 7)] * ini[6],

                     lambda ini:self.mu_ij['f{0}'.format(7)] * ini[7],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(7, 6)] * ini[7])

        self.tmat = np.zeros((self.len_ini, len(self.prop)), dtype=int)
        self.count = 0

    def replication(self, index):
        rxn = np.zeros(self.len_ini, dtype=int)
        rxn[index] = 1
        self.tmat[:, self.count] = rxn
        self.count += 1

    def transition_to_lower(self, index):
        rxn = np.zeros(self.len_ini, dtype=int)
        rxn[index] = -1
        rxn[index - 1] = 1
        self.tmat[:, self.count] = rxn
        self.count += 1

    def transition_to_higher(self, index):
        rxn = np.zeros(self.len_ini, dtype=int)
        rxn[index] = -1
        rxn[index + 1] = 1
        self.tmat[:, self.count] = rxn
        self.count += 1

    def define_lower_edge(self, index):
        self.replication(index)
        self.transition_to_higher(index)

    def define_middle(self, index):
        self.replication(index)
        self.transition_to_lower(index)
        self.transition_to_higher(index)

    def define_upper_edge(self, index):
        self.replication(index)
        self.transition_to_lower(index)

    def define_tmat(self):
        # Create death reactions
        for i in range(0, len(self.p_ini)):
            rxn = np.zeros(self.len_ini, dtype=int)
            rxn[0] = 1
            rxn[i + 1] = -1
            self.tmat[:, self.count] = rxn

            self.count += 1

        self.define_lower_edge(1)

        for i in range(2, len(self.p_ini)):
            self.define_middle(i)

        self.define_upper_edge(len(self.p_ini))


# NEED to hardcode new reactions into self.prop
class BnabFiniteSizeEffects9(BnabGillespie):
    def __init__(self, p_ini, parameters):
        BnabGillespie.__init__(self, p_ini, parameters)

        # First 9 reactions are death reactions
        self.prop = (lambda ini:self.mu_ij['mu{0}{1}'.format(1, 0)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 0)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 0)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 0)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 0)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 0)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(7, 0)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 0)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 0)] * ini[9],

                     # Edge state 1: can only hop to the right to 2
                     lambda ini:self.mu_ij['f{0}'.format(1)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(1, 2)] * ini[1],

                     lambda ini:self.mu_ij['f{0}'.format(2)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 1)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 3)] * ini[2],

                     lambda ini:self.mu_ij['f{0}'.format(3)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 2)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 4)] * ini[3],

                     lambda ini:self.mu_ij['f{0}'.format(4)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 3)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 5)] * ini[4],

                     # Middle bnAb state 5
                     lambda ini:self.mu_ij['f{0}'.format(5)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 4)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 6)] * ini[5],

                     lambda ini:self.mu_ij['f{0}'.format(6)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 5)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 7)] * ini[6],

                     lambda ini: self.mu_ij['f{0}'.format(7)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 6)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 8)] * ini[7],

                     lambda ini: self.mu_ij['f{0}'.format(8)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 7)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 9)] * ini[8],

                     # Edge state 9: can only hop to left to 8
                     lambda ini:self.mu_ij['f{0}'.format(9)] * ini[9],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(9, 8)] * ini[9])

        self.tmat = np.zeros((self.len_ini, len(self.prop)), dtype=int)


class BnabFiniteSizeEffects11(BnabGillespie):
    def __init__(self, p_ini, parameters):
        BnabGillespie.__init__(self, p_ini, parameters)

        # First 11 reactions are death reactions
        self.prop = (lambda ini:self.mu_ij['mu{0}{1}'.format(1, 0)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 0)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 0)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 0)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 0)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 0)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(7, 0)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 0)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 0)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 0)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(11, 0)] * ini[11],

                     # Edge state 1: can only hop to the right to 2
                     lambda ini:self.mu_ij['f{0}'.format(1)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(1, 2)] * ini[1],

                     lambda ini:self.mu_ij['f{0}'.format(2)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 1)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 3)] * ini[2],

                     lambda ini:self.mu_ij['f{0}'.format(3)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 2)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 4)] * ini[3],

                     lambda ini:self.mu_ij['f{0}'.format(4)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 3)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 5)] * ini[4],

                     lambda ini:self.mu_ij['f{0}'.format(5)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 4)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 6)] * ini[5],

                     # Middle bnAb state 6
                     lambda ini:self.mu_ij['f{0}'.format(6)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 5)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 7)] * ini[6],

                     lambda ini: self.mu_ij['f{0}'.format(7)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 6)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 8)] * ini[7],

                     lambda ini: self.mu_ij['f{0}'.format(8)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 7)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 9)] * ini[8],

                     lambda ini: self.mu_ij['f{0}'.format(9)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 8)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 10)] * ini[9],

                     lambda ini: self.mu_ij['f{0}'.format(10)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 9)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 11)] * ini[10],

                     # Edge state 11: can only hop to left to 10
                     lambda ini:self.mu_ij['f{0}'.format(11)] * ini[11],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(11, 10)] * ini[11])

        self.tmat = np.zeros((self.len_ini, len(self.prop)), dtype=int)


class BnabFiniteSizeEffects15(BnabGillespie):
    def __init__(self, p_ini, parameters):
        BnabGillespie.__init__(self, p_ini, parameters)

        # First 15 reactions are death reactions
        self.prop = (lambda ini:self.mu_ij['mu{0}{1}'.format(1, 0)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 0)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 0)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 0)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 0)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 0)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(7, 0)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 0)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 0)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 0)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(11, 0)] * ini[11],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(12, 0)] * ini[12],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(13, 0)] * ini[13],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(14, 0)] * ini[14],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(15, 0)] * ini[15],

                     # Edge state 1: can only hop to the right to 2
                     lambda ini:self.mu_ij['f{0}'.format(1)] * ini[1],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(1, 2)] * ini[1],

                     lambda ini:self.mu_ij['f{0}'.format(2)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 1)] * ini[2],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(2, 3)] * ini[2],

                     lambda ini:self.mu_ij['f{0}'.format(3)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 2)] * ini[3],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(3, 4)] * ini[3],

                     lambda ini:self.mu_ij['f{0}'.format(4)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 3)] * ini[4],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(4, 5)] * ini[4],

                     lambda ini:self.mu_ij['f{0}'.format(5)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 4)] * ini[5],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(5, 6)] * ini[5],

                     lambda ini:self.mu_ij['f{0}'.format(6)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 5)] * ini[6],
                     lambda ini:self.mu_ij['mu{0}{1}'.format(6, 7)] * ini[6],

                     lambda ini: self.mu_ij['f{0}'.format(7)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 6)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 8)] * ini[7],

                     # Middle bnAb state 8
                     lambda ini: self.mu_ij['f{0}'.format(8)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 7)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 9)] * ini[8],

                     lambda ini: self.mu_ij['f{0}'.format(9)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 8)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 10)] * ini[9],

                     lambda ini: self.mu_ij['f{0}'.format(10)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 9)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 11)] * ini[10],

                     lambda ini: self.mu_ij['f{0}'.format(11)] * ini[11],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(11, 10)] * ini[11],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(11, 12)] * ini[11],

                     lambda ini: self.mu_ij['f{0}'.format(12)] * ini[12],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(12, 11)] * ini[12],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(12, 13)] * ini[12],

                     lambda ini: self.mu_ij['f{0}'.format(13)] * ini[13],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(13, 12)] * ini[13],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(13, 14)] * ini[13],

                     lambda ini: self.mu_ij['f{0}'.format(14)] * ini[14],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(14, 13)] * ini[14],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(14, 15)] * ini[14],

                     # Edge state 15: can only hop to left to 14
                     lambda ini: self.mu_ij['f{0}'.format(15)] * ini[15],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(15, 14)] * ini[15])

        self.tmat = np.zeros((self.len_ini, len(self.prop)), dtype=int)


class BnabFiniteSizeEffects31(BnabGillespie):
    def __init__(self, p_ini, parameters):
        BnabGillespie.__init__(self, p_ini, parameters)

        # First 31 reactions are death reactions
        self.prop = (lambda ini: self.mu_ij['mu{0}{1}'.format(1, 0)] * ini[1],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(2, 0)] * ini[2],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(3, 0)] * ini[3],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(4, 0)] * ini[4],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(5, 0)] * ini[5],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(6, 0)] * ini[6],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 0)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 0)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 0)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 0)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(11, 0)] * ini[11],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(12, 0)] * ini[12],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(13, 0)] * ini[13],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(14, 0)] * ini[14],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(15, 0)] * ini[15],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(16, 0)] * ini[16],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(17, 0)] * ini[17],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(18, 0)] * ini[18],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(19, 0)] * ini[19],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(20, 0)] * ini[20],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(21, 0)] * ini[21],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(22, 0)] * ini[22],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(23, 0)] * ini[23],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(24, 0)] * ini[24],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(25, 0)] * ini[25],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(26, 0)] * ini[26],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(27, 0)] * ini[27],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(28, 0)] * ini[28],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(29, 0)] * ini[29],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(30, 0)] * ini[30],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(31, 0)] * ini[31],

                     # Edge state 1: can only hop to the right to 2
                     lambda ini: self.mu_ij['f{0}'.format(1)] * ini[1],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(1, 2)] * ini[1],

                     lambda ini: self.mu_ij['f{0}'.format(2)] * ini[2],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(2, 1)] * ini[2],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(2, 3)] * ini[2],

                     lambda ini: self.mu_ij['f{0}'.format(3)] * ini[3],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(3, 2)] * ini[3],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(3, 4)] * ini[3],

                     lambda ini: self.mu_ij['f{0}'.format(4)] * ini[4],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(4, 3)] * ini[4],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(4, 5)] * ini[4],

                     lambda ini: self.mu_ij['f{0}'.format(5)] * ini[5],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(5, 4)] * ini[5],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(5, 6)] * ini[5],

                     lambda ini: self.mu_ij['f{0}'.format(6)] * ini[6],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(6, 5)] * ini[6],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(6, 7)] * ini[6],

                     lambda ini: self.mu_ij['f{0}'.format(7)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 6)] * ini[7],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(7, 8)] * ini[7],

                     lambda ini: self.mu_ij['f{0}'.format(8)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 7)] * ini[8],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(8, 9)] * ini[8],

                     lambda ini: self.mu_ij['f{0}'.format(9)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 8)] * ini[9],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(9, 10)] * ini[9],

                     lambda ini: self.mu_ij['f{0}'.format(10)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 9)] * ini[10],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(10, 11)] * ini[10],

                     lambda ini: self.mu_ij['f{0}'.format(11)] * ini[11],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(11, 10)] * ini[11],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(11, 12)] * ini[11],

                     lambda ini: self.mu_ij['f{0}'.format(12)] * ini[12],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(12, 11)] * ini[12],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(12, 13)] * ini[12],

                     lambda ini: self.mu_ij['f{0}'.format(13)] * ini[13],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(13, 12)] * ini[13],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(13, 14)] * ini[13],

                     lambda ini: self.mu_ij['f{0}'.format(14)] * ini[14],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(14, 13)] * ini[14],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(14, 15)] * ini[14],

                     lambda ini: self.mu_ij['f{0}'.format(15)] * ini[15],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(15, 14)] * ini[15],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(15, 16)] * ini[15],

                     # Middle bnAb state 16
                     lambda ini: self.mu_ij['f{0}'.format(16)] * ini[16],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(16, 15)] * ini[16],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(16, 17)] * ini[16],

                     lambda ini: self.mu_ij['f{0}'.format(17)] * ini[17],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(17, 16)] * ini[17],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(17, 18)] * ini[17],

                     lambda ini: self.mu_ij['f{0}'.format(18)] * ini[18],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(18, 17)] * ini[18],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(18, 19)] * ini[18],

                     lambda ini: self.mu_ij['f{0}'.format(19)] * ini[19],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(19, 18)] * ini[19],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(19, 20)] * ini[19],

                     lambda ini: self.mu_ij['f{0}'.format(20)] * ini[20],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(20, 19)] * ini[20],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(20, 21)] * ini[20],

                     lambda ini: self.mu_ij['f{0}'.format(21)] * ini[21],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(21, 20)] * ini[21],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(21, 22)] * ini[21],

                     lambda ini: self.mu_ij['f{0}'.format(22)] * ini[22],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(22, 21)] * ini[22],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(22, 23)] * ini[22],

                     lambda ini: self.mu_ij['f{0}'.format(23)] * ini[23],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(23, 22)] * ini[23],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(23, 24)] * ini[23],

                     lambda ini: self.mu_ij['f{0}'.format(24)] * ini[24],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(24, 23)] * ini[24],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(24, 25)] * ini[24],

                     lambda ini: self.mu_ij['f{0}'.format(25)] * ini[25],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(25, 24)] * ini[25],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(25, 26)] * ini[25],

                     lambda ini: self.mu_ij['f{0}'.format(26)] * ini[26],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(26, 25)] * ini[26],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(26, 27)] * ini[26],

                     lambda ini: self.mu_ij['f{0}'.format(27)] * ini[27],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(27, 26)] * ini[27],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(27, 28)] * ini[27],

                     lambda ini: self.mu_ij['f{0}'.format(28)] * ini[28],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(28, 27)] * ini[28],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(28, 29)] * ini[28],

                     lambda ini: self.mu_ij['f{0}'.format(29)] * ini[29],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(29, 28)] * ini[29],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(29, 30)] * ini[29],

                     lambda ini: self.mu_ij['f{0}'.format(30)] * ini[30],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(30, 29)] * ini[30],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(30, 31)] * ini[30],

                     # Edge state 31: can only hop to left to 30
                     lambda ini: self.mu_ij['f{0}'.format(31)] * ini[31],
                     lambda ini: self.mu_ij['mu{0}{1}'.format(31, 30)] * ini[31])

        self.tmat = np.zeros((self.len_ini, len(self.prop)), dtype=int)
