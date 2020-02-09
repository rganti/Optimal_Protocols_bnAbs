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

    # def run_model(self, reps=50):
    #     self.define_tmat()
    #
    #     if self.trajectories:
    #         M = ModelMFPTTrajectories(p_ini=self.p_ini, vnames=self.vars, tmat=self.tmat, propensity=self.prop)
    #     else:
    #         M = Model(p_ini=self.p_ini, vnames=self.vars, tmat=self.tmat, propensity=self.prop, n_stop=self.n_stop)
    #
    #     t0 = time.time()
    #     M.run(tmax=1000, reps=reps)
    #     print('total time: ', time.time() - t0)
    #     t, series, steps = M.getStats()
    #     print(steps, 'steps')


# NEED to hardcode new reactions into self.prop

class BnabFiniteSizeEffects(BnabGillespie):
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


