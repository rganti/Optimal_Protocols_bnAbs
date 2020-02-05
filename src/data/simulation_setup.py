import time

import numpy as np

from src.data.gillespie_models import ModelMFPTTrajectories, Model
from src.data.ssc_setup import BnabModel, SSCLaunch
from src.data_process.simulation_post_process import GillespieGCExit


class BnabGillespie(object):

    def __init__(self, p_ini, fraction=7.0/8.0, sigma=0.3, death_rate=0.1, trajectories=False):

        self.p_ini = p_ini
        self.len_ini = len(self.p_ini) + 1
        self.vars = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']

        self.bnab = BnabModel(self.p_ini, fraction=fraction, sigma=sigma, death_rate=death_rate)
        self.trajectories = trajectories
        self.mu_ij = self.bnab.define_mu_ij_fixed()

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

        self.tmat = np.zeros((self.len_ini, 26), dtype=int)

    def define_tmat(self):
        # Create death reactions and N1 replication
        for i in range(0, 8):
            rxn = np.zeros(self.len_ini, dtype=int)
            if i == 7: # N1 replication reaction
                rxn[1] = 1
            else: # Death reactions
                rxn[0] = 1
                rxn[i + 1] = -1
            self.tmat[:, i] = rxn

        # Create remaining reactions
        for i in range(8, 26):
            rxn = np.zeros(self.len_ini, dtype=int)

            if (i % 3) == 0: # replication reaction
                rxn[i / 3 - 1] = 1
            elif (i % 3) == 1: # transition reaction to lower state
                rxn[i / 3 - 1] = -1
                rxn[i / 3 - 2] = 1
            elif (i % 3) == 2: # transition reaction to higher state
                rxn[i / 3 - 1] = -1
                rxn[i / 3] = 1

            self.tmat[:, i] = rxn

    def run_model(self, reps=50):
        self.define_tmat()

        if self.trajectories:
            M = ModelMFPTTrajectories(p_ini=self.p_ini, vnames=self.vars, tmat=self.tmat, propensity=self.prop)
        else:
            M = Model(p_ini=self.p_ini, vnames=self.vars, tmat=self.tmat, propensity=self.prop)

        t0 = time.time()
        M.run(tmax=1000, reps=reps)
        print('total time: ', time.time() - t0)
        t, series, steps = M.getStats()
        print(steps, 'steps')


class Simulation(object):

    def __init__(self, p_n, sigma=0.25, fraction=7.0/8.0, death_rate=0.1, trajectories=False):
        self.p_n = p_n
        self.gillespie_bnab = BnabGillespie(self.p_n, fraction=fraction, sigma=sigma, death_rate=death_rate,
                                            trajectories=trajectories)

        np.savetxt("event_prob", p_n, fmt='%f')
        np.savetxt("fitness", self.gillespie_bnab.bnab.fitness.f, fmt='%f')
        np.savetxt("sigma", [self.gillespie_bnab.bnab.fitness.sigma], fmt='%f')
        np.savetxt("death_rate", [self.gillespie_bnab.bnab.mu_i0], fmt='%f')
        np.savetxt("fraction", [self.gillespie_bnab.bnab.fraction], fmt='%f')

    def run(self, reps=200):
        self.gillespie_bnab.run_model(reps=reps)

        print("Starting post-processing...")
        post_process = GillespieGCExit()
        post_process.populate_pn(index=1)

        print("Printing SSC script w/ parameters")
        bnab_ssc = SSCLaunch(self.p_n, sigma=self.gillespie_bnab.bnab.fitness.sigma,
                             fraction=self.gillespie_bnab.bnab.fraction,
                             death_rate=self.gillespie_bnab.bnab.mu_i0)
        bnab_ssc.generate_ssc_script("bnab_fitness")