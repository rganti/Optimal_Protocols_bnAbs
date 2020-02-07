import os

import numpy as np

from src.data.gillespie_models import ModelMFPTTrajectories, ModelIndividualGC
from src.data.simulation_setup import Simulation, BnabGillespie, BnabFiniteSizeEffects
from src.general.directory_handling import make_and_cd
from src.data_process.simulation_post_process import print_info, GillespieGCExit
from src.visualization.visualize_fitness import Injection


class Procedure(object):

    def __init__(self, sigma_1=1.0, death_rate=0.1, fraction=7.0/8.0):
        self.n_initial = [20.0, 5.0, 0.0, 0.0, 0.0, 5.0, 20.0]
        self.p_ini = self.n_initial / np.sum(self.n_initial)
        self.fraction = fraction
        self.death_rate = death_rate
        self.home = os.getcwd()
        self.sigma = [sigma_1, 0.4]
        self.total_bnabs = 0.0

    def run_cocktail(self):
        make_and_cd("Cocktail")
        bnab_cocktail = Simulation(self.p_ini, sigma=self.sigma[1], fraction=self.fraction,
                                   death_rate=self.death_rate)
        bnab_cocktail.run()
        os.chdir(self.home)

    def get_i1_values(self):
        injection_1 = Injection(sigma=self.sigma[0])
        delta_s1 = injection_1.delta_h

        return injection_1.f, delta_s1

    def next_injection(self, index, sigma):
        make_and_cd("injection_{0}_sig_{1}".format(index, round(sigma, 2)))
        n_ave = np.loadtxt("../n_ave")
        p_n = n_ave / np.sum(n_ave)
        np.savetxt("event_prob", p_n, fmt='%f')
        success_exit = np.loadtxt("../successful_exit")
        reps = len(success_exit)

        bnab = Simulation(p_n, sigma=sigma, fraction=self.fraction, death_rate=self.death_rate)
        bnab.run(reps=int(reps))

    def run_sequential(self):

        # Injection 1
        # make_and_cd("Sequential")
        make_and_cd("Sigma_{0}".format(round(self.sigma[0], 2)))
        sigma_1_directory = os.getcwd()

        bnab_sequential = Simulation(self.p_ini, sigma=self.sigma[0], fraction=self.fraction,
                                     death_rate=self.death_rate)
        bnab_sequential.run(reps=50)

        # seq_dir = os.getcwd()

        # Injection 2
        self.next_injection(index=2, sigma=self.sigma[1])
        # os.chdir(seq_dir)

        n4 = np.loadtxt("n4")
        self.total_bnabs = np.sum(n4)

        os.chdir(sigma_1_directory)

        np.savetxt("total_bnabs", [self.total_bnabs], fmt='%f')

    def run_protocols(self):
        self.run_sequential()


class ProcedureDelS1S2(Procedure):

    def __init__(self, sigma_1=10.0, death_rate=0.1, fraction=7.0/8.0, trajectories=False):
        Procedure.__init__(self, sigma_1=sigma_1, death_rate=death_rate, fraction=fraction)
        self.sigma = np.logspace(np.log10(sigma_1), -1.0, num=10)
        self.total_bnabs = []
        self.num_reps = 100
        self.trajectories = trajectories
        self.n_initial = [20.0, 5.0, 0.0, 0.0, 0.0, 5.0, 20.0]
        # self.n_initial = [20.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 20.0]
        self.p_ini = self.n_initial / np.sum(self.n_initial)
        self.n_stop = 200

    def run_prime(self):
        bnab_sequential = Simulation(self.p_ini, sigma=self.sigma[0], fraction=self.fraction,
                                     death_rate=self.death_rate, trajectories=self.trajectories,
                                     n_stop=self.n_stop)
        bnab_sequential.run(reps=self.num_reps)

    def run_sequential(self):

        self.run_prime()

        sigma_1_directory = os.getcwd()
        np.savetxt("sigma2_range", self.sigma[1:], fmt='%f')
        success_exit = np.loadtxt("successful_exit")

        if len(success_exit) > 0:
            # Looping over all possible sigma for injection 2
            for j in range(1, len(self.sigma)):

                # Injection 2
                self.next_injection(index=2, sigma=self.sigma[j])

                n_bnabs = np.loadtxt("n{0}".format(int((len(self.p_ini) + 1)/2)))
                self.total_bnabs.append(np.sum(n_bnabs))

                # if self.build_tree:
                #     tree_mp()

                os.chdir(sigma_1_directory)

            np.savetxt("total_bnabs", self.total_bnabs, fmt='%f')

            print_info(self.sigma)

        else:
            np.savetxt("total_bnabs", np.zeros(len(self.sigma[1:])), fmt='%f')


class ProcedureIndividualRestart(ProcedureDelS1S2):

    def __init__(self, sigma_1=10.0, death_rate=0.1, fraction=7.0/8.0, trajectories=False):
        ProcedureDelS1S2.__init__(self, sigma_1=sigma_1, death_rate=death_rate, fraction=fraction,
                                  trajectories=trajectories)

    def next_injection(self, index, sigma):
        make_and_cd("injection_{0}_sig_{1}".format(index, round(sigma, 2)))
        success_exit = np.loadtxt("../successful_exit")

        if len(success_exit.shape) == 1:
            success_exit = np.array([success_exit])

        count = 0
        # tc = []

        for entry in success_exit:
            traj_index = int(entry[0])

            # NOTE Second entry must contain identity of exit index.
            exit_index = int(entry[1])

            n_i = np.loadtxt("../hashed_traj_{0}".format(traj_index))
            n_i_exit = n_i[exit_index]

            p_ini = n_i_exit[1:]/np.sum(n_i_exit[1:])

            if len(self.p_ini) > 7:
                gillespie_bnab = BnabFiniteSizeEffects(p_ini, fraction=self.fraction, sigma=sigma,
                                                            death_rate=self.death_rate,
                                                            trajectories=self.trajectories, n_stop=self.n_stop)
            else:
                gillespie_bnab = BnabGillespie(p_ini, fraction=self.fraction, sigma=sigma,
                                               death_rate=self.death_rate, trajectories=self.trajectories,
                                               n_stop=self.n_stop)

            gillespie_bnab.define_tmat()

            if self.trajectories:
                M = ModelMFPTTrajectories(p_ini=p_ini, vnames=gillespie_bnab.vars, tmat=gillespie_bnab.tmat,
                                          propensity=gillespie_bnab.prop, n_exit=n_i_exit[1:])

            else:
                M = ModelIndividualGC(n_exit=n_i_exit[1:], p_ini=p_ini, vnames=gillespie_bnab.vars,
                                      tmat=gillespie_bnab.tmat, propensity=gillespie_bnab.prop,
                                      n_stop=gillespie_bnab.n_stop)

            M.run(tmax=1000, reps=count)
            # tc.append(M.t_exit)

            count += 1

        # np.savetxt("tc_gc_exit", tc, fmt='%f')
        np.savetxt("fitness", gillespie_bnab.bnab.fitness.f, fmt='%f')
        np.savetxt("sigma", [gillespie_bnab.bnab.fitness.sigma], fmt='%f')
        np.savetxt("death_rate", [gillespie_bnab.bnab.mu_i0], fmt='%f')
        np.savetxt("mutation_rate", [gillespie_bnab.bnab.mu_ij], fmt='%f')
        np.savetxt("fraction", [gillespie_bnab.bnab.fraction], fmt='%f')

        print("Starting post-processing...")
        post_process = GillespieGCExit(n_exit=self.n_stop, num_odes=gillespie_bnab.len_ini)
        post_process.populate_pn(index=1)
