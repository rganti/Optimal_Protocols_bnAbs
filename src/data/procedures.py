import os
import time

import numpy as np

from src.data.gillespie_models import ModelMFPTTrajectories, ModelIndividualGC, Model
from src.data.simulation_setup import BnabGillespie, BnabFiniteSizeEffects
from src.data.ssc_setup import SSCLaunch
from src.general.directory_handling import make_and_cd
from src.data_process.simulation_post_process import print_info, GillespieGCExit
from src.visualization.visualize_fitness import Injection


class Procedure(object):

    def __init__(self, parameters):
        self.n_initial = [20.0, 5.0, 0.0, 0.0, 0.0, 5.0, 20.0]
        self.p_ini = self.n_initial / np.sum(self.n_initial)

        self.parameters = parameters

        self.home = os.getcwd()
        self.sigma = [parameters['sigma'], 0.4]
        self.total_bnabs = 0.0

    def run_cocktail(self):
        make_and_cd("Cocktail")
        bnab_cocktail = Simulation(self.p_ini, self.parameters)
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

        bnab = Simulation(p_n, self.parameters)
        bnab.run(reps=int(reps))

    def run_sequential(self):

        # Injection 1
        # make_and_cd("Sequential")
        make_and_cd("Sigma_{0}".format(round(self.sigma[0], 2)))
        sigma_1_directory = os.getcwd()

        bnab_sequential = Simulation(self.p_ini, self.parameters)
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


class Simulation(object):

    def __init__(self, p_ini, parameters, trajectories=False):
        self.p_ini = p_ini
        self.trajectories = trajectories
        self.parameters = parameters

        if len(self.p_ini) == 9:
            self.gillespie_bnab = BnabFiniteSizeEffects(p_ini, parameters)
        else:
            self.gillespie_bnab = BnabGillespie(p_ini, parameters)

        self.gillespie_bnab.define_tmat()

    def run(self, reps=200):

        if self.trajectories:
            M = ModelMFPTTrajectories(p_ini=self.p_ini, vnames=self.gillespie_bnab.vars,
                                      tmat=self.gillespie_bnab.tmat, propensity=self.gillespie_bnab.prop)
        else:
            M = Model(p_ini=self.p_ini, vnames=self.gillespie_bnab.vars, tmat=self.gillespie_bnab.tmat,
                      propensity=self.gillespie_bnab.prop, n_stop=self.parameters['n_stop'])

        np.savetxt("event_prob", self.p_ini, fmt='%f')
        np.savetxt("fitness", self.gillespie_bnab.bnab.fitness.f, fmt='%f')
        np.savetxt("sigma", [self.gillespie_bnab.bnab.fitness.sigma], fmt='%f')
        np.savetxt("death_rate", [self.gillespie_bnab.bnab.mu_i0], fmt='%f')
        np.savetxt("mutation_rate", [self.gillespie_bnab.bnab.mu_ij], fmt='%f')
        np.savetxt("fraction", [self.gillespie_bnab.bnab.fraction], fmt='%f')

        t0 = time.time()
        M.run(tmax=1000, reps=reps)
        print('total time: ', time.time() - t0)

        print("Starting post-processing...")
        post_process = GillespieGCExit(n_exit=self.parameters['n_stop'], num_odes=self.gillespie_bnab.len_ini)
        post_process.populate_pn(index=1)

        print("Printing SSC script w/ parameters")
        bnab_ssc = SSCLaunch(self.p_ini, self.parameters)
        bnab_ssc.generate_ssc_script("bnab_fitness")


class ProcedureDelS1S2(Procedure):

    def __init__(self, parameters, trajectories=False):
        Procedure.__init__(self, parameters)
        self.sigma = np.logspace(np.log10(parameters['sigma']), -1.0, num=10)
        self.total_bnabs = []
        self.num_reps = 100
        self.trajectories = trajectories
        # self.n_initial = [20.0, 5.0, 0.0, 0.0, 0.0, 5.0, 20.0]
        self.n_initial = parameters['n_initial']
        self.p_ini = self.n_initial / np.sum(self.n_initial)
        self.parameters = parameters

    def run_prime(self):
        bnab_prime = Simulation(self.p_ini, self.parameters, trajectories=self.trajectories)
        bnab_prime.run(reps=self.num_reps)

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

    def next_injection(self, index, sigma):
        make_and_cd("injection_{0}_sig_{1}".format(index, round(sigma, 2)))

        self.parameters['sigma'] = sigma
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

            bnab_next_injection = Simulation(self.p_ini, self.parameters, trajectories=self.trajectories)

            # if len(self.p_ini) > 7:
            #     gillespie_bnab = BnabFiniteSizeEffects(p_ini, fraction=self.fraction, sigma=sigma,
            #                                            death_rate=self.death_rate, n_stop=self.n_stop)
            # else:
            #     gillespie_bnab = BnabGillespie(p_ini, fraction=self.fraction, sigma=sigma, death_rate=self.death_rate,
            #                                    n_stop=self.n_stop)
            #
            # gillespie_bnab.define_tmat()

            if self.trajectories:
                M = ModelMFPTTrajectories(p_ini=p_ini, vnames=bnab_next_injection.gillespie_bnab.vars,
                                          tmat=bnab_next_injection.gillespie_bnab.tmat,
                                          propensity=bnab_next_injection.gillespie_bnab.prop, n_exit=n_i_exit[1:])

            else:
                M = ModelIndividualGC(n_exit=n_i_exit[1:], p_ini=p_ini, vnames=bnab_next_injection.gillespie_bnab.vars,
                                      tmat=bnab_next_injection.gillespie_bnab.tmat,
                                      propensity=bnab_next_injection.gillespie_bnab.prop,
                                      n_stop=self.parameters['n_stop'])

            M.run(tmax=1000, reps=count)
            # tc.append(M.t_exit)

            count += 1

        # np.savetxt("tc_gc_exit", tc, fmt='%f')
        np.savetxt("fitness", bnab_next_injection.gillespie_bnab.bnab.fitness.f, fmt='%f')
        np.savetxt("sigma", [bnab_next_injection.gillespie_bnab.bnab.fitness.sigma], fmt='%f')
        np.savetxt("death_rate", [bnab_next_injection.gillespie_bnab.bnab.mu_i0], fmt='%f')
        np.savetxt("mutation_rate", [bnab_next_injection.gillespie_bnab.bnab.mu_ij], fmt='%f')
        np.savetxt("fraction", [bnab_next_injection.gillespie_bnab.bnab.fraction], fmt='%f')

        print("Starting post-processing...")
        post_process = GillespieGCExit(n_exit=self.parameters['n_stop'],
                                       num_odes=bnab_next_injection.gillespie_bnab.len_ini)
        post_process.populate_pn(index=1)
