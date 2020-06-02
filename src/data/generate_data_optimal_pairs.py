import argparse
import os
import socket

import numpy as np
from src.data.generate_protocol_curves import SlurmProtocolCurves, QsubProtocolCurves
from src.data_process.simulation_post_process import GillespieGCExit
from src.data_process.trajectories_post_process import ComputeTrajectorySuccessProbability
from src.general.directory_handling import make_and_cd
from src.general.queuing import run_sbatch, run_qsub

from src.data.procedures import ProcedureDelS1S2, define_n_initial


class SlurmOptimalPairs(SlurmProtocolCurves):
    def __init__(self, sigma_1, sigma_2, file):
        SlurmProtocolCurves.__init__(self, sigma_1, simulation_time=60, nodes=1, ppn=1)
        self.sigma_2 = sigma_2
        self.file = file

    def set_python_script(self, q):
        q.write("python {0} --sigma1 {1} --sigma2 {2} \n".format(self.file, self.sigma_1, self.sigma_2))


class QsubOptimalPairs(QsubProtocolCurves):
    def __init__(self, sigma_1, sigma_2, file):
        QsubProtocolCurves.__init__(self, sigma_1, simulation_time=20, nodes=1, ppn=1)
        self.sigma_2 = sigma_2
        self.file = file

    def set_python_script(self, q):
        q.write("python {0} --sigma1 {1} --sigma2 {2} \n".format(self.file, self.sigma_1,
                                                                 self.sigma_2))


class ProcedureDelS1S2OptimalPairs(ProcedureDelS1S2):

    def __init__(self, parameters, trajectories=False, trees=False):
        ProcedureDelS1S2.__init__(self, parameters, trajectories=trajectories, trees=trees)
        self.sigma = [parameters['sigma'], parameters['sigma2']]
        self.num_reps = 20

    def post_process(self, bnab_next_injection):
        print("Starting post-processing...")
        process_hashed_files = GillespieGCExit(n_exit=self.parameters['n_stop'],
                                               num_odes=bnab_next_injection.gillespie_bnab.len_ini)
        process_hashed_files.populate_pn(index=1)

        if self.trajectories:
            process_trajectories = ComputeTrajectorySuccessProbability(num_odes=len(bnab_next_injection.p_ini)+1)
            process_trajectories.process_trajectories()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submitting immune protocol calculations.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sigma1', dest='sigma1', action='store', type=float,
                        help="Variance of prime fitness for simulation protocols.")
    parser.add_argument('--sigma2', dest='sigma2', action='store',
                        type=float, help="Variance of boost fitness for simulation protocols.")

    args = parser.parse_args()

    parameters = {'n_initial': define_n_initial(15), 'death_rate': 0.02,
                  'fraction': 7.0 / 8.0, 'mutation_rate': 0.05, 'n_stop': 200}

    if args.sigma1:
        parameters['sigma'] = args.sigma1
        parameters['sigma2'] = args.sigma2

        protocol = ProcedureDelS1S2OptimalPairs(parameters, trajectories=False)
        protocol.run_sequential()

    else:
        optimal_pairs = np.loadtxt("optimal_pairs")
        make_and_cd("optimal_boost")

        home = os.getcwd()

        for sigma_1, sigma_2 in optimal_pairs[1:-1]:
            make_and_cd("Sigma_{0}".format(round(sigma_1, 2)))

            sigma_directory = os.getcwd()

            for t in range(30):
                make_and_cd("Trial_{0}".format(t))

                if socket.gethostname() == "eofe4.mit.edu" or socket.gethostname() == "eofe7.cm.cluster":
                    sbatch = SlurmOptimalPairs(sigma_1, sigma_2, os.path.realpath(__file__))
                    sbatch.generate_sbatch()
                    run_sbatch()
                else:
                    qsub = QsubOptimalPairs(sigma_1, sigma_2, os.path.realpath(__file__))
                    qsub.generate_qsub()
                    run_qsub()

                os.chdir(sigma_directory)

            os.chdir(home)

            # if socket.gethostname() == "eofe4.mit.edu":
            #     sbatch = SlurmOptimalPairs(sigma_1, sigma_2)
            #     sbatch.generate_sbatch()
            #     run_sbatch()
            # else:
            #     qsub = QsubOptimalPairs(sigma_1, sigma_2)
            #     qsub.generate_qsub()
            #     run_qsub()
            #
            # os.chdir(home)
