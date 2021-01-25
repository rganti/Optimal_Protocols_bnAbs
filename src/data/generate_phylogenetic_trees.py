import argparse
import os
import socket

import numpy as np

from src.data.generate_data_optimal_pairs import ProcedureDelS1S2OptimalPairs, SlurmOptimalPairs, QsubOptimalPairs
from src.data.gillespie_models import define_n_initial
from src.general.directory_handling import make_and_cd
from src.general.queuing import run_sbatch, run_qsub

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

        protocol = ProcedureDelS1S2OptimalPairs(parameters, trees=True)
        protocol.run_sequential()

    else:
        optimal_pairs = np.loadtxt("optimal_pairs")
        make_and_cd("phylogenetic_trees")

        home = os.getcwd()

        for sigma_1, sigma_2 in optimal_pairs[1:-2]:
            make_and_cd("Sigma_{0}".format(round(sigma_1, 2)))

            # sigma_directory = os.getcwd()

            # for t in range(30):
            #     make_and_cd("Trial_{0}".format(t))

            if socket.gethostname() == "eofe4.mit.edu" or socket.gethostname() == "eofe7.cm.cluster":
                sbatch = SlurmOptimalPairs(sigma_1, sigma_2, os.path.realpath(__file__))
                sbatch.generate_sbatch()
                run_sbatch()
            else:
                qsub = QsubOptimalPairs(sigma_1, sigma_2, os.path.realpath(__file__))
                qsub.generate_qsub()
                run_qsub()

            # os.chdir(sigma_directory)

            os.chdir(home)
