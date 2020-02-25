'''This script runs through all possible prime fitness values and all possible boost fitness values to generate
the protocol curves showing bnabs/gc produced versus KL distance of p1 and f2.'''

import argparse
import os
import socket
import time

import numpy as np

from src.data.procedures import ProcedureDelS1S2, define_n_initial
from src.general.directory_handling import make_and_cd
from src.general.queuing import QsubHeader, SlurmHeader, run_sbatch, run_qsub


class QsubProtocolCurves(object):
    def __init__(self, sigma_1, simulation_time=30, nodes=1, ppn=1):
        self.sigma_1 = sigma_1
        self.header = QsubHeader(simulation_name="protocols_{0}".format(round(self.sigma_1, 2)),
                                 simulation_time=simulation_time, nodes=nodes, ppn=ppn)
        self.script_name = "generate_protocol_curves.py"

    def set_python_script(self, q):
        q.write("python {0}/{1} --sigma1 {2} \n".format(os.path.dirname(__file__), self.script_name, self.sigma_1))

    def generate_qsub(self):
        q = open("qsub.sh", "w")
        self.header.set_qsub_header(q)
        self.set_python_script(q)
        q.close()


class SlurmProtocolCurves(object):
    def __init__(self, sigma_1, simulation_time=30, nodes=1, ppn=1):
        self.sigma_1 = sigma_1
        self.header = SlurmHeader(simulation_name="optimal_curves", simulation_time=simulation_time, nodes=nodes,
                                  ppn=ppn)
        self.script_name = "generate_protocol_curves.py"

    def set_python_script(self, q):
        q.write("python {0}/{1} --sigma1 {2} \n".format(os.path.dirname(__file__), self.script_name, self.sigma_1))

    def generate_sbatch(self):
        q = open("sbatch.sh", "w")
        self.header.set_header(q)
        self.set_python_script(q)
        q.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submitting immune protocol calculations.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sigma1', dest='sigma1', action='store', type=float,
                        help="Variance of prime fitness for simulation protocols.")

    args = parser.parse_args()

    parameters = {'n_initial': define_n_initial(15), 'sigma': args.sigma1, 'death_rate': 0.02,
                  'fraction': 7.0 / 8.0, 'mutation_rate': 0.05, 'n_stop': 200}

    if args.sigma1:
        protocol = ProcedureDelS1S2(parameters)
        # protocol.run_sequential()
        protocol.run_prime()
    else:
        make_and_cd("Trials_dr_{0}_muij_{1}_bins_{2}".format(parameters['death_rate'], parameters['mutation_rate'],
                                                             len(parameters['n_initial'])))

        home = os.getcwd()

        # sigma_1_range = np.logspace(1.0, -1.0, num=15)
        # # sigma_1_range = np.logspace(0.6, 0.0, num=20)
        sigma_1_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        np.savetxt("sigma_1_range", sigma_1_range[::-1], fmt='%f')

        for sigma1 in sigma_1_range:
            make_and_cd("Sigma_{0}".format(round(sigma1, 2)))

            sigma_directory = os.getcwd()

            for t in range(30):
                make_and_cd("Trial_{0}".format(t))

                if socket.gethostname() == "eofe4.mit.edu" or socket.gethostname() == "eofe7.cm.cluster":
                    sbatch = SlurmProtocolCurves(sigma_1=sigma1)
                    sbatch.generate_sbatch()
                    run_sbatch()
                else:
                    qsub = QsubProtocolCurves(sigma_1=sigma1)
                    qsub.generate_qsub()
                    run_qsub()

                os.chdir(sigma_directory)

            os.chdir(home)

