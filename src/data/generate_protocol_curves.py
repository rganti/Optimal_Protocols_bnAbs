'''This script runs through all possible prime fitness values and all possible boost fitness values to generate
the protocol curves showing bnabs/gc produced versus KL distance of p1 and f2.'''

import argparse
import os
import socket

import numpy as np

print(str(os.path.dirname(__file__)))

from src.data.procedures import ProcedureIndividualRestart
from src.general.directory_handling import make_and_cd
from src.general.queuing import QsubHeader, SlurmHeader, run_sbatch, run_qsub


def set_python_script(q, sigma):
    q.write("python {0}/generate_protocol_curves.py --sigma1 {1} \n".format(os.path.dirname(__file__), sigma))


class QsubProtocolCurves(object):
    def __init__(self, sigma_1, simulation_time=30):
        self.sigma_1 = sigma_1
        self.header = QsubHeader(simulation_name="protocols_{0}".format(round(self.sigma_1, 2)),
                                 simulation_time=simulation_time)

    def generate_qsub(self):
        q = open("qsub.sh", "w")
        self.header.set_qsub_header(q)
        set_python_script(q, self.sigma_1)
        q.close()


class SlurmProtocolCurves(object):
    def __init__(self, sigma_1, simulation_time=30, nodes=1, ppn=1):
        self.sigma_1 = sigma_1
        self.header = SlurmHeader(simulation_name="optimal_curves", simulation_time=simulation_time, nodes=nodes,
                                  ppn=ppn)

    def generate_sbatch(self):
        q = open("sbatch.sh", "w")
        self.header.set_header(q)
        set_python_script(q, self.sigma_1)
        q.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submitting immune protocol calculations.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sigma1', dest='sigma1', action='store', type=float,
                        help="Variance of prime fitness for simulation protocols.")

    args = parser.parse_args()

    if args.sigma1:
        protocol = ProcedureIndividualRestart(sigma_1=args.sigma1)
        protocol.run_sequential()
    else:
        home = os.getcwd()
        sigma_1_range = np.logspace(1.0, -1.0, num=15)
        sigma_1_range = [sigma_1_range[4], sigma_1_range[6]]

        for sigma1 in sigma_1_range:
            make_and_cd("Sigma_{0}".format(round(sigma1, 2)))

            if socket.gethostname() == "eofe4.mit.edu":
                sbatch = SlurmProtocolCurves(sigma_1=sigma1)
                sbatch.generate_sbatch()
                run_sbatch()
            else:
                qsub = QsubProtocolCurves(sigma_1=sigma1)
                qsub.generate_qsub()
                run_qsub()

            os.chdir(home)
