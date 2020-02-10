import datetime
import os
import subprocess

from numpy.random import multinomial
import numpy as np

from src.visualization.visualize_fitness import Injection


class BnabModel(object):

    def __init__(self, p_ini, parameters):

        self.ntot = 50
        self.p_ini = p_ini
        self.num_odes = len(self.p_ini) + 1

        events = multinomial(self.ntot, p_ini)
        new = [i for i in events]

        # Initial Distribution: new = [20.0, 5.0, 0.0, 0.0, 0.0, 5.0, 20.0]
        self.n_0 = new
        self.fitness = Injection(sigma=parameters['sigma'], num_odes=self.num_odes)

        self.mu_i0 = parameters['death_rate']
        self.mu_ij = parameters['mutation_rate']
        self.fraction = parameters['fraction']
        self.n_initial, self.record = self.set_shared()

        self.forward_rates = {}
        self.forward_rxns = []

    def set_shared(self):
        n_initial = {}
        record = []
        for i in range(0, self.num_odes):
            if i > 0:
                # if len(self.n_0) > 1:
                n_initial["N{0}".format(i)] = int(np.round(self.n_0[i - 1]))
                # else:
                #     mid = int(self.num_odes/2.0)
                #     if (i == mid - 2) or (i == mid + 2):
                #         n_initial["N{0}".format(i)] = int(0.6 * self.n_0[0])
                #     elif (i == mid - 1) or (i == mid + 1):
                #         n_initial["N{0}".format(i)] = int(0.2 * self.n_0[0])
                #     elif i == mid:
                #         n_initial["N{0}".format(i)] = 0  # int(0.2 * self.n_0[0])
                #     else:
                #         n_initial["N{0}".format(i)] = self.n_0[0]
            record.append("N{0}".format(i))

        return n_initial, record

    def add_reaction(self, reactant, product, rate):
        forward_key = ''.join(reactant) + '_' + ''.join(product)
        self.forward_rates[forward_key] = rate
        self.forward_rxns.append([reactant, product])

    def define_reactions(self):
        mu_ij_dict = self.define_mu_ij()

        # Define death rates
        for i in range(1, self.num_odes):
            self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(0)], mu_ij_dict['mu{0}{1}'.format(i, 0)])

        # Define edge reaction 1
        i = 1
        self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(i), 'N{0}'.format(i)], mu_ij_dict['f{0}'.format(i)])
        self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(i + 1)], mu_ij_dict['mu{0}{1}'.format(i, i + 1)])

        # Define middle transitions
        for i in range(2, self.num_odes-1):
            self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(i - 1)], mu_ij_dict['mu{0}{1}'.format(i, i - 1)])
            self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(i), 'N{0}'.format(i)], mu_ij_dict['f{0}'.format(i)])
            self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(i + 1)], mu_ij_dict['mu{0}{1}'.format(i, i + 1)])

        # Define edge reaction num_odes - 1
        i = self.num_odes - 1
        self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(i), 'N{0}'.format(i)], mu_ij_dict['f{0}'.format(i)])
        self.add_reaction(['N{0}'.format(i)], ['N{0}'.format(i - 1)], mu_ij_dict['mu{0}{1}'.format(i, i - 1)])

        np.savetxt("fitness", self.fitness.f, fmt='%f')

    def define_mu_ij(self):
        mu_ij_dict = {}

        mu_i0 = self.mu_i0
        mu_ij = self.mu_ij

        for i in range(1, self.num_odes):
            mu_ij_dict['f{0}'.format(i)] = self.fitness.f[i - 1]
            mu_ij_dict['mu{0}{1}'.format(i, 0)] = mu_i0
            fraction = self.fraction

            dist = i - (self.num_odes / 2)
            deleterious_edge = fraction * mu_ij
            non_deleterious_edge = (1.0 - fraction) * mu_ij

            if abs(dist) == (self.num_odes / 2) - 1:
                if dist < 0:
                    mu_ij_dict['mu{0}{1}'.format(i, i + 1)] = non_deleterious_edge
                else:
                    mu_ij_dict['mu{0}{1}'.format(i, i - 1)] = non_deleterious_edge

            if abs(dist) < (self.num_odes / 2) - 1:
                if dist < 0:
                    mu_ij_dict['mu{0}{1}'.format(i, i - 1)] = deleterious_edge
                    mu_ij_dict['mu{0}{1}'.format(i, i + 1)] = non_deleterious_edge
                else:
                    mu_ij_dict['mu{0}{1}'.format(i, i - 1)] = non_deleterious_edge
                    mu_ij_dict['mu{0}{1}'.format(i, i + 1)] = deleterious_edge

            if abs(dist) == 0:
                mu_ij_dict['mu{0}{1}'.format(i, i - 1)] = 0.5 * mu_ij
                mu_ij_dict['mu{0}{1}'.format(i, i + 1)] = 0.5 * mu_ij

        return mu_ij_dict


class SharedCommands(object):

    def __init__(self, n_initial, record):
        self.n_initial = n_initial
        self.record = record

    def initialize(self, f):
        for key, value in self.n_initial.items():
            f.write("new {0} at {1}\n".format(key, value))

    def record_species(self, f):
        for item in self.record:
            f.write("record {0}\n".format(item))
        f.write("\n")


class SSCLaunch(object):
    def __init__(self, p_ini, parameters):
        self.bnab = BnabModel(p_ini, parameters)

        self.bnab.define_reactions()

        self.num_files = 200
        self.run_time = 1000  # must be integer
        self.time_step = 1.0

        self.simulation_time = 20

    def set_simulation_time(self):
        simulation_time = self.simulation_time  # self.run_time * (20.0 / 1000)
        return simulation_time

    def define_region(self, f):
        f.write('region World box width 1 height 1 depth 1\n')
        f.write('subvolume edge 1\n\n')

    def define_reactions(self, f, rxn, rate):
        for i in range(len(rxn)):
            input_string = "rxn "
            destroy_string = ""
            input = rxn[i][0]
            output = rxn[i][1]

            rate_key = ""
            for item in input:
                input_string += "{0}:{1} ".format(item.lower(), item)
                destroy_string += "destroy {0}; ".format(item.lower())
                rate_key += item

            rate_key += "_"
            output_string = ""
            for item in output:
                output_string += "new {0}; ".format(item)
                rate_key += item

            rate_string = "at {0} -> ".format(rate[rate_key])

            f.write(input_string + rate_string + destroy_string + output_string + "\n")

    def generate_qsub(self, simulation_name):
        q = open("qsub.sh", "w")
        q.write("#PBS -m ae\n")
        q.write("#PBS -q short\n")
        q.write("#PBS -p 999\n")
        q.write("#PBS -V\n")
        q.write("#PBS -l walltime={1},nodes=1:ppn=1 -N {0}\n\n".format(simulation_name,
                                                                       datetime.timedelta(
                                                                           minutes=self.set_simulation_time())))
        q.write("cd $PBS_O_WORKDIR\n\n")
        q.write("echo $PBS_JOBID > job_id\n")
        # if "test" in os.path.basename(os.getcwd()):
        #     q.write("rm traj*\n")
        #     q.write("rm hashed*\n")
        q.write("EXE_FILE={0}\n".format(simulation_name))
        q.write("RUN_TIME={0}\n".format(self.run_time))
        q.write("STEP={0}\n\n".format(self.time_step))
        q.write("for j in {1.." + str(self.num_files) + "}\n")
        q.write("do\n")
        q.write("FILE=hashed_traj_$j\n\n")
        q.write('\t while ! test -f "$FILE" \n')
        q.write("\t do \n")
        # if self.time_step == self.run_time:
        #     q.write("\t ./$EXE_FILE -e $RUN_TIME > traj_$j\n")
        # else:
        q.write("\t\t ./$EXE_FILE -e $RUN_TIME -t $STEP > traj\n")
        q.write("\t\t python ~/BNAB_fitness/bnab_post_process.py --index $j \n")
        q.write("\t\t RUN_TIME=$((RUN_TIME + 5)) \n")
        # q.write("\t\t echo No exit for j = $j\n")
        q.write("\t done \n\n")
        q.write("\t RUN_TIME={0}\n".format(self.run_time))
        q.write("done\n\n")
        self.qsub_post_process(q)
        q.close()

    def generate_qsub_ensemble(self, simulation_name):
        q = open("qsub.sh", "w")
        q.write("#PBS -m ae\n")
        q.write("#PBS -q short\n")
        q.write("#PBS -p 999\n")
        q.write("#PBS -V\n")
        q.write("#PBS -l walltime={1},nodes=1:ppn=1 -N {0}\n\n".format(simulation_name,
                                                                       datetime.timedelta(
                                                                           minutes=self.set_simulation_time())))
        q.write("cd $PBS_O_WORKDIR\n\n")
        q.write("echo $PBS_JOBID > job_id\n")
        q.write("EXE_FILE={0}\n".format(simulation_name))
        q.write("RUN_TIME={0}\n".format(self.run_time))
        q.write("STEP={0}\n\n".format(self.time_step))
        q.write("for j in {1.." + str(self.num_files) + "}\n")
        q.write("do\n")
        q.write("\t ./$EXE_FILE -e $RUN_TIME -t $STEP > traj_$j\n")
        q.write("done \n\n")
        self.qsub_post_process(q)
        q.close()

    def qsub_post_process(self, q):
        self.bnab_post_process(q)

    def bnab_post_process(self, q):
        q.write("python ~/BNAB_fitness/bnab_post_process.py\n\n")

    def generate_shell_script(self):
        pass

    def compile_script(self, script_name):
        subprocess.call(["ssc", "--save-expanded=network", "{0}".format(script_name)])

    def generate_ssc_script(self, simulation_name):
        script_name = simulation_name + ".rxn"
        shared = SharedCommands(self.bnab.n_initial, self.bnab.record)

        f = open(script_name, "w")

        self.define_region(f)
        f.write("-- Forward reactions \n")

        self.define_reactions(f, self.bnab.forward_rxns, self.bnab.forward_rates)

        f.write("\n")
        shared.initialize(f)
        f.write("\n")
        shared.record_species(f)

        f.close()

    def generate(self, simulation_name):
        self.generate_ssc_script(simulation_name)
        self.compile_script(simulation_name + ".rxn")
        self.generate_qsub(simulation_name)

    def generate_ensemble(self, simulation_name):
        self.generate_ssc_script(simulation_name)
        self.compile_script(simulation_name + ".rxn")
        self.generate_qsub_ensemble(simulation_name)

    def run_shell(self):
        pass

    def run(self):
        (stdout, stderr) = subprocess.Popen(["qsub {0}".format("qsub.sh")], shell=True, stdout=subprocess.PIPE,
                                            cwd=os.getcwd()).communicate()