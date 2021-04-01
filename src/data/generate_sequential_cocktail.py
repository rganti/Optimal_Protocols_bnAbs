from src.data.gillespie_models import define_n_initial
from src.data.procedures import Procedure

if __name__ == "__main__":
    parameters = {'n_initial': define_n_initial(15), 'death_rate': 0.02, 'fraction': 7.0 / 8.0, 'mutation_rate': 0.05,
                  'n_stop': 200, 'sigma': 1.2, 'sigma2': 0.7, 'sigma3': 0.3}

    protocol = Procedure(parameters)
    protocol.run_sequential()
