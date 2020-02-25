from src.data.procedures import Procedure, define_n_initial

if __name__ == "__main__":

    parameters = {'n_initial': define_n_initial(15), 'death_rate': 0.02, 'fraction': 7.0 / 8.0, 'mutation_rate': 0.05,
                  'n_stop': 200, 'sigma': 1.3, 'sigma2': 1.0, 'sigma3': 0.8}

    protocol = Procedure(parameters)
    protocol.run_cocktail()
