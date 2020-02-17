import numpy as np


def find_optimal_pairs():
    sigma_1_range = np.loadtxt("sigma_1_range")
    optimal_pairs = []
    optimal_pairs_bnabs = []

    for sigma_1 in sigma_1_range:
        bnabs = np.loadtxt("Sigma_{0}/mean_bnabs".format(round(sigma_1, 2)))
        max_index = np.where(bnabs == max(bnabs))[0][0]

        optimal_pairs_bnabs.append(bnabs[max_index])
        sigma_2_range = np.loadtxt("Sigma_{0}/sigma2_range".format(round(sigma_1, 2)))
        sigma_2 = sigma_2_range[max_index]
        optimal_pairs.append([sigma_1, sigma_2])

    np.savetxt("optimal_pairs", optimal_pairs, fmt='%f')
    np.savetxt("optimal_pairs_bnabs", optimal_pairs_bnabs, fmt='%f')


if __name__ == "__main__":

    find_optimal_pairs()
