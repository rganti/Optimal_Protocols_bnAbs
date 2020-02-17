import pickle


def pickle_out_data(data, pickle_name):
    pickle_out = open(pickle_name + ".pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_in_data(pickle_name):
    pickle_in = open(pickle_name + ".pickle", "rb")
    sequences = pickle.load(pickle_in)
    pickle_in.close()

    return sequences


def write_results(data, i):
    f = open("hashed_traj_{0}".format(i), "w")
    for line in data:
        x = [l for l in line]
        for entry in x:
            f.write(str(entry) + " ")
        f.write("\n")
    f.close()