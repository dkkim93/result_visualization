import numpy as np


def postprocess_data(data):
    """
        Handle cases when data includes special character
        E.g., -1.531], [2.35, etc
    """
    special_character_n = ["[", "]", ","]

    for special_character in special_character_n:
        special_index = data.find(special_character)

        if special_index == -1:
            pass  # NOTE Special character is not included, thus pass
        else:
            if special_index == 0:
                data = data[1:]
            else:
                data = data[0:special_index]
            break

    return data


def read_data_from_raw(path, key, index):
    with open(path) as f:
        content = f.read().splitlines()

    data_n = []
    for line in content:
        if key in line:
            data = line.split()[index]
            data = postprocess_data(data)

            data_n.append(float(data))

    return data_n


def moving_average(data_set, periods=10):
    # Apply convolution for averaging data
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')
