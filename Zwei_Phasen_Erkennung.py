import numpy as np
import pandas as pd

FILE_PATH = "C:/Projekte/Drei_Phasen_Erkennung/Raw_Data/2_Phasen_430V_FullLoad_1,6%_Speed_35%.csv"
FEATURE_BIASES = np.array(
    [
        2.2296546,
        11.718262,
        0.8022357,
        -0.9774496,
        3.9054496,
        -0.9765625,
        -9.766113,
        -2.9301758,
        -5.8598633,
        -2.9301758,
        -55.23152,
    ],
    dtype=np.float32,
)
STDS = np.array(
    [
        0.1449258,
        0.24325738,
        0.12326112,
        0.12848863,
        0.2021018,
        0.07310376,
        0.23516719,
        0.19519691,
        0.10768995,
        0.22614168,
        0.00310855,
    ],
    dtype=np.float32,
)
WEIGHTS = np.array(
    [
        2.60765275,
        0.77121237,
        2.21937962,
        -1.04119292,
        -0.71503676,
        1.12618569,
        0.04239991,
        -0.19107477,
        1.12125466,
        -3.05517863,
        -0.06443683,
    ],
    dtype=np.float32,
)
BIAS = np.array([0.00076851], dtype=np.float32)


def get_data(file_path):
    """
    Gets the data and prepares it!
    """
    X = pd.read_csv(file_path, delimiter=";").iloc[1:]
    X.columns = [col.replace(" ", "_") for col in X.columns]
    X = np.array(pd.to_numeric(X["DC_link_voltage"]))
    return X


def get_data_sample(X):
    """
    Gets a random segment of X that is 50 long!
    """
    start = np.random.randint(0, len(X) - 50)
    segment = X[start : start + 50]
    return segment


def feature_transform(X, biases):
    input_length = len(X)
    num_shifts = input_length - 9

    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 4],
            [0, 3, 7],
            [0, 5, 8],
            [1, 2, 7],
            [1, 4, 6],
            [1, 7, 8],
            [2, 4, 7],
            [3, 4, 5],
            [3, 6, 8],
            [5, 6, 7],
        ],
        dtype=np.int32,
    )

    num_kernels = len(indices)

    features = np.zeros(num_kernels, dtype=np.float32)

    _X = np.array(X)
    A = -_X
    G = _X + _X
    for kernel_index, index in enumerate(indices):

        outputs = np.zeros(num_shifts)
        for start in range(num_shifts):

            end = start + 9
            sub_array = A[start:end].copy()
            sub_array[index] = G[start:end][index]
            outputs[start] = np.sum(sub_array)

        features[kernel_index] = np.mean(outputs > biases[kernel_index])

    return features


def standart_scaler(X, stds):
    return X / stds


def logistic_regression(X, weights, bias):
    z = np.sum(weights * X) + bias
    y = 1 / (1 + np.exp(-z))
    return y[0]


def main():
    X_prepared = get_data(FILE_PATH)
    X_sample = get_data_sample(X_prepared)
    features = feature_transform(X_sample, FEATURE_BIASES)
    standarized_features = standart_scaler(features, STDS)
    prediction = logistic_regression(standarized_features, WEIGHTS, BIAS)
    if prediction > 0.7:
        print("Es sind 2 Phasen angeschlossen!")
    elif prediction < 0.3:
        print("Es sind 3 Phasen angeschlossen!")
    else:
        print(f"Das Modell ist sich nicht sicher! Prediction:{prediction}")


if __name__ == "__main__":
    main()
