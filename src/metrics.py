import numpy as np
from numpy.typing import NDArray


def knn_accuracy(expected_feats: NDArray, predicted_feats: list):
    """
    Calcula a acurácia entre os atributos verdadeiros e os atributos previstos.
    """
    return np.sum(expected_feats == predicted_feats) / len(expected_feats)
