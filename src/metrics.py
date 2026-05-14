import numpy as np
import pandas as pd
from numpy.typing import NDArray


def knn_accuracy(expected_feats: NDArray, predicted_feats: list):
    """
    Calcula a acurácia entre os atributos verdadeiros e os atributos previstos.
    """
    return np.sum(expected_feats == predicted_feats) / len(expected_feats)


def knn_compare(samples: NDArray, expected: NDArray, predicted: list) -> pd.DataFrame:
    """
    Exibe a relação entre atributos esperados e efetivamente previstos.

    Args:
        `samples` (NDArray): o conjunto de pontos a serem analizados.
        `expected` (NDArray): o conjunto de atributos esperado para `samples`.
        `predicted` (list): o conjunto de atributos previstos pelo KNN.
    """
    data = []

    for i in range(len(samples)):
        data.append({
            'Ponto': str(samples[i]),
            'Atributo Esperado': expected[i],
            'Atributo Previsto': predicted[i]
        })

    return pd.DataFrame(data)
