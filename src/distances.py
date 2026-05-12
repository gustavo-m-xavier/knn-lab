import numpy as np
from numpy.typing import NDArray


def euclidean_distance(x: NDArray, y: NDArray):
    """
    Calcula a distância euclidiana entre dois vetores x e y.

    params:
        x: array, vetor de características
        y: array, vetor de características
    returns:
        float, a distância euclidiana entre x e y
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x: NDArray, y: NDArray):
    """
    Calcula a distância de Manhattan entre dois vetores x e y.

    params:
        x: array, vetor de características
        y: array, vetor de características
    returns:
        float, a distância de Manhattan entre x e y
    """
    return np.sum(np.abs(x - y))


def minkowski_distance(x: NDArray, y: NDArray, order: int = 3):
    """
    Calcula a distância de Minkowski entre dois vetores x e y.

    params:
        x: array, vetor de características
        y: array, vetor de características
        p: int, ordem da distância de Minkowski, valor padrão é 3
    returns:
        float, a distância de Minkowski entre x e y
    """
    return np.sum(np.abs(x - y) ** order) ** (1 / order)


def hamming_distance(x: NDArray, y: NDArray):
    """
    Calcula a distância de Hamming entre dois vetores x e y.

    A distância de Hamming representa a proporção de elementos
    diferentes entre os dois vetores.

    params:
        x: array, vetor de características
        y: array, vetor de características

    returns:
        float, a distância de Hamming entre x e y
    """

    if len(x) != len(y):
        raise ValueError("Os vetores precisam ter o mesmo tamanho.")

    return np.sum(x != y) / len(x)