import numpy as np

def euclidean_distance(x, y):
    """
    Calcula a distância euclidiana entre dois vetores x e y.

    params:
        x: array, vetor de características
        y: array, vetor de características
    returns:
        float, a distância euclidiana entre x e y
    """
    return np.sqrt(np.sum((x - y)** 2))

def manhattan_distance(x, y):
    """
    Calcula a distância de Manhattan entre dois vetores x e y.

    params:
        x: array, vetor de características
        y: array, vetor de características
    returns:
        float, a distância de Manhattan entre x e y
    """
    return np.sum(np.abs(x - y))

def minkowski_distance(x, y, p=3):
    """
    Calcula a distância de Minkowski entre dois vetores x e y.

    params:
        x: array, vetor de características
        y: array, vetor de características
        p: int, ordem da distância de Minkowski, valor padrão é 3
    returns:
        float, a distância de Minkowski entre x e y
    """
    return np.sum(np.abs(x - y) ** p) ** (1 / p)