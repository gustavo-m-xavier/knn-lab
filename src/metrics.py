import numpy as np


def accuracy(true_labels, predicted_labels):
    """
    Calcula a acurácia entre os rótulos verdadeiros e os rótulos previstos.
    """
    return np.sum(true_labels == predicted_labels) / len(true_labels)
