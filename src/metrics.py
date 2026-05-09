import numpy as np

def accuracy(y_true, y_pred):
    """
    Calcula a acurácia entre os rótulos verdadeiros e os rótulos previstos.
    """
    return np.sum(y_true == y_pred) / len(y_true)