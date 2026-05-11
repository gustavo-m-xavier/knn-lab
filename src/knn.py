from collections import Counter
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray


class KNN:
    """
    Classe de implementação do algoritmo K-Nearest Neighbors (KNN)
    """

    def __init__(self, distance_formula: Callable[[NDArray, NDArray], Any], neighbors=3):
        """
        Inicializa o classificador KNN.

        params:
            distance_formula (Callable[[Any, Any], Any]): função de distância a ser utilizada
            neighbors (int): número de vizinhos mais próximos
        """
        self.k = neighbors
        self.distance_formula = distance_formula

    def fit(self, samples: NDArray, labels: NDArray):
        """
        Treina o classificador KNN.

        params:
            samples: array, conjunto de dados de treinamento
            labels: array, rótulos de treinamento
        """
        self.samples = samples
        self.labels = labels

    def predict(self, test_samples: NDArray) -> list:
        """
        Faz previsões para os dados de teste.

        params:
            test_samples (NDArray): conjunto de dados de teste
        returns:
            list: rótulos previstos
        """
        predictions = []

        for sample in test_samples:
            distances = []

            for x_train in self.samples:
                distance = self.distance_formula(sample, x_train)

                distances.append(distance)

            k_indexes = np.argsort(distances)[:self.k]
            k_labels = [self.labels[i] for i in k_indexes]
            most_common = Counter(k_labels).most_common(1)

            predictions.append(most_common[0][0])

        return predictions
