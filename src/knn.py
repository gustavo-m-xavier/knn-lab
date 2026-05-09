from collections import Counter
import numpy as np

class KNN:
    """
    Classe de implementação do algoritmo K-Nearest Neighbors (KNN)
    """
    def __init__(self, k=3, distance_function=None):
        """
        Inicializa o classificador KNN.

        params:
            k: int, número de vizinhos mais próximos
            distance_function: função, função de distância a ser utilizada
        returns:
            None
        """
        self.k = k
        self.distance_function = distance_function

    def fit(self, X, y):
        """
        Treina o classificador KNN.

        params:
            X: array, conjunto de dados de treinamento
            y: array, rótulos de treinamento
        returns:
            None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Faz previsões para os dados de teste.

        params:
            X: array, conjunto de dados de teste
        returns:
            list, rótulos previstos
        """
        predictions = []

        for sample in X:
            distances = []

            for x_train in self.X_train:
                distance = self.distance_function(sample, x_train)
                distances.append(distance)

            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]

            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])

        return predictions