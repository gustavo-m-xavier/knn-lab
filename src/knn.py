from collections import Counter
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray

KValueStrategy = Literal['overfitting', 'ideal', 'underfitting']


class KNN:
    """
    Classe de implementação do algoritmo K-Nearest Neighbors (KNN)
    """

    def __init__(self, distance_formula: Callable[[NDArray, NDArray], float],
                 neighbors: int = 3,
                 k_value_strategy: KValueStrategy = 'underfitting'):
        """
        Inicializa o classificador KNN.

        params:
            `distance_formula (Callable[[NDArray, NDArray], float])`: função de cálculo da distância entre pontos a ser utilizada
            `neighbors (int)`: número de vizinhos mais próximos
            `k_value_strategy (KValueStrategy)`: OPCIONALMENTE define uma heurística para determinar o valor final de K:
                - `'overfitting'`: define K como `1` para forçar o modelo a capturar cada ponto, incluíndo possíveis ruídos.
                - `'ideal'`: define uma fronteira 'suave' ao ignorar ruídos isolados, definido K como `(int) sqrt(neighbors)`.
                - `'underfitting'` (valor padrão): define o valor de `neighbors` como o valor final de K
        """
        if neighbors < 1:
            raise ValueError(
                "Pelo menos um vizinho deve ser definido como hiperparâmetro do algoritmo.")

        match(k_value_strategy):
            case 'overfitting':
                self.k = 1

            case 'ideal':
                self.k = int(round(np.sqrt(neighbors)))

            case _:
                self.k = neighbors

        self.distance_formula = distance_formula

    def fit(self, samples: NDArray, feats: NDArray):
        """
        Configura o classificador KNN ao definir um dataset de referências para realização das previsões.

        params:
            samples (NDArray): conjunto de pontos de treinamento
            labels (NDArray): os atributos dos pontos de treinamento
        """
        self.ref_samples = samples
        self.ref_feats = feats

    def predict(self, test_samples: NDArray) -> list:
        """
        Realiza previsões para os dados de teste `test_samples` à partir do dataset de referências.

        params:
            test_samples (NDArray): conjunto de dados de teste
        returns:
            list: uma lista dos atributos previstos para `test_samples`
        """
        predictions = []

        for test in test_samples:
            distances = []

            for ref in self.ref_samples:
                distance = self.distance_formula(test, ref)

                distances.append(distance)

            k_indexes = np.argsort(distances)[:self.k]
            k_features = [self.ref_feats[i] for i in k_indexes]
            most_common = Counter(k_features).most_common(1)

            if len(most_common) > 0:
                predictions.append(most_common[0][0])

        return predictions
