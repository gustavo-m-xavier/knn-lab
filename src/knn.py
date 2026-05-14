from collections import Counter
from typing import Callable, Literal, Optional
import heapq
import numpy as np
from numpy.typing import NDArray

KValueStrategy = Literal['overfitting', 'ideal', 'underfitting']


class _KDNode:
    """
    Nó da árvore KD usado para armazenar uma amostra de referência.

    params:
        point (NDArray): ponto de treinamento armazenado no nó
        index (int): índice do ponto no conjunto de treinamento
        axis (int): dimensão usada como pivô para a divisão
    """
    __slots__ = ('point', 'index', 'left', 'right', 'axis')

    def __init__(self, point: NDArray, index: int, axis: int):
        self.point = point
        self.index = index
        self.axis = axis
        self.left: Optional['_KDNode'] = None
        self.right: Optional['_KDNode'] = None


class KNN:
    """
    Classe de implementação do algoritmo K-Nearest Neighbors (KNN) com índice KD-tree.

    Esta implementação utiliza uma árvore KD para otimizar a busca dos k vizinhos mais próximos,
    reduzindo a complexidade de O(n²) do algoritmo brute-force para aproximadamente O(log n)
    por consulta após uma construção inicial O(n log n).

    Comparação com algoritmo original:
    - Brute-force: O(n²) - calcula distância para todos os pontos em cada predição
    - KD-tree: O(n log n) construção + O(log n) por consulta - particiona o espaço eficientemente

    params:
        distance_formula (Callable[[NDArray, NDArray], float]): função de cálculo da distância
        neighbors (int): número de vizinhos mais próximos (padrão: 3)
        k_value_strategy (KValueStrategy): estratégia para determinar k final
    """

    def __init__(self, distance_formula: Callable[[NDArray, NDArray], float],
                 neighbors: int = 3,
                 k_value_strategy: KValueStrategy = 'underfitting'):
        """
        Inicializa o classificador KNN.

        params:
            `distance_formula (Callable[[NDArray, NDArray], float])`: função de cálculo da distância entre pontos a ser utilizada
            `neighbors (int)`: número de vizinhos mais próximos
            `k_value_strategy (KValueStrategy)`: opcional define uma heurística para determinar o valor final de K:
                - `'overfitting'`: define K como `1` para forçar o modelo a capturar cada ponto, incluindo possíveis ruídos.
                - `'ideal'`: define uma fronteira 'suave' ao ignorar ruídos isolados, definido K como `(int) sqrt(neighbors)`.
                - `'underfitting'` (valor padrão): define o valor de `neighbors` como o valor final de K
        """
        if neighbors < 1:
            raise ValueError(
                "Pelo menos um vizinho deve ser definido como hiperparâmetro do algoritmo.")

        match k_value_strategy:
            case 'overfitting':
                self.k = 1
            case 'ideal':
                self.k = int(round(np.sqrt(neighbors)))
            case _:
                self.k = neighbors

        self.distance_formula = distance_formula
        self.ref_samples: Optional[NDArray] = None
        self.ref_feats: Optional[NDArray] = None
        self.tree: Optional[_KDNode] = None
        self.dim: Optional[int] = None

    def _build_tree(self, indices: list[int], depth: int = 0) -> Optional[_KDNode]:
        """
        Constrói recursivamente a KD-tree a partir do conjunto de amostras de referência.

        params:
            indices (list[int]): índices das amostras que devem ser inseridas na subárvore
            depth (int): profundidade atual da recursão, usada para escolher o eixo de divisão
        returns:
            Optional[_KDNode]: raiz da subárvore construída
        """
        if not indices:
            return None

        axis = depth % self.dim

        indices.sort(key=lambda idx: self.ref_samples[idx][axis])

        median = len(indices) // 2
        node = _KDNode(self.ref_samples[indices[median]],
                       indices[median], axis)
        node.left = self._build_tree(indices[:median], depth + 1)
        node.right = self._build_tree(indices[median + 1:], depth + 1)

        return node

    def _search_tree(self, node: Optional[_KDNode], sample: NDArray,
                     k: int, best: list[tuple[float, int]]):
        """
        Busca os k vizinhos mais próximos de `sample` na KD-tree.

        params:
            node (Optional[_KDNode]): nó atual da árvore a ser pesquisado
            sample (NDArray): ponto de consulta para o qual os vizinhos são buscados
            k (int): número de vizinhos a retornarem
            best (list[tuple[float, int]]): heap de tamanho limitado com as melhores correspondências encontradas
        """
        if node is None:
            return

        distance = self.distance_formula(sample, node.point)

        if len(best) < k:
            heapq.heappush(best, (-distance, node.index))
        elif distance < -best[0][0]:
            heapq.heapreplace(best, (-distance, node.index))

        axis_distance = abs(sample[node.axis] - node.point[node.axis])
        first_branch = node.left if sample[node.axis] < node.point[node.axis] else node.right
        second_branch = node.right if sample[node.axis] < node.point[node.axis] else node.left

        self._search_tree(first_branch, sample, k, best)

        if len(best) < k or axis_distance < -best[0][0]:
            self._search_tree(second_branch, sample, k, best)

    def _k_nearest_indices(self, sample: NDArray) -> list[int]:
        """
        Retorna os índices dos k vizinhos mais próximos para um ponto de consulta.

        params:
            sample (NDArray): ponto de consulta usado na busca
        returns:
            list[int]: índices dos k pontos mais próximos no conjunto de referência
        """
        best: list[tuple[float, int]] = []
        self._search_tree(self.tree, sample, self.k, best)
        return [index for _, index in sorted([(-distance, index) for distance, index in best])]

    def fit(self, samples: NDArray, feats: NDArray):
        """
        Configura o classificador KNN ao definir um dataset de referências para realização das previsões.

        params:
            samples (NDArray): conjunto de pontos de treinamento
            labels (NDArray): os atributos dos pontos de treinamento
        """
        self.ref_samples = np.asarray(samples)
        self.ref_feats = np.asarray(feats)

        if self.ref_samples.ndim == 1:
            self.ref_samples = self.ref_samples.reshape(-1, 1)

        if self.ref_samples.shape[0] != len(self.ref_feats):
            raise ValueError(
                "O número de amostras deve corresponder ao número de atributos.")

        if self.ref_samples.shape[0] == 0:
            raise ValueError("O conjunto de treinamento não pode estar vazio.")

        self.dim = self.ref_samples.shape[1]
        self.k = min(self.k, len(self.ref_samples))
        self.tree = self._build_tree(list(range(len(self.ref_samples))))

    def predict(self, test_samples: NDArray) -> list:
        """
        Realiza previsões para os dados de teste `test_samples` a partir do dataset de referências.

        params:
            test_samples (NDArray): conjunto de dados de teste
        returns:
            list: uma lista dos atributos previstos para `test_samples`
        """
        if self.tree is None or self.ref_samples is None or self.ref_feats is None:
            raise ValueError(
                "O modelo deve ser ajustado com fit() antes de realizar previsões.")

        test_samples = np.asarray(test_samples)

        if test_samples.ndim == 1:
            test_samples = test_samples.reshape(-1, self.dim)

        predictions = []

        for sample in test_samples:
            k_indices = self._k_nearest_indices(sample)
            k_feats = [self.ref_feats[idx] for idx in k_indices]
            most_common = Counter(k_feats).most_common(1)

            if most_common:
                predictions.append(most_common[0][0])

        return predictions
