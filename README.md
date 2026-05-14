# knn-lab

Estudo do algoritmo KNN utilizando algoritmos de distâncias diferentes:

- Distância Euclidiana
- Distância de Manhattan
- Distância de Minkowski

## Objetivos

- Entender como o KNN funciona
- Comparar métricas de distância
- Analizar comportamento e precisão

## Otimização com KD-tree

A implementação atual utiliza uma **árvore KD (KD-tree)** para otimizar a busca dos k vizinhos mais próximos, reduzindo significativamente a complexidade computacional em relação ao algoritmo original.

### Comparação de Complexidade

**Algoritmo Original (Brute-force):**
- Complexidade: **O(n²)** para cada predição
- Para cada ponto de teste, calcula a distância para todos os n pontos de treinamento
- Simples de implementar, mas ineficiente para grandes datasets

**Algoritmo Otimizado (KD-tree):**
- Complexidade: **O(n log n)** para construção da árvore + **O(log n)** por consulta
- Constrói uma estrutura de dados espacial durante o treinamento
- Busca eficiente através de particionamento recursivo do espaço
- Ideal para datasets de alta dimensionalidade e consultas frequentes

### Benefícios da KD-tree

- **Performance:** Redução drástica no tempo de predição para grandes datasets
- **Escalabilidade:** Melhor adaptação a datasets crescentes
- **Eficiência:** Evita cálculos desnecessários através de poda de ramos da árvore
- **Mantém Precisão:** Resultados idênticos ao algoritmo original

## Como Rodar o Projeto

Para rodar o projeto, basta clonar o repositório, criar um environment e baixar as dependências

```cmd
python3 -m venv .venv

pip install -r requirements.txt
```

Baixe as extensões do nootebok jupyter para ter uma boa experiência no desenvolvimento e visutalização dos notebooks que estão na pasta `/notebooks`
