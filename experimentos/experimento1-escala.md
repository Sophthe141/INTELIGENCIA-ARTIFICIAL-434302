# Experimento 1: Impacto do Fator de Escala ($\sqrt{d_k}$) na Atenção

## Objetivo
Avaliar o impacto da divisão das pontuações de atenção pelo fator de escala $\sqrt{d_k}$ na distribuição de probabilidades calculada pela função Softmax e na estabilidade dos gradientes durante o treinamento.

## Metodologia
1. Foram gerados vetores aleatórios de *Query* ($Q$) e *Key* ($K$) com dimensão de embedding $d_k = 1024$.
2. Foram calculadas as pontuações de atenção sem o fator de escala ($QK^T$) e com o fator de escala ($QK^T / \sqrt{d_k}$).
3. Aplicou-se a função Softmax sobre ambas as matrizes de pontuação para obter as distribuições de probabilidade.
4. Os resultados foram plotados em histogramas comparativos de probabilidade.

## Imagem do Experimento
![Experimento de Escala](grafico_escala.png)

## Análise dos Resultados
* **Sem Escala ($QK^T$):** Como a variância do produto escalar cresce linearmente com a dimensão $d_k$, os valores brutos das pontuações atingiram magnitudes elevadas. Isso causou a saturação da função Softmax, resultando em uma distribuição "pontiaguda" (próxima de um vetor *one-hot*), onde um único token recebe probabilidade próxima de $1.0$ e os demais recebem $0.0$.
* **Com Escala ($QK^T / \sqrt{d_k}$):** A divisão pela raiz da dimensão reescalonou a variância das pontuações para próximo de $1$. Isso manteve a Softmax em uma região de distribuição suave e derivável.

## Conclusão
O uso do fator de escala $\sqrt{d_k}$ é indispensável para evitar o problema de desaparecimento de gradiente (*vanishing gradient*) em modelos Transformer de alta dimensão, garantindo que o aprendizado ocorra de forma contínua durante o *backpropagation*.