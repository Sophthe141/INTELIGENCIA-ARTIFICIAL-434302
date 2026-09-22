# Experimento 2: Validação da Máscara Causal

## Objetivo
Verificar se a aplicação da máscara triangular superior preenchida com $-\infty$ garante que o modelo não acesse informações de tokens futuros na sequência durante o cálculo da atenção.

## Metodologia
1. A classe `CausalAttention` foi instanciada com janela de contexto igual a 4.
2. Uma sequência de teste composta por 4 tokens (`["O", "modelo", "aprende", "contexto"]`) foi passada pelo módulo.
3. A matriz de pesos de atenção resultante (`attn_weights`) foi extraída e plotada no formato de *heatmap*.

## Imagem do Experimento
![Matriz de Atenção Causal](matriz_causal.png)

## Análise dos Resultados
* Como observado na matriz visualizada, todos os elementos posicionados acima da diagonal principal ($j > i$) possuem peso exatamente igual a $0.00$.
* O token na posição $i$ atribui pesos de atenção exclusivamente aos tokens nas posições de $0$ até $i$.
* A conversão das posições mascaradas para $-\infty$ antes do cálculo da Softmax garante que $e^{-\infty} = 0$, zerando a probabilidade de vazamento de informação do futuro.

## Conclusão
A implementação da máscara causal funciona conforme o esperado, preservando a propriedade autoregressiva necessária para a arquitetura de modelos de linguagem generativos (como o GPT).