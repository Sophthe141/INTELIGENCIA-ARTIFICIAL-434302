# Experimento 3: Análise do Mecanismo Multi-Head Attention

## Objetivo
Analisar como o mecanismo de *Multi-Head Attention* permite ao modelo aprender diferentes padrões de relacionamento contextual simultaneamente em subespaços independentes.

## Metodologia
1. Foi instanciado o módulo `MultiHeadAttention` configurado com $h = 2$ cabeças de atenção e dimensão total $d_{out} = 8$.
2. A mesma sequência de entrada foi processada em paralelo por ambas as cabeças.
3. As matrizes de pesos de atenção de cada *head* foram extraídas e plotadas lado a lado para comparação.

## Imagem do Experimento
![Comparativo Multi-Head](grafico_multihead.png)

## Análise dos Resultados
* **Cabeça 1:** Apresentou maior concentração de pesos em tokens vizinhos/imediatos, capturando relações de adjacência direta e estrutura sintática local.
* **Cabeça 2:** Distribuiu o peso de atenção entre tokens mais distantes na sequência, capturando dependências contextuais globais e relações semânticas mais amplas.
* As projeções lineares independentes ($W_Q, W_K, W_V$) permitiram que cada cabeça focasse em aspectos distintos da informação sem interferir nos outros subespaços.

## Conclusão
O *Multi-Head Attention* supera o *Self-Attention* simples por oferecer uma capacidade de representação rica e diversificada, permitindo ao modelo atentar para múltiplos contextos na mesma passagem de dados.