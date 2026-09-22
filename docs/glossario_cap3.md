# Glossário Técnico - Sprint 3 (Mecanismos de Atenção)

## Conceitos Fundamentais
* **Self-Attention (Autoatenção):** Mecanismo que permite a cada elemento de uma sequência calcular uma representação ponderada baseada no seu relacionamento com todos os outros elementos da mesma sequência.
* **Queries, Keys e Values ($Q, K, V$):** Vetores obtidos por projeções lineares da entrada:
  * **Query ($q^{(i)}$):** Vetor que busca contexto.
  * **Key ($k^{(j)}$):** Vetor que atua como rótulo de correspondência.
  * **Value ($v^{(j)}$):** Vetor com o conteúdo de informação a ser agregado.
* **Pontuações de Atenção (Attention Scores):** Valores escalares obtidos pelo produto escalar entre vetores *Query* e *Key* ($q^{(i)} \cdot k^{(j)}$), medindo a afinidade entre tokens antes da normalização.
* **Pesos de Atenção (Attention Weights):** Probabilidades obtidas ao aplicar a função Softmax às pontuações de atenção. A soma de todos os pesos referentes a um token é igual a 1.

## Variações e Arquiteturas
* **Scaled Dot-Product Attention:** Mecanismo em que as pontuações de produto escalar são divididas por $\sqrt{d_k}$ (onde $d_k$ é a dimensão do vetor *Key*). Evita a saturação da função Softmax e previne o desaparecimento de gradientes (*vanishing gradients*) durante o treinamento.
* **Causal Attention (Atenção Causal / Mascarada):** Variação que impede que um token acesse posições futuras na sequência. Aplica uma máscara triangular superior com valores $-\infty$ nas pontuações antes do cálculo da Softmax.
* **Multi-Head Attention (Atenção Multi-Cabeça):** Arquitetura que divide as projeções de entrada em $h$ subespaços (*heads*) independentes, permitindo ao modelo aprender diferentes tipos de relações contextuais simultaneamente.
* **Dropout:** Técnica de regularização aplicada aos pesos de atenção durante o treinamento, zerando aleatoriamente conexões para evitar sobreajuste (*overfitting*).