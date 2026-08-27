
# Análise Técnica - Sprint 2: Processamento de Dados para LLM

## 1. Fundamentação Teórica e Respostas Técnicas

* **Processamento de Texto Bruto:** LLMs dependem de operações algébricas e multiplicações de matrizes. Redes neurais não processam strings diretamente; exigem representações numéricas continuas.
* **Função do Vocabulário:** Mapeia unidirecionalmente e bidirecionalmente cada token único para um ID inteiro, servindo como o dicionário central do modelo.
* **Token vs Token ID:** Token é a unidade de texto (ex: `"modelo"`). Token ID é a chave inteira equivalente no vocabulário (ex: `1`).
* **Inadequação dos IDs como Semântica:** IDs são identificadores categóricos arbitrários. O ID `10` não possui proximidade matemática ou relação de significado com o ID `11`.
* **Função dos Embeddings:** Mapeiam IDs para vetores contínuos de números reais onde distâncias (ex: similaridade de cosseno) refletem relações semânticas do texto.
* **Necessidade da Posição:** A arquitetura Transformer processa todos os tokens em paralelo. Sem o *Positional Embedding*, a ordem temporal e sintática da frase seria perdida.
* **Relação Contexto vs Amostras:** Quanto maior o tamanho do contexto, menor é a quantidade de amostras completas extraídas do mesmo texto.
* **Impacto da Dimensão do Embedding:** Dimensões maiores capturam mais nuances do idioma, mas expandem a memória necessária para armazenar tensores e os pesos da rede.
* **Função do DataLoader:** Gerencia o empacotamento em lotes (*batches*), o embaralhamento e a iteração eficiente de memória na GPU/CPU.
* **Conexão com a Próxima Sprint:** O tensor resultante `[Batch, Contexto, Dimensão]` é a entrada direta para o mecanismo de *Self-Attention* da Sprint 3.

## 2. Análise dos Experimentos Realizados

Com base nos testes executados no ambiente, foram observados os seguintes comportamentos:

1. **Impacto do Contexto (Contexto 2 vs Contexto 4):**
   * Com contexto 2, foram geradas **11 amostras**; com contexto 4, a quantidade caiu para **9 amostras**. Isso comprova que contextos mais longos consomem mais janela do texto base para formar um único bloco.
2. **Impacto do Batch Size (Batch 2 vs Batch 4):**
   * O lote altera a primeira dimensão do tensor final (`torch.Size([2, 4, 16])` para `torch.Size([4, 4, 16])`). Lotes maiores otimizam o paralelismo do hardware, porém exigem mais memória VRAM.
3. **Impacto da Dimensão do Embedding (Dim 16 vs Dim 64):**
   * O tensor final expandiu sua terceira dimensão de `16` para `64` (`torch.Size([2, 4, 64])`). Isso quadruplica o tamanho ocupado pelo tensor em memória.