# Glossário - Capítulo 4: Arquitetura GPT e Geração

Este glossário técnico reúne os principais conceitos teóricos e práticos abordados no Capítulo 4 do livro *Build a Large Language Model (From Scratch)* para a Sprint 4 da disciplina de Inteligência Artificial.

---

### 1. Transformer
* **Tradução:** Transformer (mantido no original) ou Arquitetura Baseada em Transformadores.
* **Definição:** Arquitetura de rede neural profunda introduzida no artigo *"Attention Is All You Need"* (Vaswani et al., 2017), baseada inteiramente no mecanismo de atenção, dispensando recorrência (RNNs) e convoluções (CNNs). Processa sequências inteiras em paralelo.
* **Função no modelo GPT:** Constitui o arcabouço estrutural do GPT (Generative Pre-trained Transformer). Especificamente, o GPT adota uma variante baseada no decodificador (*decoder-only*), empilhando blocos com atenção causal mascarada para a modelagem autoregressiva de linguagem.
* **Relação com outros conceitos:** Contém os *Transformer Blocks*, alimentando-se dos *Embeddings* e projetando representações latentes para a *Language Model Head*.
* **Exemplo conceitual:** Uma fábrica composta por múltiplos andares de refinamento (blocos Transformer), onde a entrada de texto bruta é progressivamente contextualizada para prever o próximo elemento.

---

### 2. Transformer Block
* **Tradução:** Bloco Transformer.
* **Definição:** Módulo estrutural repetível composto pela integração de um mecanismo de atenção multi-cabeça (*Multi-Head Attention*), uma rede feed-forward densa (*FeedForward Network*), camadas de normalização (*LayerNorm*) e conexões residuais (*Residual Connections*).
* **Função no modelo GPT:** Cada bloco recebe uma representação contextual e a refina ainda mais. O empilhamento de $N$ blocos permite que o modelo aprenda características linguísticas de diferentes níveis de abstração (sintaxe, semântica, correferência, contexto de longo alcance).
* **Relação com outros conceitos:** Utiliza a arquitetura *Pre-LayerNorm*, aplicando *LayerNorm* antes da *Multi-Head Attention* e antes da *FeedForward Network*, somando a entrada original via conexão residual.
* **Exemplo computacional:** Recebe tensor de ativações com formato `(batch_size, seq_len, emb_dim)` e devolve tensor de mesma dimensão `(batch_size, seq_len, emb_dim)`.

---

### 3. LayerNorm (Layer Normalization)
* **Tradução:** Normalização de Camada.
* **Definição:** Técnica de normalização que ajusta as ativações de cada amostra individualmente através da média zero e variância unitária ao longo da dimensão de características ($d_{model}$), aplicando em seguida dois parâmetros aprendíveis: escala ($\gamma$ ou *scale*) e deslocamento ($\beta$ ou *shift*).
* **Função no modelo GPT:** Garante estabilidade numérica ao longo das camadas profundas, acelerando a convergência do treinamento e prevenindo os problemas de explosão ou desaparecimento do gradiente (*exploding/vanishing gradients*).
* **Relação com outros conceitos:** Difere do *BatchNorm* por normalizar cada token individualmente, sendo independente do tamanho do lote (*batch size*). No GPT-2/GPT-3, adota-se a convenção *Pre-LayerNorm*, normalizando as ativações antes de passá-las para os sub-blocos.
* **Exemplo computacional:** Para um vetor $x$, calcula-se:
  $$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
  onde $\mu$ é a média e $\sigma^2$ é a variância calculadas sobre a dimensão do embedding.

---

### 4. Conexões Residuais (Residual Connections / Skip Connections)
* **Tradução:** Conexões Residuais ou Conexões de Atalho (*Skip Connections*).
* **Definição:** Estrutura introduzida nas ResNets (He et al., 2015) em que a entrada de um bloco é somada diretamente à sua saída: $y = F(x) + x$.
* **Função no modelo GPT:** Permite a propagação desimpedida do fluxo de gradientes durante a retropropagação (*backpropagation*) pelas dezenas de blocos do modelo. Evita a perda de informação contextual básica ao longo das transformações não-lineares.
* **Relação com outros conceitos:** É aplicada em dois pontos chave dentro de cada *Transformer Block*: em torno do módulo de *Multi-Head Attention* e em torno da *FeedForward Network*.
* **Exemplo conceitual:** Permitir que o sinal de entrada original atravesse uma "via expressa", enquanto a rede aprende apenas o resíduo/diferencial de ajuste $F(x)$.

---

### 5. FeedForward Network (FFN / MLP)
* **Tradução:** Rede Alimentada Adiante ou Módulo Multicamadas (MLP).
* **Definição:** Sub-rede densa composta por duas projeções lineares separadas por uma função de ativação não-linear (tipicamente *GELU*) e *Dropout*.
* **Função no modelo GPT:** Enquanto a atenção foca na troca de informações entre diferentes tokens ao longo da sequência, a rede Feed-Forward processa e transforma as representações de cada token de forma independente (*position-wise*), expandindo a capacidade de memorização e conhecimento semântico do modelo.
* **Relação com outros conceitos:** Realiza uma projeção de expansão de dimensão ($d_{model} \to 4 \times d_{model}$) seguida de contração ($4 \times d_{model} \to d_{model}$).
* **Exemplo computacional:**
  $$\text{FFN}(x) = \text{Dropout}(\text{GELU}(x W_1 + b_1)) W_2 + b_2$$

---

### 6. GELU (Gaussian Error Linear Unit)
* **Tradução:** Unidade Linear de Erro Gaussiano.
* **Definição:** Função de ativação suave e não-linear introduzida por Hendrycks & Gimpel (2016), calculada como $\text{GELU}(x) = x \cdot \Phi(x)$, onde $\Phi(x)$ é a função de distribuição cumulativa da distribuição normal padrão.
* **Função no modelo GPT:** Substitui a função *ReLU* tradicional. Permite uma transição contínua e suave para valores próximos e inferiores a zero, mantendo pequenos gradientes negativos em vez de zerá-los abruptamente (o que causaria o fenômeno de "neurônios mortos").
* **Relação com outros conceitos:** Utilizada como função de ativação intermediária dentro da *FeedForward Network*. Frequentemente aproximada computacionalmente por:
  $$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$
* **Exemplo conceitual:** Funciona como uma comporta probabilística suave: quanto maior o valor de $x$, maior a certeza de mantê-lo; para valores negativos, diminui suavemente o sinal.

---

### 7. Dropout
* **Tradução:** Dropout ou Abandono Aleatório.
* **Definição:** Técnica de regularização que desativa (zera) aleatoriamente uma fração $p$ dos neurônios/pesos durante a fase de treinamento, reescalonando os restantes por $\frac{1}{1 - p}$.
* **Função no modelo GPT:** Previne o sobreajuste (*overfitting*), impedindo que os neurônios desenvolvam co-adaptações espúrias nos dados de treino.
* **Relação com outros conceitos:** Aplicado após os *embeddings*, após a normalização dos pesos de atenção e após as conexões residuais no *Transformer Block*. Durante a inferência/geração, o Dropout deve ser explicitamente desativado (`model.eval()`).
* **Exemplo computacional:** Com taxa $p = 0.1$, aproximadamente 10% das ativações do tensor são substituídas por zero durante o treino.

---

### 8. Logits
* **Tradução:** Logits (Valores não normalizados de predição).
* **Definição:** Vetores de números reais produzidos pela última camada linear do modelo (camada de saída), representando a pontuação bruta de afinidade para cada token do vocabulário antes da aplicação de qualquer função probabilística.
* **Função no modelo GPT:** Representam a preferência do modelo pelo próximo token. Um valor mais alto indica maior probabilidade prevista para o token correspondente.
* **Relação com outros conceitos:** São transformados em distribuições de probabilidade através da função *Softmax* para cálculo da perda (*Cross-Entropy Loss*) no treinamento ou seleção do token na *Geração Autoregressiva*.
* **Exemplo computacional:** Para um vocabulário de 50.257 tokens, o modelo gera um vetor de dimensão `[50257]` com valores como `[2.45, -0.12, 11.8, ...]`.

---

### 9. Language Model Head (LM Head)
* **Tradução:** Cabeça do Modelo de Linguagem ou Projeção Linear de Saída.
* **Definição:** Camada totalmente conectada (`nn.Linear`) posicionada no topo do modelo Transformer, que mapeia a representação latente final de cada token ($d_{model}$) para o tamanho do vocabulário ($V$).
* **Função no modelo GPT:** Realiza a projeção dimensional $d_{model} \to \text{vocab\_size}$, produzindo os *Logits* para cada posição da sequência.
* **Relação com outros conceitos:** Frequentemente, em arquiteturas GPT, os pesos da *LM Head* podem compartilhar a mesma matriz dos embeddings de entrada (*weight tying*), embora possam ser mantidos separados.
* **Exemplo computacional:** Em PyTorch: `nn.Linear(emb_dim, vocab_size, bias=False)`.

---

### 10. Inferência e Geração Autoregressiva
* **Tradução:** Inferência e Geração Autoregressiva.
* **Definição:** Processo iterativo de geração de texto no qual as previsões passadas do próprio modelo são retroalimentadas como entradas para as próximas previsões.
* **Função no modelo GPT:** Viabiliza a geração contínua de texto a partir de um prompt inicial:
  1. O modelo processa o contexto inicial.
  2. Obtém os *logits* referentes à última posição.
  3. Seleciona o próximo token (via *greedy*, *temperature* ou *top-k*).
  4. Anexa o novo token ao contexto.
  5. Repete o ciclo até atingir o número máximo de tokens ou encontrar o token de parada (`<|endoftext|>`).
* **Relação com outros conceitos:** Depende fundamentalmente da *Causal Attention*, que garante que o modelo só possa olhar para os tokens anteriores durante o cálculo.
* **Exemplo conceitual:** Escrever uma frase palavra por palavra, lendo tudo o que já foi escrito até então para decidir a próxima palavra mais adequada.

---

### 11. Temperatura (Temperature Sampling)
* **Tradução:** Amostragem por Temperatura.
* **Definição:** Hiperparâmetro escalar $T > 0$ que escala os *logits* antes da aplicação da função Softmax:
  $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
* **Função no modelo GPT:** Controla a aleatoriedade e criatividade do texto gerado:
  * $T \to 0$: Aproxima-se da busca gananciosa (*greedy search*), gerando respostas previsíveis e determinísticas.
  * $T = 1.0$: Mantém a distribuição probabilística original do modelo.
  * $T > 1.0$: Suaviza a distribuição (aumenta o peso de tokens menos prováveis), aumentando a diversidade e criatividade, porém elevando o risco de alucinações ou incoerências.
* **Relação com outros conceitos:** É combinada com a amostragem categórica (`torch.multinomial`) e o filtro *Top-K*.
* **Exemplo computacional:** Se os logits forem `[2.0, 4.0]`, com $T=0.5$ tornam-se `[4.0, 8.0]` (aumentando muito a distância entre eles); com $T=2.0$ tornam-se `[1.0, 2.0]` (achatando a curva).

---

### 12. Amostragem Top-K (Top-K Sampling)
* **Tradução:** Amostragem Top-K ou Filtragem por $K$ Melhores.
* **Definição:** Técnica de truncamento na qual apenas os $K$ tokens com os maiores logits são mantidos, enquanto os logits de todos os demais tokens são substituídos por $-\infty$ antes da normalização Softmax.
* **Função no modelo GPT:** Elimina a "cauda longa" de tokens altamente improváveis e irrelevantes, garantindo que o modelo nunca selecione palavras absurdas ou desconexas durante a amostragem estocástica.
* **Relação com outros conceitos:** É aplicada logo após o escalonamento por *Temperatura* e antes da função *Softmax*.
* **Exemplo conceitual:** Se $K=50$, mesmo que o vocabulário possua 50.257 palavras, o sorteio da próxima palavra será restrito estritamente entre as 50 opções mais plausíveis.
