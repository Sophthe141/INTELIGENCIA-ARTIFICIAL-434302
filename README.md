# INTELIGENCIA-ARTIFICIAL-434302

Projeto de faculdade do componente curricular **INTELIGÊNCIA ARTIFICIAL E SISTEMAS INTELIGENTES** visando a construção de um Large Language Model (LLM) do zero.

## Descrição do Projeto

Este repositório contém o desenvolvimento passo a passo de um Large Language Model (LLM) criado do zero, como parte do Projeto Integrador da disciplina de Inteligência Artificial da **Universidade do Oeste de Santa Catarina (Unoesc)**. O objetivo é compreender e implementar a arquitetura interna de modelos de linguagem baseados na arquitetura GPT, passando pela preparação de dados, mecanismos de atenção, construção do Transformer Block e geração autoregressiva de texto.

A referência principal do projeto é o livro *"Build a Large Language Model (From Scratch)"* — Sebastian Raschka.

## Estrutura do Repositório

```text
INTELIGENCIA-ARTIFICIAL-434302/
├── README.md                # Descrição pública do projeto
├── requirements.txt         # Dependências (PyTorch, Jupyter, Matplotlib, etc.)
│
├── src/                     # Código-fonte modular e reutilizável
│   ├── pipeline_dados.py    # Sprint 2: Tokenização, vocabulário, embeddings e DataLoader
│   ├── atencao.py           # Sprint 3: SelfAttention, CausalAttention, MultiHeadAttention
│   └── modelo_gpt.py        # Sprint 4: LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel
│
├── notebooks/               # Notebooks de experimentação e análise
│   ├── 00_teste_pytorch.ipynb
│   ├── sprint2.ipynb        # Pipeline de dados e embeddings
│   ├── sprint3.ipynb        # Mecanismos de atenção
│   └── sprint4.ipynb        # Arquitetura GPT e inferência autoregressiva
│
├── docs/                    # Entregáveis oficiais da disciplina
│   ├── glossario_cap1.md    # Sprint 0
│   ├── glossario_cap2.md    # Sprint 2
│   ├── glossario_cap3.md    # Sprint 3
│   └── glossario_cap4.md    # Sprint 4
│
├── experimentos/            # Gráficos e resultados gerados pelos notebooks
└── relatorios/              # Relatórios de entrega de cada Sprint
```

## Ambiente de Desenvolvimento

O projeto utiliza **Python** e a biblioteca **PyTorch**.  
Para reproduzir o ambiente:

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install torch jupyter matplotlib
```

---

## Linha do Tempo das Sprints

### ✅ Sprint 0 — Ambiente e Repositório

* Configuração do repositório Git e ambiente virtual Python com PyTorch.
* Leitura do Capítulo 1 (*Understanding Large Language Models*) e do Apêndice de PyTorch.
* **Entregável:** `docs/glossario_cap1.md`

---

### ✅ Sprint 2 — Pipeline de Dados e Embeddings (Capítulo 2)

Implementação do fluxo completo de pré-processamento de texto para LLMs:

```
Texto → Tokenização → Token IDs → Sequências de treinamento → Embeddings → Positional Embeddings → DataLoader
```

* **Código modular:** `src/pipeline_dados.py`
  * Tokenização BPE, construção de vocabulário, conversão para Token IDs.
  * Embeddings de tokens e Positional Embeddings.
  * `DataLoader` para organização de lotes de treinamento.
* **Experimentos:** `notebooks/sprint2.ipynb`
* **Entregáveis:** `docs/glossario_cap2.md`, `relatorios/analise_sprint2.md`

---

### ✅ Sprint 3 — Mecanismos de Atenção (Capítulo 3)

Implementação e validação do núcleo do mecanismo de atenção da arquitetura Transformer/GPT:

* **Código modular:** `src/atencao.py`
  * `SelfAttention`: Projeções lineares $W_q, W_k, W_v$ e produto escalar de atenção.
  * `CausalAttention`: Scaled dot-product ($\div \sqrt{d_k}$), máscara autoregressiva triangular e dropout.
  * `MultiHeadAttention`: Atenção multi-cabeça paralelizada com projeção de saída linear.
* **Notebook:** `notebooks/sprint3.ipynb`
* **Experimentos (`experimentos/`):**
  * `experimento1-escala.md` + `grafico_escala.png`: Impacto de $\sqrt{d_k}$ na prevenção de saturação da Softmax.
  * `experimento2_causal.md` + `matriz_causal.png`: Validação da máscara autoregressiva.
  * `experimento3_multihead.md` + `grafico_multihead.png`: Especialização das cabeças de atenção.
* **Entregável:** `docs/glossario_cap3.md`

---

### ✅ Sprint 4 — Arquitetura GPT e Geração Autoregressiva (Capítulo 4)

Integração de todos os componentes anteriores em uma arquitetura GPT funcional, com geração autoregressiva de texto:

* **Código modular:** `src/modelo_gpt.py`
  * `LayerNorm`: Normalização das ativações por amostra (parâmetros treináveis $\gamma$ e $\beta$).
  * `GELU`: Gaussian Error Linear Unit — ativação não-linear suave.
  * `FeedForward`: Rede feed-forward com expansão $4 \times d_{model}$.
  * `TransformerBlock`: Pre-LayerNorm com `MultiHeadAttention` + `FeedForward` via conexões residuais.
  * `GPTModel`: Pilha de $N$ blocos Transformer com embeddings de tokens e posição, LayerNorm final e `out_head` projetando para o vocabulário (logits).
  * `generate_text_simple` / `generate_text_advanced`: Loop autoregressivo com suporte a **Temperatura** e **Top-K**.
* **Notebook:** `notebooks/sprint4.ipynb`
* **Experimentos (`experimentos/`):**
  * `experimento4_parametros.md` + `grafico_parametros_sprint4.png`: Análise de hiperparâmetros da arquitetura.
  * `experimento4_layernorm.md` + `grafico_layernorm_sprint4.png`: Impacto do Layer Normalization.
  * `experimento4_geracao.md` + `grafico_geracao_temp_topk.png`: Efeito de Temperatura e Top-K na geração.
* **Entregáveis:** `docs/glossario_cap4.md`, `src/modelo_gpt.py`, `notebooks/sprint4.ipynb`
