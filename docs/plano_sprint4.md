# Plano de Ação — Sprint 4 (Arquitetura GPT e Geração)

Este documento foi gerado para registrar o estado de transição e permitir continuidade imediata entre sessões e contas.

---

## 📌 Estado Atual do Projeto

* **Disciplina:** Inteligência Artificial (Unoesc)
* **Objetivo:** Construção de um Large Language Model (LLM) do zero baseado no livro *Build a Large Language Model (From Scratch)*.
* **Sprints Concluídas:**
  * **Sprint 0:** Preparação de ambiente e repositório.
  * **Sprint 2:** Pipeline de dados em `src/pipeline_dados.py` (tokenização BPE, embeddings, DataLoader).
  * **Sprint 3:** Mecanismos de Atenção em `src/atencao.py` (`SelfAttention`, `CausalAttention`, `MultiHeadAttention`), experimentos 1, 2 e 3 e `docs/glossario_cap3.md`.
* **Pendente da Sprint 3:** O arquivo `src/atencao.py` está funcional e testado, aguardando inclusão/commit no Git.

---

## 🚀 Sprint 4 — Entregáveis (Capítulo 4 — Prazo: 29/10)

1. **Glossário do Capítulo 4 (`docs/glossario_cap4.md`):**
   - Conceitos: Transformer, Transformer Block, LayerNorm, Conexões Residuais, FeedForward Network, GELU, Dropout, Logits, Language Model Head, Inferência e Geração Autoregressiva.
2. **Implementação do Transformer Block (`src/modelo_gpt.py`):**
   - `LayerNorm`: Normalização das ativações por amostra.
   - `GELU`: Ativação não linear suave.
   - `FeedForward`: Expansão e contração de dimensões ($d_{model} \to 4 \times d_{model} \to d_{model}$).
   - `TransformerBlock`: Integração Pre-LayerNorm com `MultiHeadAttention` e `FeedForward` via conexões residuais (skip connections).
3. **Construção do Modelo GPT (`GPTModel`):**
   - Entrada de token IDs $\to$ Embeddings de tokens + Embeddings de posição.
   - Empilhamento de $N$ blocos Transformer.
   - Camada final de normalização (`LayerNorm`).
   - Camada Linear final (`out_head`) projetando para `vocab_size` (logits).
4. **Inferência e Geração Autoregressiva:**
   - Loop autoregressivo: sequência inicial $\to$ logits $\to$ softmax $\to$ próximo token $\to$ concatenação ao contexto.
   - Implementação de amostragem gananciosa (*greedy*) e exploração com **Temperatura** e **Top-K**.
5. **Notebook de Experimentos (`notebooks/sprint4.ipynb`):**
   - Variação de hiperparâmetros (número de blocos, cabeças de atenção, dimensão de embedding).
   - Teste de geração e análise do custo computacional.

---

## 🏁 Checklist para a Próxima Sessão

- [ ] Commitar `src/atencao.py` no Git.
- [ ] Criar `docs/glossario_cap4.md`.
- [ ] Implementar `src/modelo_gpt.py`.
- [ ] Criar `notebooks/sprint4.ipynb` com testes de inferência e geração de texto.
