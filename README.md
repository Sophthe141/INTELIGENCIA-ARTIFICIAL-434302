# INTELIGENCIA-ARTIFICIAL-434302
Projeto de faculdade do componente curricular INTELIGÊNCIA ARTIFICIAL E SISTEMAS INTELIGENTES visando a construção de um Large Language Model (LLM).

## Descrição do Projeto
Este repositório contém o desenvolvimento passo a passo de um Large Language Model (LLM) criado do zero, como parte do Projeto Integrador da disciplina de Inteligência Artificial. O objetivo é compreender e implementar a arquitetura interna de modelos de linguagem, passando pela preparação de dados, mecanismo de atenção, pré-treinamento e fine-tuning.

## Estrutura do Repositório
* `src/`: Código-fonte principal do modelo.
* `notebooks/`: Notebooks Jupyter para experimentação e validação de ideias.
* `docs/`: Documentação técnica e glossários.
* `experimentos/`: Logs e resultados de treinamentos.
* `relatorios/`: Relatórios de entrega de cada Sprint.

## Ambiente de Desenvolvimento
O projeto utiliza **Python** e a biblioteca **PyTorch**. 
Para reproduzir o ambiente:
1. Crie um ambiente virtual: `python -m venv venv`
2. Ative o ambiente: `source venv/bin/activate` (Linux/Mac) ou `venv\Scripts\activate` (Windows)
3. Instale as dependências: `pip install torch jupyter`

## 🚀 Sprint 3 — Mecanismos de Atenção (Concluída)

Nesta etapa, implementamos e validamos o núcleo do mecanismo de atenção utilizado na arquitetura Transformer/GPT:

* **Implementações em PyTorch (`notebooks/sprint3.ipynb`):**
  * Módulo de `SelfAttention` com parâmetros treináveis.
  * Módulo de `CausalAttention` com escalonamento ($\sqrt{d_k}$) e máscara triangular causal.
  * Módulo de `MultiHeadAttention` com projeções paralelas em subespaços e recombinação linear.
* **Experimentos e Documentação (`experimentos/`):**
  * `experimento1_escala.md`: Demonstração gráfica do impacto de $\sqrt{d_k}$ na prevenção de saturação da Softmax.
  * `experimento2_causal.md`: Validação da máscara autoregressiva para impedir vazamento de tokens futuros.
  * `experimento3_multihead.md`: Análise comparativa da especialização de diferentes cabeças de atenção.
* **Glossário:** Atualizado com os conceitos do Capítulo 3 em `docs/glossario_cap3.md`.
