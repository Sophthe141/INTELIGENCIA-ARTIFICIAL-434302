# Glossário - Capítulo 2: Working with Text Data

### 1. Token
* **Tradução:** Token (mantido no original) ou Símbolo.
* **Definição:** A menor unidade de texto em que uma string é dividida durante o pré-processamento.
* **Função no modelo:** Serve como a peça fundamental de informação que o modelo vai processar, já que redes neurais não leem frases inteiras de uma vez.
* **Relação com outros conceitos:** É o produto da etapa de *Tokenização* e a entrada para a geração de *Token IDs*.
* **Exemplo conceitual:** Na frase "Eu amo IA", os tokens poderiam ser ["Eu", "amo", "IA"].

### 2. Vocabulary
* **Tradução:** Vocabulário.
* **Definição:** Um dicionário (ou conjunto) que mapeia todos os tokens únicos conhecidos pelo modelo para números inteiros específicos.
* **Função no modelo:** Garantir que o modelo tenha uma referência padronizada de quais "palavras" ele conhece e como elas são traduzidas para o mundo numérico.
* **Relação com outros conceitos:** Faz a ponte bidirecional entre *Tokens* e *Token IDs*.
* **Exemplo computacional:** `{"Eu": 0, "amo": 1, "IA": 2, "<|unk|>": 3}`.

### 3. Token ID
* **Tradução:** Identificador de Token.
* **Definição:** Um número inteiro único que representa um token específico com base no vocabulário.
* **Função no modelo:** Converter dados textuais (strings) em dados numéricos discretos (inteiros) que o computador consegue armazenar em tensores iniciais.
* **Relação com outros conceitos:** É o passo intermediário entre o *Token* e o *Embedding*.
* **Exemplo computacional:** O token "IA" se transforma no Token ID `2`.

### 4. Embedding (Word Embedding)
* **Tradução:** Incorporação (raramente traduzido) ou Vetor de Incorporação.
* **Definição:** Uma representação vetorial densa e contínua de um Token ID, onde números reais capturam o significado semântico do token.
* **Função no modelo:** Transformar inteiros isolados em um espaço matemático onde tokens com significados parecidos fiquem próximos, permitindo que a rede neural faça cálculos e identifique padrões de linguagem.
* **Relação com outros conceitos:** Substitui os *Token IDs* e é somado aos *Positional Embeddings*.
* **Exemplo conceitual:** O Token ID `2` vira um vetor de 3 dimensões: `[0.34, -1.20, 0.88]`.

### 5. Positional Embedding
* **Tradução:** Incorporação Posicional.
* **Definição:** Um vetor adicionado ao embedding original de um token para indicar qual é a sua posição exata na frase.
* **Função no modelo:** Como a arquitetura Transformer processa todos os tokens de uma vez (em paralelo) e não sequencialmente, ela perderia a ordem das palavras. O positional embedding resolve isso injetando a noção de ordem.
* **Relação com outros conceitos:** É somado ao *Embedding* de palavra antes dos dados entrarem nas camadas de atenção.
* **Exemplo conceitual:** Diferenciar a palavra "cachorro" na frase "O cachorro mordeu o homem" da frase "O homem mordeu o cachorro".

### 6. DataLoader
* **Tradução:** Carregador de Dados.
* **Definição:** Um utilitário de software (muito comum no PyTorch) que iterativamente agrupa os dados em lotes (batches).
* **Função no modelo:** Fornecer amostras de treinamento de forma eficiente para a GPU/CPU, gerenciando o tamanho do lote e embaralhando os dados.
* **Relação com outros conceitos:** Pega as *Sequências de treinamento* (entradas e alvos) criadas e as entrega para o modelo processar.
