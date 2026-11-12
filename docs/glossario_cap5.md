# Glossário — Capítulo 5: treinamento e avaliação

Os glossários dos capítulos anteriores estão em `docs/glossario_cap1.md` a `docs/glossario_cap4.md`; juntos com este arquivo formam o glossário cumulativo.

## Cross-entropy loss — perda de entropia cruzada
**Definição:** média do negativo do logaritmo natural da probabilidade atribuída ao token correto. **Função:** medir o erro das previsões do GPT e fornecer o objetivo a minimizar. **Relação:** aplica log-softmax aos logits e compara com os alvos deslocados; a perplexidade é a exponencial da média. **Exemplo:** probabilidade 0,5 para o alvo resulta em perda −ln(0,5) ≈ 0,693.

## Perplexity — perplexidade
**Definição:** exponencial da perda média por token. **Função:** expressar a incerteza média na previsão, com valores menores melhores na mesma avaliação. **Relação:** depende da cross-entropy e da tokenização, de modo que valores com tokenizadores diferentes não são diretamente comparáveis. **Exemplo:** perda ln(10) corresponde a perplexidade 10.

## Training and validation sets — conjuntos de treinamento e validação
**Definição:** trechos usados, respectivamente, para ajustar pesos e avaliar generalização durante o desenvolvimento. **Função:** acompanhar aprendizado e detectar sobreajuste. **Relação:** a validação não participa da retropropagação; separar antes de criar janelas evita sua sobreposição entre conjuntos. **Exemplo:** primeiros 80% dos caracteres para treino, restantes para validação.

## Batch and epoch — lote e época
**Definição:** lote é um conjunto de exemplos processados em uma atualização; época é uma passagem pelo conjunto de treino. **Função:** organizar o cálculo de gradientes e a duração do treinamento. **Relação:** tamanho do lote e número de exemplos determinam quantas atualizações ocorrem por época. **Exemplo:** 100 janelas em lotes de 32 produzem quatro atualizações por época, incluindo o lote incompleto.

## Backpropagation and gradient — retropropagação e gradiente
**Definição:** cálculo, pela regra da cadeia, das derivadas da perda em relação aos pesos. **Função:** indicar como alterar os parâmetros para reduzir o erro. **Relação:** `loss.backward()` calcula gradientes e o otimizador os utiliza; zerá-los evita acumulação involuntária. **Exemplo:** `zero_grad`, `backward`, `step`.

## Optimizer and learning rate — otimizador e taxa de aprendizado
**Definição:** o otimizador aplica atualizações aos pesos; a taxa de aprendizado controla sua escala. **Função:** reduzir a perda ao longo dos lotes. **Relação:** AdamW usa estimativas adaptativas de momentos e aplica decaimento de pesos desacoplado. **Exemplo:** AdamW com `lr=0.001` e `weight_decay=0.01`.

## Overfitting — sobreajuste
**Definição:** adaptação excessiva aos exemplos de treino, prejudicando a generalização. **Função na avaliação:** indica quando continuar ajustando pesos deixa de melhorar dados não usados nas atualizações. **Relação:** pode aparecer como perda de treino em queda enquanto a de validação sobe; dropout e mais dados podem mitigá-lo. **Exemplo:** escolher o checkpoint com menor perda de validação em vez do último.

## Decoding, temperature and top-k — decodificação, temperatura e top-k
**Definição:** regras para selecionar o próximo token a partir dos logits. Greedy escolhe o maior; amostragem sorteia conforme probabilidades; temperatura ajusta a distribuição e top-k restringe candidatos. **Função:** controlar as saídas na inferência. **Relação:** não alteram os pesos treinados; temperatura menor tende a concentrar probabilidades. **Exemplo:** top-k=10 limita a escolha aos dez maiores logits, salvo empates.

## Checkpoint — ponto de salvamento
**Definição:** arquivo com estado do modelo e informações necessárias para restaurá-lo. **Função:** preservar pesos e permitir inferência posterior. **Relação:** configuração e vocabulário devem corresponder aos pesos; para retomar o treinamento também é necessário o estado do otimizador e, para reprodução exata, os estados aleatórios e do carregador. **Exemplo:** salvar `state_dict` e configuração em arquivo `.pt`.
