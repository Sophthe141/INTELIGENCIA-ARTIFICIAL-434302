# Glossário — Sprint 6: adaptação de modelos de linguagem

Este glossário complementa `glossario_cap1.md` a `glossario_cap5.md`. Sua referência é *Build a Large Language Model (From Scratch)*, de Sebastian Raschka, capítulos 6 (classificação) e 7 (instruções), disponível em PDF na pasta `livro/`. Os exemplos abaixo são ilustrativos; não representam experimentos executados neste projeto.

## Fine-Tuning — ajuste fino
**Definição:** treinamento adicional de um modelo que já passou por pré-treinamento, usando dados relacionados a uma tarefa ou comportamento desejado. **Função:** adaptar parâmetros previamente aprendidos. **Relação:** pode ser realizado para classificação ou para seguir instruções; depende de pesos pré-treinados. **Exemplo:** ajustar um GPT para identificar mensagens de spam.

## Pretrained Weights — pesos pré-treinados
**Definição:** valores dos parâmetros obtidos em uma etapa anterior de treinamento. **Função:** fornecer ao modelo um ponto de partida que já incorpora padrões aprendidos dos dados. **Relação:** o carregamento exige correspondência entre arquitetura, dimensões, parâmetros e tokenizador; pesos aleatórios ainda não carregam esse aprendizado. **Exemplo:** carregar os pesos de um GPT antes de adaptá-lo a mensagens rotuladas.

## Transfer Learning — aprendizado por transferência
**Definição:** aproveitamento do conhecimento adquirido em uma tarefa ou conjunto de dados para outra tarefa. **Função:** reduzir a necessidade de aprender tudo novamente a partir de parâmetros aleatórios. **Relação:** o fine-tuning é uma forma de transferência que ajusta parte ou todos os parâmetros pré-treinados. **Exemplo:** usar representações de um modelo de linguagem para classificar mensagens.

## Classification Fine-Tuning — ajuste fino para classificação
**Definição:** adaptação supervisionada para escolher entre classes predefinidas. **Função:** produzir uma categoria para cada entrada, como spam ou não spam. **Relação:** utiliza exemplos rotulados, uma cabeça de classificação e uma perda calculada a partir dos logits de classe. **Exemplo:** treinar com mensagens e seus rótulos correspondentes.

## Instruction Fine-Tuning — ajuste fino para instruções
**Definição:** adaptação supervisionada com exemplos de instruções e respostas desejadas. **Função:** melhorar a capacidade de responder a solicitações em linguagem natural. **Relação:** mantém uma tarefa de previsão de tokens, mas treina com textos formatados como instrução, entrada opcional e resposta. **Exemplo:** aprender a responder a uma instrução de tradução com o texto traduzido.

## Classification Head — cabeça de classificação
**Definição:** camada de saída que projeta uma representação do modelo em pontuações para classes. **Função:** converter a representação contextual em logits de classificação. **Relação:** substitui a projeção de saída para o vocabulário na adaptação descrita no capítulo 6; nesse exemplo usa-se a representação da última posição para classificar a mensagem. **Exemplo:** uma camada linear com duas saídas para spam e não spam.

## Frozen Layers — camadas congeladas
**Definição:** camadas cujos parâmetros não são atualizados durante uma etapa de ajuste. **Função:** preservar parte dos pesos pré-treinados e reduzir a quantidade de parâmetros ajustados. **Relação:** em PyTorch, `requires_grad=False` impede o cálculo de gradientes para esses parâmetros; congelamento de parâmetros e modo de avaliação são controles distintos. **Exemplo:** congelar os blocos iniciais e treinar a cabeça de classificação.

## Trainable Parameters — parâmetros treináveis
**Definição:** parâmetros habilitados a receber gradientes e ser atualizados pelo otimizador. **Função:** permitir que o modelo se adapte aos novos dados. **Relação:** dependem da política de congelamento e devem ser incluídos no otimizador. **Exemplo:** parâmetros da nova cabeça e de um bloco Transformer descongelado.

## Supervised Fine-Tuning — ajuste fino supervisionado
**Definição:** ajuste usando exemplos que fornecem a saída esperada para cada entrada. **Função:** orientar as previsões do modelo por meio de uma perda calculada em relação aos alvos. **Relação:** pode usar rótulos de classe ou respostas textuais; a perda e o formato do alvo variam conforme a tarefa. **Exemplo:** pares de mensagem e rótulo ou de instrução e resposta.

## Instruction Dataset — conjunto de dados de instruções
**Definição:** coleção de exemplos com instruções, respostas esperadas e, quando necessário, entradas adicionais. **Função:** fornecer exemplos para ensinar o comportamento desejado. **Relação:** precisa ser formatado e tokenizado antes de compor lotes; deve ser dividido para treinamento e avaliação. **Exemplo:** instrução “resuma”, entrada com um parágrafo e resposta com seu resumo.

## Training Batch — lote de treinamento
**Definição:** grupo de exemplos processados em conjunto em uma etapa de treinamento. **Função:** organizar o cálculo da perda e dos gradientes para atualização dos pesos. **Relação:** sequências de comprimentos diferentes podem exigir preenchimento (*padding*), truncamento e tratamento apropriado dos alvos na perda. **Exemplo:** oito mensagens tokenizadas e seus oito rótulos.

## Classification Accuracy — acurácia de classificação
**Definição:** proporção de previsões corretas entre os exemplos avaliados. **Função:** medir quantas classificações coincidem com seus rótulos. **Relação:** é calculada sobre um conjunto de avaliação; em dados desbalanceados, precisão, revocação e F1 também podem ser úteis. **Exemplo:** 90 acertos em 100 mensagens resultam em acurácia de 90%.

## Evaluation Dataset — conjunto de avaliação
**Definição:** exemplos reservados para medir o desempenho do modelo sem atualizar seus parâmetros. **Função:** estimar o comportamento em dados não usados nas atualizações. **Relação:** validação auxilia escolhas durante o desenvolvimento; teste independente é usado para avaliação final. **Exemplo:** mensagens rotuladas separadas antes do treinamento.

## Response Evaluation — avaliação de respostas
**Definição:** análise das respostas geradas para verificar sua adequação às instruções. **Função:** examinar se o modelo produz conteúdo útil e compatível com a tarefa. **Relação:** a perda de validação não mede sozinha todos os aspectos da qualidade; o capítulo 7 apresenta avaliação por pessoas e por outro modelo, além de comparações com respostas esperadas. **Exemplo:** avaliar se uma resposta atende ao pedido e preserva os fatos fornecidos.
