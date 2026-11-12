# Sprint 5 — treinamento e avaliação do GPT

Os resultados foram refeitos após corrigir os glossários e alinhar o dropout da rede feed-forward ao capítulo 4 do livro.

## Dados e método

O experimento usa os glossários dos capítulos 1 a 4 deste repositório, concatenados nessa ordem. São 18.193 caracteres (SHA-256 `ec94b67318c3b27f0f27d193996198a3749e159874afeff04a7af21343c5de4d`). Os primeiros 80% (14.554 caracteres) formam o treino e o restante a validação, antes da construção das janelas. O tokenizador didático é **por caractere**, diferente do BPE GPT-2 do livro: vocabulário de 102 símbolos construído apenas no treino, incluindo `<unk>`; não houve caracteres desconhecidos na validação. As janelas de 32 caracteres e passo 32 geram 454 exemplos de treino e 113 de validação, com alvo deslocado em um caractere. Trechos finais insuficientes são descartados. Não há sobreposição de janelas entre conjuntos; ainda assim, ambos são glossários do mesmo projeto, portanto a validação não representa textos gerais independentes.

O modelo tem embeddings de dimensão 48, 2 blocos, 4 cabeças, contexto 32, dropout 0,1 e saída para 102 caracteres. Treinamos na CPU com lotes de 32, AdamW (weight decay 0,01), semente 123 e uma thread. A avaliação é feita em modo `eval`, sem gradientes, sobre todos os lotes; a perda é média ponderada pelo número de tokens. A entropia cruzada mede −log da probabilidade do caractere correto, e a perplexidade é `exp(loss)`, ambas por caractere. Antes do treinamento, a perda era 4,764 no treino e 4,747 na validação (perplexidades 117,17 e 115,25).

| Configuração | Épocas | Taxa | Loss treino | Loss validação | Perplexidade validação | Tempo CPU |
|---|---:|---:|---:|---:|---:|---:|
| Base | 8 | 0,001 | 2,813 | 2,925 | 18,63 | 6,5 s |
| Taxa menor | 8 | 0,0002 | 3,422 | 3,461 | 31,84 | 6,6 s |
| Mais épocas | 16 | 0,001 | 2,491 | 2,699 | 14,86 | 13,4 s |

Os tempos incluem avaliação inicial e por época e dependem do computador. Mantida a inicialização, a taxa menor convergiu mais lentamente em oito épocas; isso não prova que seja inferior com mais atualizações. Dobrar as épocas reduziu ambas as perdas e aproximadamente dobrou o tempo. A diferença entre treino e validação aumentou (0,111 na base e 0,207 em 16 épocas), mas a perda de validação também caiu: há uma lacuna de generalização, sem evidência suficiente de sobreajuste crescente com perda de validação subindo. Históricos completos, parâmetros, tempos e versões estão em `experimentos/sprint5_resultados.json`.

## Decodificação e checkpoint

As amostras abaixo usam o **mesmo modelo de 16 épocas**, prompt `O modelo de linguagem`, 100 caracteres novos e semente 456 para os sorteios. Alterar a decodificação não altera os pesos. As saídas completas constam no JSON.

- Greedy: `O modelo de linguagema de de de de de de de de de...` — repetição forte.
- Amostragem com temperatura 1: `O modelo de linguagemos me altincores pêI**** pontelomo...` — maior variedade e muitos trechos sem sentido.
- Temperatura 0,7: `O modelo de linguagemos te altincores pê **** dontelomo...` — sequência diferente, ainda pouco coerente.
- Temperatura 0,7 e top-k 10: `O modelo de linguagemos te altincores petemo dententemo...` — candidatos mais restritos, mas sem garantir coerência.

São amostras individuais, não uma medida estatística de qualidade. A tarefa por caractere, o corpus pequeno e especializado, o contexto curto e os poucos passos explicam os limites observados; o resultado não é comparável diretamente à perplexidade por token BPE do livro.

O script salva `dataset/sprint5_checkpoint.pt`, ignorado pelo Git, contendo pesos, configuração, vocabulário, estado do AdamW e metadados. Ele carrega esse arquivo, compara os logits exatamente com os do modelo salvo e gera as amostras usando o modelo recuperado. Para continuar o treinamento, recrie AdamW e carregue `dados['otimizador']` com `load_state_dict`; a retomada exata exigiria também estados aleatórios e posição do embaralhamento, que este checkpoint não armazena.

## Reprodução

Na raiz do projeto, com as dependências instaladas:

```bash
./venv/bin/python -m src.experimento_sprint5
./venv/bin/python -m unittest discover -s tests -v
```

O primeiro comando recria o checkpoint e sobrescreve o JSON de resultados; tempos e pequenas diferenças numéricas podem variar conforme ambiente. A execução registrada usou Python 3.14.4 e PyTorch 2.13.0+cu130 em CPU. Para inferência independente, use `carregar_checkpoint`, reconstrua o mapa de caracteres a partir de `dados['caracteres']`, codifique o prompt, chame `generate_text_advanced` e decodifique os IDs pela mesma lista.
