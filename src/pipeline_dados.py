import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Dataset para Treinamento
class GPTDataset(Dataset):
    def __init__(self, x_lista, y_lista):
        self.x = torch.tensor(x_lista)
        self.y = torch.tensor(y_lista)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. Pipeline Completo de Processamento
def pipeline_dados(texto, tamanho_contexto=4, batch_size=2, dimensao_emb=16):
    # Tokenização
    tokens = [item for item in re.split(r'([,.:;?_!"()\']|--|\s)', texto) if item.strip()]
    
    # Vocabulário e IDs
    tokens_unicos = sorted(set(tokens))
    tokens_unicos.extend(["<|unk|>", "<|endoftext|>"])
    vocab = {token: id_num for id_num, token in enumerate(tokens_unicos)}
    
    token_ids = [vocab.get(t, vocab["<|unk|>"]) for t in tokens]

    # Sequências (Entradas e Alvos)
    entradas, alvos = [], []
    for i in range(len(token_ids) - tamanho_contexto):
        entradas.append(token_ids[i : i + tamanho_contexto])
        alvos.append(token_ids[i + 1 : i + tamanho_contexto + 1])

    # DataLoader
    dataset = GPTDataset(entradas, alvos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Embeddings
    camada_emb_token = nn.Embedding(len(vocab), dimensao_emb)
    camada_emb_posicao = nn.Embedding(tamanho_contexto, dimensao_emb)

    lote_x, lote_y = next(iter(dataloader))
    
    emb_tokens = camada_emb_token(lote_x)
    emb_posicoes = camada_emb_posicao(torch.arange(tamanho_contexto))
    
    entrada_final = emb_tokens + emb_posicoes
    return entrada_final, dataloader

if __name__ == "__main__":
    texto_teste = "O modelo de linguagem aprende a prever a próxima palavra e gerar texto."
    tensor_saida, _ = pipeline_dados(texto_teste)
    print(f"Pipeline executado com sucesso! Shape de saída: {tensor_saida.shape}")