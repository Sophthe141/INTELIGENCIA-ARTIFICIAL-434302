"""Treinamento e avaliação autoregressiva de um GPT pequeno (Sprint 5)."""
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.modelo_gpt import GPTModel


class Janelas(Dataset):
    """Pares deslocados em um token, construídos após separar os dados."""
    def __init__(self, ids, contexto, passo=1):
        self.ids, self.contexto = ids, contexto
        self.inicios = range(0, len(ids) - contexto, passo)
        if not self.inicios:
            raise ValueError('O trecho precisa conter mais tokens que o contexto')

    def __len__(self):
        return len(self.inicios)

    def __getitem__(self, i):
        inicio = self.inicios[i]
        trecho = self.ids[inicio:inicio + self.contexto + 1]
        return torch.tensor(trecho[:-1]), torch.tensor(trecho[1:])


def perda_lote(modelo, x, y, dispositivo):
    logits = modelo(x.to(dispositivo))
    return F.cross_entropy(logits.flatten(0, 1), y.to(dispositivo).flatten())


def avaliar(modelo, loader, dispositivo):
    """Média ponderada por token; preserva o modo anterior do modelo."""
    anterior = modelo.training
    modelo.eval()
    soma, total = 0.0, 0
    try:
        with torch.no_grad():
            for x, y in loader:
                n = y.numel()
                soma += perda_lote(modelo, x, y, dispositivo).item() * n
                total += n
    finally:
        modelo.train(anterior)
    if not total:
        raise ValueError('O DataLoader está vazio')
    loss = soma / total
    return {'loss': loss, 'perplexity': math.exp(loss) if loss < 709 else float('inf')}


def treinar(modelo, treino, validacao, otimizador, epocas, dispositivo):
    historico = []
    modelo.to(dispositivo)
    def registrar(epoca):
        historico.append({'epoca': epoca, 'treino': avaliar(modelo, treino, dispositivo),
                          'validacao': avaliar(modelo, validacao, dispositivo)})
    registrar(0)
    for epoca in range(1, epocas + 1):
        modelo.train()
        for x, y in treino:
            otimizador.zero_grad(set_to_none=True)
            loss = perda_lote(modelo, x, y, dispositivo)
            loss.backward()
            otimizador.step()
        registrar(epoca)
    return historico


def salvar_checkpoint(caminho, modelo, otimizador, caracteres, metadados):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': modelo.cfg, 'modelo': modelo.state_dict(),
                'otimizador': otimizador.state_dict(), 'caracteres': caracteres,
                'metadados': metadados}, caminho)


def carregar_checkpoint(caminho, dispositivo='cpu'):
    dados = torch.load(caminho, map_location=dispositivo, weights_only=True)
    modelo = GPTModel(dados['config']).to(dispositivo)
    modelo.load_state_dict(dados['modelo'])
    modelo.eval()
    return modelo, dados


if __name__ == '__main__':
    print('Execute python -m src.experimento_sprint5 para reproduzir os experimentos.')
