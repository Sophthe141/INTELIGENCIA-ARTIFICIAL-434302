"""Experimentos reproduzíveis, executados na CPU com os glossários do projeto."""
import hashlib
import json
import platform
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.modelo_gpt import GPTModel, generate_text_advanced
from src.treinamento import Janelas, treinar, salvar_checkpoint, carregar_checkpoint


def main():
    torch.set_num_threads(1)
    texto = '\n'.join(Path(f'docs/glossario_cap{i}.md').read_text(encoding='utf-8') for i in range(1, 5))
    # O vocabulário usa apenas caracteres de treino; os desconhecidos viram <unk>.
    corte = int(len(texto) * .8)
    caracteres = ['<unk>'] + sorted(set(texto[:corte]))
    vocab = {c: i for i, c in enumerate(caracteres)}
    codificar = lambda s: [vocab.get(c, 0) for c in s]
    contexto, lote = 32, 32
    treino = Janelas(codificar(texto[:corte]), contexto, passo=32)
    validacao = Janelas(codificar(texto[corte:]), contexto, passo=32)
    cfg = dict(vocab_size=len(caracteres), context_length=contexto, emb_dim=48,
               n_heads=4, n_layers=2, drop_rate=0.1, qkv_bias=False)
    configs = [('base', 0.001, 8), ('lr_menor', 0.0002, 8), ('mais_epocas', 0.001, 16)]
    resultado = {'corpus_sha256': hashlib.sha256(texto.encode()).hexdigest(),
                 'fontes': [f'docs/glossario_cap{i}.md' for i in range(1, 5)],
                 'corte_caracteres': corte, 'total_caracteres': len(texto),
                 'caracteres_desconhecidos_validacao': sum(c not in vocab for c in texto[corte:]),
                 'janelas_treino': len(treino), 'janelas_validacao': len(validacao),
                 'passo': 32, 'batch_size': lote, 'config_modelo': cfg,
                 'semente': 123, 'dispositivo': 'cpu', 'torch': torch.__version__,
                 'python': platform.python_version(), 'experimentos': []}
    for nome, lr, epocas in configs:
        torch.manual_seed(123)
        modelo = GPTModel(cfg)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
        loader = DataLoader(treino, batch_size=lote, shuffle=True, generator=torch.Generator().manual_seed(123))
        val_loader = DataLoader(validacao, batch_size=lote, shuffle=False)
        inicio = time.perf_counter()
        historico = treinar(modelo, loader, val_loader, otimizador, epocas, 'cpu')
        tempo = time.perf_counter() - inicio
        item = {'nome': nome, 'lr': lr, 'epocas': epocas, 'weight_decay': 0.01,
                'segundos': tempo, 'historico': historico}
        resultado['experimentos'].append(item)
        print(nome, round(tempo, 1), historico[-1], flush=True)
        if nome == 'mais_epocas':
            caminho = 'dataset/sprint5_checkpoint.pt'
            salvar_checkpoint(caminho, modelo, otimizador, caracteres, {'experimento': nome, 'epocas': epocas, 'lr': lr})
            carregado, _ = carregar_checkpoint(caminho)
            x = torch.tensor([codificar('O modelo de linguagem')])
            modelo.eval()
            with torch.no_grad():
                assert torch.equal(modelo(x), carregado(x))
            saidas = {}
            for titulo, temperatura, top_k in [('greedy', 0.0, None), ('amostragem', 1.0, None), ('temperatura_baixa', 0.7, None), ('top_k', 0.7, 10)]:
                torch.manual_seed(456)
                ids = generate_text_advanced(carregado, x, 100, contexto, temperatura, top_k)[0].tolist()
                saidas[titulo] = ''.join(caracteres[i] for i in ids)
            item['geracoes'] = saidas
            item['checkpoint_inferencia_identica'] = True
    Path('experimentos/sprint5_resultados.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
