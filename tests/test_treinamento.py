import math
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.modelo_gpt import GPTModel
from src.treinamento import Janelas, avaliar, perda_lote, treinar, salvar_checkpoint, carregar_checkpoint


class TestTreinamento(unittest.TestCase):
    def test_janelas_e_loss(self):
        dados = Janelas([0, 1, 2, 3, 4], 2, 2)
        self.assertEqual(len(dados), 2)
        self.assertEqual(dados[1][0].tolist(), [2, 3])
        self.assertEqual(dados[1][1].tolist(), [3, 4])
        cfg = dict(vocab_size=5, context_length=2, emb_dim=4, n_heads=1, n_layers=1, drop_rate=0.0)
        modelo = GPTModel(cfg)
        loader = DataLoader(dados, batch_size=1)
        resultado = avaliar(modelo, loader, 'cpu')
        self.assertTrue(modelo.training)
        self.assertAlmostEqual(resultado['perplexity'], math.exp(resultado['loss']))
        x, y = next(iter(loader))
        esperado = torch.nn.functional.cross_entropy(modelo(x).reshape(-1, 5), y.reshape(-1))
        self.assertTrue(torch.allclose(perda_lote(modelo, x, y, 'cpu'), esperado))

    def test_treina_e_checkpoint(self):
        torch.manual_seed(123)
        cfg = dict(vocab_size=4, context_length=3, emb_dim=8, n_heads=2, n_layers=1, drop_rate=0.0)
        modelo = GPTModel(cfg)
        loader = DataLoader(Janelas([0, 1, 2, 3] * 8, 3), batch_size=8)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=.01)
        historico = treinar(modelo, loader, loader, otimizador, 3, 'cpu')
        self.assertLess(historico[-1]['treino']['loss'], historico[0]['treino']['loss'])
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / 'modelo.pt'
            salvar_checkpoint(caminho, modelo, otimizador, ['a', 'b', 'c', 'd'], {'epocas': 3})
            recuperado, dados = carregar_checkpoint(caminho)
            modelo.eval()
            x, _ = next(iter(loader))
            self.assertTrue(torch.equal(modelo(x), recuperado(x)))
            self.assertEqual(dados['metadados']['epocas'], 3)
            self.assertTrue(dados['otimizador']['state'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
