import unittest
import torch
from src.modelo_gpt import GPTModel, FeedForward, generate_text_advanced


class TestModelo(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.cfg = dict(vocab_size=11, context_length=8, emb_dim=8,
                        n_heads=2, n_layers=1, drop_rate=0.1)
        self.modelo = GPTModel(self.cfg).eval()

    def test_causalidade(self):
        x = torch.tensor([[1, 2, 3, 4]])
        outro = torch.tensor([[1, 2, 8, 9]])
        with torch.no_grad():
            a, b = self.modelo(x), self.modelo(outro)
        torch.testing.assert_close(a[:, :2], b[:, :2])

    def test_dropout_da_ffn_esta_no_bloco_residual(self):
        self.assertFalse(any(isinstance(c, torch.nn.Dropout) for c in FeedForward(self.cfg).modules()))

    def test_eos_rejeita_lote_sem_parada_individual(self):
        with self.assertRaisesRegex(ValueError, 'único exemplo'):
            generate_text_advanced(self.modelo, torch.ones((2, 2), dtype=torch.long), 3, 8, eos_id=0)


if __name__ == '__main__':
    unittest.main()
