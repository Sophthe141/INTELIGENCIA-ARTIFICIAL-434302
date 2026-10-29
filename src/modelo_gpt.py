import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Tratamento flexível de importação do módulo de atenção
try:
    from src.atencao import MultiHeadAttention
except (ModuleNotFoundError, ImportError):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.atencao import MultiHeadAttention


# ==============================================================================
# 1. Layer Normalization
# ==============================================================================
class LayerNorm(nn.Module):
    """
    Normalização de Camada (Layer Normalization).
    Normaliza as ativações de cada amostra individualmente pela média zero e 
    variância unitária sobre a última dimensão (emb_dim), aplicando escala (gamma)
    e deslocamento (beta) aprendíveis.
    """
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, emb_dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# ==============================================================================
# 2. Ativação GELU (Gaussian Error Linear Unit)
# ==============================================================================
class GELU(nn.Module):
    """
    Função de Ativação GELU (Gaussian Error Linear Unit) aproximada.
    Utilizada na arquitetura GPT original:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


# ==============================================================================
# 3. Rede Feed-Forward (FFN / MLP)
# ==============================================================================
class FeedForward(nn.Module):
    """
    Rede Feed-Forward posição a posição (Position-wise Feed-Forward Network).
    Aplica uma expansão de dimensão (emb_dim -> 4 * emb_dim), ativação GELU,
    projeção de retorno (4 * emb_dim -> emb_dim) e Dropout para regularização.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(cfg.get("drop_rate", 0.1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        return self.layers(x)


# ==============================================================================
# 4. Bloco Transformer (Transformer Block com Pre-LayerNorm)
# ==============================================================================
class TransformerBlock(nn.Module):
    """
    Bloco Transformer com arquitetura Pre-LayerNorm e conexões residuais (skip connections).
    Integração de:
    1. LayerNorm + Multi-Head Attention Causal + Conexão residual
    2. LayerNorm + FeedForward Network + Conexão residual
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg.get("drop_rate", 0.0),
            qkv_bias=cfg.get("qkv_bias", False)
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg.get("drop_rate", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Sub-bloco de Atenção com Pre-LayerNorm e Conexão Residual
        shortcut = x
        x_norm = self.norm1(x)
        attn_out, _ = self.att(x_norm)
        x = shortcut + self.drop_shortcut(attn_out)

        # 2. Sub-bloco FeedForward com Pre-LayerNorm e Conexão Residual
        shortcut = x
        x_norm = self.norm2(x)
        ffn_out = self.ff(x_norm)
        x = shortcut + self.drop_shortcut(ffn_out)

        return x


# ==============================================================================
# 5. Arquitetura GPT Completa (GPTModel)
# ==============================================================================
class GPTModel(nn.Module):
    """
    Modelo GPT Completo (Decoder-only Transformer).
    Fluxo de processamento:
    Token IDs -> Token Embeddings + Positional Embeddings
              -> Dropout
              -> N x TransformerBlocks
              -> LayerNorm Final
              -> LM Head (Projeção Linear para vocab_size -> Logits)
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg.get("drop_rate", 0.0))

        # Empilhamento de N blocos Transformer
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        # in_idx shape: (batch_size, seq_len)
        batch_size, seq_len = in_idx.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"Sequência de entrada ({seq_len}) excede context_length ({self.cfg['context_length']})"
            )

        # Soma de embeddings de token e posição
        tok_embeds = self.tok_emb(in_idx)  # (batch_size, seq_len, emb_dim)
        positions = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(positions)  # (seq_len, emb_dim)

        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)  # (batch_size, seq_len, emb_dim)
        x = self.final_norm(x)  # (batch_size, seq_len, emb_dim)
        logits = self.out_head(x)  # (batch_size, seq_len, vocab_size)
        return logits


# ==============================================================================
# 6. Funções de Inferência e Geração Autoregressiva
# ==============================================================================
def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int
) -> torch.Tensor:
    """
    Geração autoregressiva simples baseada em busca gananciosa (greedy decoding).
    A cada iteração:
    1. Trunca o contexto para caber na janela máxima do modelo.
    2. Calcula os logits da última posição.
    3. Seleciona o token com maior valor de logit (argmax).
    4. Concatena o novo token à sequência gerada.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Trunca para os últimos context_size tokens
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Seleciona os logits da última posição: (batch_size, vocab_size)
        logits_last = logits[:, -1, :]
        idx_next = torch.argmax(logits_last, dim=-1, keepdim=True)  # (batch_size, 1)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text_advanced(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None
) -> torch.Tensor:
    """
    Geração autoregressiva com suporte a:
    - Escalonamento por Temperatura (controls randomness)
    - Filtragem Top-K (elimina a cauda longa de tokens improváveis)
    - Critério de parada precoce por token especial (eos_id)
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits_last = logits[:, -1, :]

        # 1. Filtro Top-K
        if top_k is not None and top_k > 0:
            top_k_val = min(top_k, logits_last.size(-1))
            top_logits, _ = torch.topk(logits_last, top_k_val)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits_last = torch.where(
                logits_last < min_val,
                torch.tensor(float('-inf'), device=logits_last.device),
                logits_last
            )

        # 2. Amostragem com Temperatura ou Busca Gananciosa
        if temperature > 0.0:
            scaled_logits = logits_last / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits_last, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            idx = torch.cat((idx, idx_next), dim=1)
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ==============================================================================
# 7. Configurações Pré-definidas do GPT-2
# ==============================================================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Tamanho do vocabulário do GPT-2 BPE
    "context_length": 1024, # Tamanho máximo do contexto
    "emb_dim": 768,         # Dimensão do embedding (d_model)
    "n_heads": 12,          # Número de cabeças de atenção
    "n_layers": 12,         # Número de blocos Transformer empilhados
    "drop_rate": 0.1,       # Taxa de dropout
    "qkv_bias": False       # Bias nas projeções lineares Q, K, V
}

GPT_CONFIG_TESTE = {
    "vocab_size": 1000,
    "context_length": 128,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 3,
    "drop_rate": 0.1,
    "qkv_bias": False
}


if __name__ == "__main__":
    print("=" * 60)
    print("Testando componentes do modelo GPT (Sprint 4)")
    print("=" * 60)
    torch.manual_seed(123)

    batch_size = 2
    seq_len = 8
    cfg = GPT_CONFIG_TESTE

    # 1. Teste LayerNorm
    ln = LayerNorm(emb_dim=cfg["emb_dim"])
    x_test = torch.randn(batch_size, seq_len, cfg["emb_dim"])
    out_ln = ln(x_test)
    print(f"LayerNorm: Saída {out_ln.shape} | Média={out_ln.mean().item():.4f}, Var={out_ln.var().item():.4f}")

    # 2. Teste GELU
    gelu = GELU()
    out_gelu = gelu(x_test)
    print(f"GELU: Saída {out_gelu.shape}")

    # 3. Teste FeedForward
    ffn = FeedForward(cfg)
    out_ffn = ffn(x_test)
    print(f"FeedForward: Saída {out_ffn.shape} (esperado {x_test.shape})")

    # 4. Teste TransformerBlock
    block = TransformerBlock(cfg)
    out_block = block(x_test)
    print(f"TransformerBlock: Saída {out_block.shape}")

    # 5. Teste GPTModel Completo
    model = GPTModel(cfg)
    input_ids = torch.randint(0, cfg["vocab_size"], (batch_size, seq_len))
    logits = model(input_ids)
    print(f"GPTModel: Entrada {input_ids.shape} -> Logits {logits.shape}")

    # 6. Teste de Geração Simples e Avançada
    prompt_ids = torch.randint(0, cfg["vocab_size"], (1, 4))
    gen_simple = generate_text_simple(model, prompt_ids, max_new_tokens=6, context_size=cfg["context_length"])
    print(f"Geração Simples (Greedy): Prompt={prompt_ids.shape} -> Gerado={gen_simple.shape}")

    gen_adv = generate_text_advanced(
        model, prompt_ids, max_new_tokens=6, context_size=cfg["context_length"],
        temperature=0.8, top_k=20
    )
    print(f"Geração Avançada (Top-K=20, Temp=0.8): Gerado={gen_adv.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal de parâmetros (Config Teste): {total_params:,}")
    print("Todos os testes do modelo_gpt.py foram executados com sucesso!")
