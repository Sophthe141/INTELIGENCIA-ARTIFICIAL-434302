import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Mecanismo básico de Autoatenção (Self-Attention) com projeções lineares treináveis (W_q, W_k, W_v).
    Calcula os vetores de contexto ponderados sem fator de escala e sem máscara causal.
    """
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor):
        # x shape: (b, num_tokens, d_in)
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys    = self.W_key(x)    # (b, num_tokens, d_out)
        values  = self.W_value(x)  # (b, num_tokens, d_out)

        # Attention scores: Q * K^T
        attn_scores = queries @ keys.transpose(-2, -1)  # (b, num_tokens, num_tokens)

        # Pesos de atenção normalizados via Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Vetores de contexto: Attention Weights * V
        context_vec = attn_weights @ values  # (b, num_tokens, d_out)
        return context_vec, attn_weights


class CausalAttention(nn.Module):
    """
    Mecanismo de Atenção Causal Escalonada (Scaled Dot-Product Causal Attention).
    Aplica:
    1. Escalonamento das pontuações por sqrt(d_k) para estabilidade de gradientes.
    2. Máscara triangular superior com -inf para preservar a autoregressão (evitar vazamento de tokens futuros).
    3. Dropout nos pesos de atenção para regularização.
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Buffer persistente para a máscara triangular superior
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys    = self.W_key(x)    # (b, num_tokens, d_out)
        values  = self.W_value(x)  # (b, num_tokens, d_out)

        # Scaled Dot-Product: divide por sqrt(d_k)
        d_k = keys.shape[-1]
        attn_scores = (queries @ keys.transpose(-2, -1)) / (d_k ** 0.5)

        # Aplicação da máscara causal autoregressiva
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalização Softmax e Dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, num_tokens, d_out)
        return context_vec, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Mecanismo de Atenção Multi-Cabeça (Multi-Head Attention) vetorizado.
    Realiza projeções lineares conjuntas que são divididas em múltiplos subespaços (heads),
    permitindo que o modelo capture diferentes relações de dependência simultaneamente.
    Inclui projeção linear final (out_proj) e máscara causal.
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float = 0.0, num_heads: int = 2, qkv_bias: bool = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out deve ser divisível por num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape

        # Projeção e reordenação de dimensões: (b, num_heads, num_tokens, head_dim)
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product por cabeça
        attn_scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Máscara causal aplicada a todas as cabeças e lotes
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Agregação e recombinação das cabeças: (b, num_tokens, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec), attn_weights


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_size, num_tokens, d_in = 2, 4, 8
    d_out = 16
    
    x_exemplo = torch.rand(batch_size, num_tokens, d_in)
    
    # Teste Self-Attention
    sa = SelfAttention(d_in=d_in, d_out=d_out)
    ctx_sa, w_sa = sa(x_exemplo)
    print(f"Self-Attention: Saída={ctx_sa.shape}, Pesos={w_sa.shape}")

    # Teste Causal Attention
    ca = CausalAttention(d_in=d_in, d_out=d_out, context_length=num_tokens)
    ctx_ca, w_ca = ca(x_exemplo)
    print(f"Causal Attention: Saída={ctx_ca.shape}, Pesos={w_ca.shape}")

    # Teste Multi-Head Attention
    mha = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=num_tokens, num_heads=4)
    ctx_mha, w_mha = mha(x_exemplo)
    print(f"Multi-Head Attention: Saída={ctx_mha.shape}, Pesos={w_mha.shape}")
    print("Módulos de atenção validados com sucesso!")
