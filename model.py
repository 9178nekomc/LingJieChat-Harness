# model.py — MiniMoE 模型定义
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (DIM, N_LAYERS, N_HEADS, N_KV_HEADS, D_FF, MAX_SEQ_LEN,
                    N_EXPERTS, TOP_K, DROPOUT)




# ============================================================
# LoRA 低秩适配层
# ============================================================
class LoRALinear(nn.Module):
    """给 Linear 加 LoRA。冻结原权重，只训练 A/B 两个低秩矩阵。"""
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B 初始化为 0，保证初始时 LoRA 贡献为 0

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora(model, r=8, alpha=16, dropout=0.0, targets=None):
    """遍历模型，先冻结所有参数，再给指定名字的 Linear 层加 LoRA"""
    if targets is None:
        targets = ["wq", "wk", "wv", "wo"]

    # 关键：先冻结全部参数
    for p in model.parameters():
        p.requires_grad = False

    count = 0
    for name, module in model.named_modules():
        for target in targets:
            if name.endswith(target):
                # 找到父模块
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)
                old_layer = getattr(parent, attr_name)
                new_layer = LoRALinear(old_layer, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, new_layer)
                count += 1

    print(f"LoRA applied to {count} layers")
    return model


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight


def precompute_rope(head_dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh), cos/sin: (T, Dh/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.n_kv = N_KV_HEADS
        self.head_dim = DIM // N_HEADS
        self.n_groups = N_HEADS // N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS * self.head_dim, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * self.head_dim, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * self.head_dim, bias=False)
        self.wo = nn.Linear(N_HEADS * self.head_dim, DIM, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_gate = nn.Linear(DIM, D_FF, bias=False)
        self.w_up = nn.Linear(DIM, D_FF, bias=False)
        self.w_down = nn.Linear(D_FF, DIM, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_experts = N_EXPERTS
        self.top_k = TOP_K
        self.router = nn.Linear(DIM, N_EXPERTS, bias=False)
        self.experts = nn.ModuleList([SwiGLU() for _ in range(N_EXPERTS)])
        self.aux_loss = 0.0

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.size(0)

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 保存路由信息供外部统计
        self.last_router_probs = probs.detach()
        self.last_topk_idx = topk_idx.detach()

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx_k = topk_idx[:, k]
            w_k = topk_probs[:, k]
            for e in range(self.n_experts):
                mask = (idx_k == e)
                if not mask.any():
                    continue
                expert_out = self.experts[e](x_flat[mask])
                out[mask] += w_k[mask].unsqueeze(-1) * expert_out

        if self.training:
            one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).float()
            f = one_hot.sum(dim=(0, 1)) / (N * self.top_k)
            P = probs.mean(dim=0)
            self.aux_loss = self.n_experts * (f * P).sum() * 0.05
        else:
            self.aux_loss = 0.0

        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(DIM)
        self.attn = Attention()
        self.ln2 = RMSNorm(DIM)
        self.moe = MoE()

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.moe(self.ln2(x))
        return x


class MiniMoE(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = RMSNorm(DIM)
        self.head = nn.Linear(DIM, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        head_dim = DIM // N_HEADS
        cos, sin = precompute_rope(head_dim, MAX_SEQ_LEN)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)
        cos, sin = self.cos[:T], self.sin[:T]

        aux_sum = 0.0
        for block in self.blocks:
            x = block(x, cos, sin, mask)
            aux_sum = aux_sum + block.moe.aux_loss

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            if loss_mask is not None:
                loss_per = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    reduction='none')
                loss = (loss_per * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-9)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    ignore_index=0)
            loss = loss + aux_sum / len(self.blocks)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MAX_SEQ_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            if eos_id is not None and idx_next.item() == eos_id:
                break
        return idx
