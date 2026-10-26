"""MIPIC objectives ported from active notebook cells 13/14.

Attention KL is deliberately zero; top-k retains selected padding and CKA is
computed in FP64, including singleton selections, exactly as in the notebook.
"""

import math
from typing import List, Optional, Tuple, Dict, Union, Callable

import torch
from torch import nn
from torch.nn import functional as F


def pool_hidden(hidden, mask=None, pooling="cls"):
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":  # Compatibility when evaluating older mean checkpoints.
        weights = mask.unsqueeze(-1).float()
        return (hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
    raise ValueError("pooling must be cls or mean.")


class CKALoss(nn.Module):

    def __init__(self, eps=1e-08):
        super().__init__()
        self.eps = eps

    def compute_cka_single(self, X, Y):
        X = X.to(torch.float64)
        Y = Y.to(torch.float64)
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        XTY = X.t().matmul(Y)
        XTX = X.t().matmul(X)
        YTY = Y.t().matmul(Y)
        numerator = torch.norm(XTY, 'fro') ** 2
        denominator = torch.norm(XTX, 'fro') * torch.norm(YTY, 'fro') + self.eps
        cka_sim = numerator / denominator
        return cka_sim

    def forward(self, SH, TH):
        if SH.dim() == 3:
            B, k, d_s = SH.shape
            _, _, d_t = TH.shape
            cka_sims = []
            for i in range(B):
                cka_sim = self.compute_cka_single(SH[i], TH[i])
                cka_sims.append(cka_sim)
            avg_cka_sim = torch.stack(cka_sims).mean()
        elif SH.dim() == 2:
            avg_cka_sim = self.compute_cka_single(SH, TH)
        else:
            raise ValueError(f'Expected 2D or 3D input, got shape {SH.shape}')
        return 1.0 - avg_cka_sim.float()


class HorizontalAttentionAlignment(nn.Module):

    def __init__(self, d_small: int, d_full: int=768, d_att: int=64):
        super().__init__()
        self.d_att = d_att
        self.proj_small = nn.Linear(d_small, d_att)
        self.proj_full = nn.Linear(d_full, d_att)
        self.W_Q = nn.Linear(d_att, d_att)
        self.W_K = nn.Linear(d_att, d_att)

    def compute_attention_dist(self, hidden: torch.Tensor, proj: nn.Linear, mask: Optional[torch.Tensor]=None, temperature: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = hidden.shape
        h_proj = proj(hidden)
        q_cls = self.W_Q(h_proj[:, 0, :])
        k_all = self.W_K(h_proj)
        scores = torch.matmul(q_cls.unsqueeze(1), k_all.transpose(1, 2))
        scores = scores.squeeze(1) / math.sqrt(self.d_att)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(scores / temperature, dim=-1)
        return (attn_probs, scores)

    def forward(self, h_small: torch.Tensor, h_full: torch.Tensor, mask: Optional[torch.Tensor]=None, temperature: float=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        small_probs, small_scores = self.compute_attention_dist(h_small, self.proj_small, mask, temperature)
        full_probs, full_scores = self.compute_attention_dist(h_full, self.proj_full, mask, temperature)
        eps = 1e-08
        small_probs = small_probs + eps
        full_probs = full_probs + eps
        kl_loss = F.kl_div(small_probs.log(), full_probs, reduction='batchmean', log_target=False)
        full_scores = 0
        # Active notebook behavior: KL is disabled; top-k indices carry no
        # gradient to the attention projections.
        kl_loss = 0
        return (kl_loss, small_scores, full_scores)


class SubmatrixCKALoss(nn.Module):

    def __init__(self, eps: float=1e-08):
        super().__init__()
        self.cka_loss = CKALoss(eps=eps)

    def select_top_k_tokens(self, hidden: torch.Tensor, selection_scores: torch.Tensor, k: int, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        B, L, d = hidden.shape
        if mask is not None:
            scores_masked = selection_scores.masked_fill(mask == 0, float('-inf'))
        else:
            scores_masked = selection_scores
        _, top_k_indices = torch.topk(scores_masked, k=min(k, L), dim=1)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, d)
        selected_hidden = torch.gather(hidden, 1, top_k_indices_expanded)
        # Keep padding selected when valid tokens < k, as in the notebook.
        return selected_hidden

    def forward(self, h_small: torch.Tensor, h_full: torch.Tensor, small_scores: torch.Tensor, k: int, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        small_sub = self.select_top_k_tokens(h_small, small_scores, k, mask)
        full_sub = self.select_top_k_tokens(h_full, small_scores, k, mask)
        loss = self.cka_loss(small_sub, full_sub)
        return loss


class PipelineInfoNCELoss(nn.Module):

    def __init__(self, d_src: int, d_tgt: int, d_hidden: int=256):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_src, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_tgt))

    def forward(self, src_hidden: torch.Tensor, tgt_hidden: torch.Tensor, mask: Optional[torch.Tensor]=None, temperature: float=0.07) -> torch.Tensor:
        u_src = src_hidden[:, 0, :]
        v_tgt = tgt_hidden[:, 0, :]
        u_proj = self.phi(u_src)
        v_tgt = v_tgt.detach()
        u_proj = F.normalize(u_proj, dim=-1)
        v_tgt = F.normalize(v_tgt, dim=-1)
        logits = torch.matmul(u_proj, v_tgt.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class TotalAlignmentLoss(nn.Module):

    def __init__(self, d_full: int=768, matryoshka_dims: List[int]=[768, 512, 256, 128, 64, 32, 16], align_layers: List[int]=[4, 8], pipeline_pairs: List[Tuple[int, ...]]=None, alpha: float=1.0, beta: float=1.0, gamma: float=1.0, k_map: Optional[Union[Dict[int, int], Callable[[int], int]]]=None, base_k: int=32, temperature: float=1.0, d_att: int=64):
        super().__init__()
        self.d_full = d_full
        self.matryoshka_dims = sorted([d for d in matryoshka_dims if d <= d_full], reverse=True)
        self.align_layers = align_layers
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        if k_map is None:
            self.k_map = lambda d: max(8, int(d / d_full * base_k))
        elif callable(k_map):
            self.k_map = k_map
        elif isinstance(k_map, dict):
            self.k_map = lambda d: k_map.get(d, max(8, int(d / d_full * base_k)))
        else:
            raise ValueError('k_map must be None, callable, or dict')
        if pipeline_pairs is None:
            pipeline_pairs = [(3, 16, 5, 32, 7, 64, 8, 128, 9, 256, 10, 512, 11, 768)]
        self.parsed_pipeline_pairs = []
        for pipe_config in pipeline_pairs:
            if len(pipe_config) % 2 != 0:
                raise ValueError(f'Pipeline config must have even length: {pipe_config}')
            stages = [(pipe_config[i], pipe_config[i + 1]) for i in range(0, len(pipe_config), 2)]
            for i in range(len(stages) - 1):
                layer_i, dim_i = stages[i]
                layer_j, dim_j = stages[i + 1]
                self.parsed_pipeline_pairs.append((layer_i, dim_i, layer_j, dim_j))
        self.attn_modules = nn.ModuleDict()
        for layer_idx in align_layers:
            for dim in self.matryoshka_dims:
                if dim >= d_full:
                    continue
                key = f'layer_{layer_idx}_dim_{dim}'
                self.attn_modules[key] = HorizontalAttentionAlignment(d_small=dim, d_full=d_full, d_att=d_att)
        self.cka_module = SubmatrixCKALoss()
        self.infonce_modules = nn.ModuleDict()
        for layer_i, dim_i, layer_j, dim_j in self.parsed_pipeline_pairs:
            key = f'pipe_{layer_i}_{dim_i}_to_{layer_j}_{dim_j}'
            self.infonce_modules[key] = PipelineInfoNCELoss(d_src=dim_i, d_tgt=dim_j, d_hidden=max(dim_i, dim_j) // 2)

    def forward(self, hidden_states: List[torch.Tensor], mask: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        total_att_loss = 0.0
        total_cka_loss = 0.0
        total_chain_loss = 0.0
        att_count = 0
        cka_count = 0
        for layer_idx in self.align_layers:
            full_layer = hidden_states[layer_idx]
            for dim in self.matryoshka_dims:
                if dim >= self.d_full:
                    continue
                small_layer = full_layer[..., :dim]
                k_i = self.k_map(dim)
                key = f'layer_{layer_idx}_dim_{dim}'
                attn_module = self.attn_modules[key]
                att_loss, small_scores, full_scores = attn_module(h_small=small_layer, h_full=full_layer, mask=mask, temperature=self.temperature)
                total_att_loss += att_loss
                att_count += 1
                cka_loss = self.cka_module(h_small=small_layer, h_full=full_layer, small_scores=small_scores, k=k_i, mask=mask)
                total_cka_loss += cka_loss
                cka_count += 1
        if att_count > 0:
            total_att_loss = total_att_loss / att_count
        if cka_count > 0:
            total_cka_loss = total_cka_loss / cka_count
        for layer_i, dim_i, layer_j, dim_j in self.parsed_pipeline_pairs:
            src_hidden = hidden_states[layer_i][..., :dim_i]
            tgt_hidden = hidden_states[layer_j][..., :dim_j]
            key = f'pipe_{layer_i}_{dim_i}_to_{layer_j}_{dim_j}'
            infonce_module = self.infonce_modules[key]
            chain_loss = infonce_module(src_hidden=src_hidden, tgt_hidden=tgt_hidden, mask=mask, temperature=0.07)
            total_chain_loss += chain_loss
        total_chain_loss = total_chain_loss / len(self.parsed_pipeline_pairs)
        total_loss = self.alpha * total_att_loss + self.beta * total_cka_loss + self.gamma * total_chain_loss
        return {'total_loss': total_loss, 'att_loss': total_att_loss, 'cka_loss': total_cka_loss, 'chain_loss': total_chain_loss}


def Matry_infonce(a, b, temperature=0.07, nested_dims=[64, 128, 256, 512, 768]):
    assert a.dim() == 2, f'Expected a to be 2D [batch, dim], got shape {a.shape}'
    assert b.dim() == 2, f'Expected b to be 2D [batch, dim], got shape {b.shape}'
    assert a.shape == b.shape, f'a and b must have same shape, got {a.shape} vs {b.shape}'
    total_loss = 0.0
    all_logits = {}
    full_dim = a.size(1)
    for dim in nested_dims:
        if dim > full_dim:
            print(f'Warning: skipping dim={dim} as it exceeds full_dim={full_dim}')
            continue
        q = a[:, :dim]
        k = b[:, :dim]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.matmul(q, k.T) / temperature
        labels = torch.arange(q.size(0), device=q.device)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss
        all_logits[f'dim_{dim}'] = logits
    return (total_loss, all_logits)
