import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from .flag_math import orthogonality_error
from .loss import FlagProjector


def mean_pooling(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)


class FlagTextModel(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder, self.projector = encoder, projector

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def config(self):
        return self.encoder.config

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        output = self.encoder(*args, **kwargs)
        # Linear projection commutes with masked mean pooling. Existing evaluators
        # can keep using mean_pooling(output, mask)[:, :dim] without any changes.
        output.last_hidden_state = self.projector(output.last_hidden_state)
        return output

    def sentence_embeddings(self, input_ids, attention_mask, dim=None, normalize=True, **kwargs):
        output = self(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        embeddings = mean_pooling(output.last_hidden_state, attention_mask)
        dim = embeddings.size(-1) if dim is None else dim
        if isinstance(dim, bool) or not isinstance(dim, int) or not 0 < dim <= embeddings.size(-1):
            raise ValueError("dim must be an integer in [1, hidden_size].")
        embeddings = embeddings[:, :dim]
        return F.normalize(embeddings, dim=-1) if normalize else embeddings

    def save_pretrained(self, output_dir, tokenizer=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir / "encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        metadata = {
            "format_version": 1, "pooling": "mean", "dims": self.projector.dims,
            "full_dim": self.projector.U.size(0), "flag_lr": self.projector.lr,
            "flag_eps": self.projector.eps, "flag_lambda_neg": self.projector.lambda_neg,
            "flag_retraction": self.projector.retraction,
        }
        (output_dir / "flag_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        torch.save(self.projector.state_dict(), output_dir / "flag_state.pt")

    @classmethod
    def from_pretrained(cls, checkpoint_dir, **encoder_kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        metadata = json.loads((checkpoint_dir / "flag_config.json").read_text())
        if metadata.get("format_version") != 1 or metadata.get("pooling") != "mean":
            raise ValueError("Unsupported FlagMRL checkpoint format or pooling.")
        encoder = AutoModel.from_pretrained(checkpoint_dir / "encoder", **encoder_kwargs)
        if encoder.config.hidden_size != metadata["full_dim"]:
            raise ValueError("Encoder hidden size does not match the saved flag basis.")
        projector = FlagProjector(
            metadata["full_dim"], metadata["dims"], init="identity",
            lr=metadata["flag_lr"], eps=metadata["flag_eps"],
            lambda_neg=metadata["flag_lambda_neg"], retraction=metadata["flag_retraction"])
        state = torch.load(checkpoint_dir / "flag_state.pt", map_location="cpu", weights_only=True)
        if (state["U"].shape != projector.U.shape or not torch.isfinite(state["U"]).all()
                or orthogonality_error(state["U"]) > 1e-3
                or not torch.equal(state["a_col"].float(), projector.a_col)):
            raise ValueError("Invalid flag basis or coordinate weights in checkpoint.")
        projector.load_state_dict(state, strict=True)
        return cls(encoder, projector).eval()
