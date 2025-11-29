import torch
import torch.nn as nn


class ImageProjectionModel(nn.Module):
    """
    map CLIP image embeddings to cross-attention features
    """

    def __init__(
        self,
        cross_attention_dim: int = 1024,
        clip_embeddings_dim: int = 1024,
        num_tokens: int = 4,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        # linear projection: maps CLIP embeddings to N × cross_attention_dim
        self.proj = nn.Linear(
            clip_embeddings_dim, num_tokens * cross_attention_dim
        )

        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = image_embeds.shape[0]
        projected = self.proj(image_embeds)

        projected = projected.reshape(
            batch_size, self.num_tokens, self.cross_attention_dim
        )

        projected = self.norm(projected)

        return projected
