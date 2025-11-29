import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


class MaskController:
    """
    Controller for mask-based attention manipulation (ARaM)
    """

    def __init__(
        self,
        mask: torch.Tensor,
        scale_text: float = 2.0,
        scale_edit: float = 0.0,
        scale_non_edit: float = 1.0,
    ):
        self.mask = mask
        self.scale_text = scale_text
        self.scale_edit = scale_edit
        self.scale_non_edit = scale_non_edit

    def compute_mask_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        similarity: torch.Tensor,
        num_heads: int,
        scale: float = 1.0,
        is_mask_attn: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention with optional mask-based separation

        When is_mask_attn=True, computes separate attention for foreground and background.

        Args:
            query: Query tensor, shape (b x heads, seq_len, head_dim)
            key: Key tensor, shape (b x heads, seq_len, head_dim)
            value: Value tensor, shape (b x heads, seq_len, head_dim)
            similarity: Pre-computed similarity scores
            num_heads: Number of attention heads
            scale: Attention scale factor
            is_mask_attn: Whether to apply mask-based separation

        """
        batch_size = query.shape[0] // num_heads
        spatial_dim = query.shape[1]
        H = W = int(np.sqrt(spatial_dim))

        # reshape to separate heads: (batch×heads, seq, dim) -> (heads, batch×seq, dim)
        query = rearrange(query, "(b h) n d -> h (b n) d", h=num_heads)
        key = rearrange(key, "(b h) n d -> h (b n) d", h=num_heads)
        value = rearrange(value, "(b h) n d -> h (b n) d", h=num_heads)

        similarity = torch.einsum("h i d, h j d -> h i j", query, key) * scale

        if is_mask_attn and self.mask is not None:
            mask = F.interpolate(
                self.mask, (H, W), mode='bilinear', align_corners=False)
            # (1, H, W) after flat then (1, 1, H, W)
            mask = mask.flatten(0).unsqueeze(0)
            # flatten all -> (H×W,) -> (1, H×W, 1)
            mask = mask.flatten().unsqueeze(0).unsqueeze(-1)

            # create separate attention maps for foreground and background
            # foreground: mask positions attended, background masked out
            sim_foreground = similarity + \
                mask.masked_fill(mask == 0, torch.finfo(similarity.dtype).min)

            # background: background positions attended, foreground masked out
            sim_background = similarity + \
                mask.masked_fill(mask == 1, torch.finfo(similarity.dtype).min)

            similarity = torch.cat([sim_foreground, sim_background], dim=0)

        attn_weights = similarity.softmax(dim=-1)

        if len(attn_weights) == 2 * len(value):
            value = torch.cat([value] * 2, dim=0)

        # apply attention
        output = torch.einsum("h i j, h j d -> h i d", attn_weights, value)

        if is_mask_attn and self.mask is not None:
            output = rearrange(
                output, "(h1 h) (b n) d -> (h1 b) n (h d)", b=batch_size, h=num_heads)
        else:
            output = rearrange(output, "h (b n) d -> b n (h d)", b=batch_size)

        return output

    def forward_text_branch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        similarity: torch.Tensor,
        num_heads: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass for text attention branch: s_y · M · Attn(Q_l, K_y, V_y)
        """
        batch_size = query.shape[0] // num_heads
        spatial_dim = query.shape[1]
        H = W = int(np.sqrt(spatial_dim))

        if batch_size <= 2:
            out_source = self.compute_mask_attention(
                query[:num_heads],
                key[:num_heads],
                value[:num_heads],
                similarity[:num_heads],
                num_heads,
                scale=scale,
            )
            if batch_size == 2:
                mask_spatial = F.interpolate(
                    self.mask, (H, W), mode='bilinear', align_corners=False)
                mask_spatial = mask_spatial.reshape(-1, 1)  # (H×W, 1)

                out_target = self.compute_mask_attention(
                    query[-num_heads:],
                    key[-num_heads:],
                    value[-num_heads:],
                    similarity[-num_heads:],
                    num_heads,
                    scale=scale,
                )

                # apply text scale: s_y · output
                if self.scale_text is not None:
                    out_target = self.scale_text * out_target

                output = torch.cat([out_source, out_target], dim=0)
            else:
                output = out_source
        else:
            out_source = self.compute_mask_attention(
                query[:num_heads],
                key[:num_heads],
                value[:num_heads],
                similarity[:num_heads],
                num_heads,
                scale=scale,
            )

            outputs = [out_source]
            for i in range(1, batch_size):
                out_i = self.compute_mask_attention(
                    query[i*num_heads:(i+1)*num_heads],
                    key[i*num_heads:(i+1)*num_heads],
                    value[i*num_heads:(i+1)*num_heads],
                    similarity[i*num_heads:(i+1)*num_heads],
                    num_heads,
                    scale=scale,
                )
                if self.scale_text is not None:
                    out_i = self.scale_text * out_i
                outputs.append(out_i)

            output = torch.cat(outputs, dim=0)

        return output

    def forward_image_branch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        similarity: torch.Tensor,
        num_heads: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass for image attention: s_edit · M · Attn(Q_l, K_x, V_x) + s_non-edit · (1-M) · Attn(Q_l, K_x, V_x)

        Essentially seperates foreground (edit) and background (non-edit) regions
        """
        batch_size = query.shape[0] // num_heads
        spatial_dim = query.shape[1]
        H = W = int(np.sqrt(spatial_dim))

        if batch_size <= 2:
            out_source = self.compute_mask_attention(
                query[:num_heads],
                key[:num_heads],
                value[:num_heads],
                similarity[:num_heads],
                num_heads,
                scale=scale,
            )

            if batch_size == 2:
                out_target = self.compute_mask_attention(
                    query[-num_heads:],
                    key[-num_heads:],
                    value[-num_heads:],
                    similarity[-num_heads:],
                    num_heads,
                    scale=scale,
                    is_mask_attn=True
                )
                out_foreground, _ = out_target.chunk(2, dim=0)

                mask_spatial = F.interpolate(
                    self.mask, (H, W), mode='bilinear', align_corners=False)
                mask_spatial = mask_spatial.reshape(-1, 1)  # (H×W, 1)

                out_target = (
                    self.scale_edit * out_foreground * mask_spatial +
                    self.scale_non_edit * out_source * (1 - mask_spatial)
                )

                output = torch.cat([out_source, out_target], dim=0)
            else:
                output = out_source
        else:
            out_source = self.compute_mask_attention(
                query[:num_heads],
                key[:num_heads],
                value[:num_heads],
                similarity[:num_heads],
                num_heads,
                scale=scale,
            )

            outputs = [out_source]
            mask_spatial = F.interpolate(
                self.mask, (H, W), mode='bilinear', align_corners=False)
            mask_spatial = mask_spatial.reshape(-1, 1)

            for i in range(1, batch_size):
                out_i = self.compute_mask_attention(
                    query[i*num_heads:(i+1)*num_heads],
                    key[i*num_heads:(i+1)*num_heads],
                    value[i*num_heads:(i+1)*num_heads],
                    similarity[i*num_heads:(i+1)*num_heads],
                    num_heads,
                    scale=scale,
                    is_mask_attn=True,
                )

                out_fg, _ = out_i.chunk(2, dim=0)
                out_i = (
                    self.scale_edit * out_fg * mask_spatial +
                    self.scale_non_edit * out_source * (1 - mask_spatial)
                )
                outputs.append(out_i)

            output = torch.cat(outputs, dim=0)

        return output
