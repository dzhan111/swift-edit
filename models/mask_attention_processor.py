import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskAwareIPAttnProcessor(nn.Module):
    """
    IP-Adapter attention processor with mask-based attention rescaling

    Works with OR without MaskController:
    - Without controller: behaves like standard IPAdapterAttnProcessor
    - With controller: applies mask-aware attention rescaling (ARaM)
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int = None,
        scale: float = 1.0,
        num_tokens: int = 4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)

        # mask controller set during inference
        self.controller = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(
                hidden_states, kwargs.get("temb"))

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            text_hidden_states = encoder_hidden_states[:, :end_pos, :]
            ip_hidden_states = encoder_hidden_states[:, end_pos:, :]

            if attn.norm_cross:
                text_hidden_states = attn.norm_encoder_hidden_states(
                    text_hidden_states)

            encoder_hidden_states = text_hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
        else:
            ip_key = None
            ip_value = None

        if self.controller is not None and ip_key is not None:
            # use mask-aware attention with controller (ARaM)
            output = self._forward_with_mask_controller(
                attn, query, key, value, ip_key, ip_value, batch_size
            )
        else:
            output = self._forward_without_controller(
                attn, query, key, value, ip_key, ip_value, batch_size, attention_mask
            )

        output = attn.to_out[0](output)
        output = attn.to_out[1](output)

        if input_ndim == 4:
            output = output.transpose(-1, -2).reshape(batch_size,
                                                      channel, height, width)
        if attn.residual_connection:
            output = output + residual

        output = output / attn.rescale_output_factor

        return output

    def _forward_with_mask_controller(
        self,
        attn,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ip_key: torch.Tensor,
        ip_value: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:

        num_heads = attn.heads
        head_dim = query.shape[-1] // num_heads

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        ip_key = ip_key.view(batch_size, -1, num_heads,
                             head_dim).transpose(1, 2)
        ip_value = ip_value.view(
            batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # reshape to (B×H, N, D_h) for controller processing
        query = rearrange(query, 'b h n d -> (b h) n d')
        key = rearrange(key, 'b h n d -> (b h) n d')
        value = rearrange(value, 'b h n d -> (b h) n d')
        ip_key = rearrange(ip_key, 'b h n d -> (b h) n d')
        ip_value = rearrange(ip_value, 'b h n d -> (b h) n d')

        # compute text attention
        similarity_text = torch.einsum(
            'b i d, b j d -> b i j', query, key) * attn.scale
        output_text = self.controller.forward_text_branch(
            query, key, value, similarity_text, num_heads, scale=attn.scale
        )

        # compute image attention
        similarity_ip = torch.einsum(
            'b i d, b j d -> b i j', query, ip_key) * attn.scale
        output_ip = self.controller.forward_image_branch(
            query, ip_key, ip_value, similarity_ip, num_heads, scale=attn.scale
        )

        output = output_text + output_ip

        return output

    def _forward_without_controller(
        self,
        attn,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ip_key: torch.Tensor,
        ip_value: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward without mask controller (standard IP-Adapter from Equation 3).

        Args:
            attn: Attention module
            query: Query tensor
            key: Text key
            value: Text value
            ip_key: Image key (can be None)
            ip_value: Image value (can be None)
            batch_size: Batch size
            attention_mask: Optional attention mask

        Returns:
            Standard IP-Adapter attention output
        """
        head_dim = query.shape[-1] // attn.heads

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        # Text attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Image attention (if available)
        if ip_key is not None and ip_value is not None:
            ip_key = ip_key.view(batch_size, -1, attn.heads,
                                 head_dim).transpose(1, 2)
            ip_value = ip_value.view(
                batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            # Combine: Equation 3
            hidden_states = hidden_states + self.scale * ip_hidden_states

        return hidden_states
