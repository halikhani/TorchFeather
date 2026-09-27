import math

import torch
from torch import nn

from torchfeather.model.attention import (
    ScaledDotProductAttentionWrapper,
)
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.moe import FeedForward, MoE
from torchfeather.model.rope import apply_rotary_emb, precompute_freqs_cis


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.dim = model_args.dim # 2048
        self.n_heads = model_args.n_heads # 16
        self.q_lora_rank = model_args.q_lora_rank  # 0
        self.kv_lora_rank = model_args.kv_lora_rank  # 512
        self.qk_nope_head_dim = model_args.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = model_args.qk_rope_head_dim  # 64
        self.qk_head_dim = (
            model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        )  # 128 + 64 = 192
        self.v_head_dim = model_args.v_head_dim  # 128

        if self.q_lora_rank > 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps) # only the non-rope part is normalized
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = ScaledDotProductAttentionWrapper()


    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x) # (batch_size, seq_len, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x) # (batch_size, seq_len, q_lora_rank)
            q = self.wq_b(self.q_norm(q)) # (batch_size, seq_len, n_heads * qk_head_dim)

        # q: [batch_size, seq_len, n_heads, qk_head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
        # q_nope: [batch_size, seq_len, n_heads, qk_nope_head_dim]
        # q_rope: [batch_size, seq_len, n_heads, qk_rope_head_dim]
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        # q: (batch_size, seq_len, n_heads, qk_head_dim)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # Key-value projection
        # kv: [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim]

        kv = self.wkv_a(x) # [batch_size, seq_len, self.kv_lora_rank + self.qk_rope_head_dim]
        # kv: [batch_size, seq_len, kv_lora_rank] --- this is the compressed latent
        # k_rope: [batch_size, seq_len, qk_rope_head_dim] --- this is the decoupled RoPE for K
        kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # k_rope: [batch_size, seq_len, 1, qk_rope_head_dim]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        # up projection to rebuild the full KV
        # the up projection also contains W_V, which is usually kept separate
        # kv: [batch_size, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]

        kv = self.wkv_b(self.kv_norm(kv))
        # kv: [batch_size, seq_len, n_heads, qk_nope_head_dim + v_head_dim]
        kv = kv.view(batch_size, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        # k_nope: [batch_size, seq_len, n_heads, qk_nope_head_dim]
        # v: [batch_size, seq_len, n_heads, v_head_dim]
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k: (batch_size, seq_len, n_heads, qk_head_dim)
        # we need to expand k_rope because it is shared for every K heads. This basically adds a new dimension with 0 stride.
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_heads, -1)], dim=-1)


        # q: [batch_size, n_heads, seq_len, qk_head_dim]
        q = q.transpose(1, 2)
        # k: [batch_size, n_heads, seq_len, qk_head_dim]
        k = k.transpose(1, 2)
        # v: [batch_size, n_heads, seq_len, v_head_dim]
        v = v.transpose(1, 2)

        # attention as usual
        attn_output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        # output: [batch_size, seq_len, n_heads, v_head_dim]
        output = attn_output.transpose(1, 2).contiguous()
        # output: [batch_size, seq_len, n_heads * v_head_dim]
        output = output.view(batch_size, seq_len, -1)
        # output: [batch_size, seq_len, dim]
        return self.wo(output)











class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

            # This is different from the GPT2-style initialisation, as visible in the HF implementation: https://github.com/huggingface/transformers/blob/39603d0e5cdb6f00e8d473d7fcbb01032d709181/src/transformers/models/gpt2/modeling_gpt2.py#L448-L458
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
            self.layer_id = layer_id

        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """

        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

    def init_weights(
        self,
        init_std: float | None = None,
        buffer_device: torch.device | None = None,
    ):
        if buffer_device is None:
            raise ValueError("buffer_device is required for init_weights in TransformerBlock")
        
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(init_std=init_std, buffer_device=buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)

            



        
        

class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.d_model)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        ) # persis = False means that the buffer is not saved to the checkpoint (not in the state_dict)

        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.default_dtype,
            bias=False,
        )


    def init_weights(
        self,
        init_std: float | None = None,
        buffer_device: torch.device | None = None,
    ):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std, buffer_device=buffer_device)

        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

        
    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices for the ranks on the first pipeline stage. This will be the activation of the previous pipeline stage if the current rank is not on the first stage.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, freqs_cis=self.freqs_cis)
        
        h = self.norm(h) if self.norm is not None else h
        logits = self.output(h) if self.output is not None else h 

        return logits



        