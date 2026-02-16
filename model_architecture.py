"""
YouAI Model Architecture - 2B Parameter Transformer
"""

import torch
import torch.nn as nn
import math


class YouAIConfig:
    """Configuration for YouAI 2B parameter model"""
    def __init__(
        self,
        vocab_size=50257,           # GPT-2 tokenizer vocab size
        max_position_embeddings=2048,  # Context length
        hidden_size=2048,            # Embedding dimension
        num_hidden_layers=24,        # Number of transformer layers
        num_attention_heads=16,      # Attention heads
        intermediate_size=8192,      # FFN hidden dimension
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        layer_norm_eps=1e-5,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        
        # Calculate total parameters
        self.total_params = self._calculate_params()
    
    def _calculate_params(self):
        """Calculate approximate parameter count"""
        # Embeddings
        token_emb = self.vocab_size * self.hidden_size
        pos_emb = self.max_position_embeddings * self.hidden_size
        
        # Each transformer layer
        per_layer = (
            # Self-attention
            4 * self.hidden_size * self.hidden_size +  # Q, K, V, O projections
            # FFN
            2 * self.hidden_size * self.intermediate_size +
            # Layer norms (approximate)
            4 * self.hidden_size
        )
        
        all_layers = per_layer * self.num_hidden_layers
        
        # Output head
        output_head = self.hidden_size * self.vocab_size
        
        total = token_emb + pos_emb + all_layers + output_head
        return total


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        output = self.o_proj(context)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer layer"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(self.ln1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.ln2(hidden_states))
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states


class YouAIModel(nn.Module):
    """Complete YouAI 2B parameter model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"Model initialized with {self.count_parameters():,} parameters")
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Create causal attention mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=input_ids.device) * float('-inf'),
                diagonal=1
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}
    
    def generate(self, input_ids, max_length=50, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == 50256:  # EOS token
                    break
        
        return input_ids


def create_youai_2b():
    """Create a 2B parameter YouAI model"""
    config = YouAIConfig(
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=8192,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
    )
    
    model = YouAIModel(config)
    return model, config


if __name__ == "__main__":
    print("Creating YouAI 2B parameter model...")
    model, config = create_youai_2b()
    print(f"\nModel Configuration:")
    print(f"  Total Parameters: {config.total_params:,}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    print(f"  Vocabulary Size: {config.vocab_size}")
    print(f"  Max Sequence Length: {config.max_position_embeddings}")
