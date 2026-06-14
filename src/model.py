import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import EMBED_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT, ATTENTION_POOLING_DIM

class AttentionPooling(nn.Module):
    """Attention pooling layer that computes a query-based weighted average of tokens."""
    def __init__(self, embed_dim, attention_dim=ATTENTION_POOLING_DIM):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        scores = self.v(torch.tanh(self.query_proj(x)))  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(scores, dim=1)       # (batch_size, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)      # (batch_size, embed_dim)
        return pooled, attn_weights.squeeze(-1)          # pooled: (batch_size, embed_dim), weights: (batch_size, seq_len)


class VINTransformerEncoder(nn.Module):
    """VIN Transformer Encoder with multi-task and hierarchical prediction heads."""
    def __init__(self, vocab_size, target_classes_dict, regression_targets_list, embed_dim=EMBED_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT, hier_proj_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.target_classes_dict = target_classes_dict
        self.regression_targets_list = regression_targets_list
        self.embed_dim = embed_dim
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Learnable Positional Encoding (sequence length is exactly 17)
        self.position_embedding = nn.Parameter(torch.zeros(1, 17, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling
        self.attention_pooling = AttentionPooling(embed_dim)
        
        # Standard Classification Heads (Independent)
        self.class_heads = nn.ModuleDict()
        for target, num_classes in target_classes_dict.items():
            if target not in ["make_grouped", "model_final", "trim_raw", "bodyType_raw"]:
                self.class_heads[target] = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.LayerNorm(embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim // 2, num_classes)
                )
                
        # Standard Regression Heads (Independent)
        self.reg_heads = nn.ModuleDict()
        for target in regression_targets_list:
            self.reg_heads[target] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1)
            )

        # Hierarchical Stage 1: make_grouped
        num_make_classes = target_classes_dict["make_grouped"]
        self.make_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_make_classes)
        )
        self.make_proj = nn.Linear(num_make_classes, hier_proj_dim)
        
        # Hierarchical Stage 2: model_final (depends on make)
        num_model_classes = target_classes_dict["model_final"]
        self.model_head = nn.Sequential(
            nn.Linear(embed_dim + hier_proj_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_model_classes)
        )
        self.model_proj = nn.Linear(num_model_classes, hier_proj_dim)
        
        # Hierarchical Stage 3: trim_raw (depends on make, model)
        num_trim_classes = target_classes_dict["trim_raw"]
        self.trim_head = nn.Sequential(
            nn.Linear(embed_dim + 2 * hier_proj_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_trim_classes)
        )
        self.trim_proj = nn.Linear(num_trim_classes, hier_proj_dim)
        
        # Hierarchical Stage 4: bodyType_raw (depends on make, model, trim)
        num_body_classes = target_classes_dict["bodyType_raw"]
        self.body_head = nn.Sequential(
            nn.Linear(embed_dim + 3 * hier_proj_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_body_classes)
        )

    def forward(self, tokens, teacher_forcing_targets=None):
        # tokens shape: (batch_size, 17)
        batch_size = tokens.size(0)
        
        # Embedding and Positional Encoding
        x = self.token_embedding(tokens) + self.position_embedding # (batch_size, 17, embed_dim)
        
        # Transformer encoding
        enc_out = self.transformer_encoder(x) # (batch_size, 17, embed_dim)
        
        # Pooling to shared feature vector
        shared_feature, attn_weights = self.attention_pooling(enc_out) # shared_feature: (batch_size, embed_dim)
        
        # Predict independent classification targets
        outputs = {}
        for target, head in self.class_heads.items():
            outputs[target] = head(shared_feature)
            
        # Predict independent regression targets
        for target, head in self.reg_heads.items():
            outputs[target] = head(shared_feature).squeeze(-1) # (batch_size,)
            
        # Hierarchical prediction: Stage 1 (make_grouped)
        make_logits = self.make_head(shared_feature)
        outputs["make_grouped"] = make_logits
        
        # Determine whether to use teacher forcing or predicted distribution
        if self.training and teacher_forcing_targets is not None and "make_grouped" in teacher_forcing_targets:
            # One-hot representation of ground truth
            make_target = teacher_forcing_targets["make_grouped"]
            make_probs = F.one_hot(make_target, num_classes=self.target_classes_dict["make_grouped"]).float()
        else:
            make_probs = F.softmax(make_logits, dim=-1)
            
        make_proj = self.make_proj(make_probs)
        
        # Stage 2 (model_final)
        model_input = torch.cat([shared_feature, make_proj], dim=-1)
        model_logits = self.model_head(model_input)
        outputs["model_final"] = model_logits
        
        if self.training and teacher_forcing_targets is not None and "model_final" in teacher_forcing_targets:
            model_target = teacher_forcing_targets["model_final"]
            model_probs = F.one_hot(model_target, num_classes=self.target_classes_dict["model_final"]).float()
        else:
            model_probs = F.softmax(model_logits, dim=-1)
            
        model_proj = self.model_proj(model_probs)
        
        # Stage 3 (trim_raw)
        trim_input = torch.cat([shared_feature, make_proj, model_proj], dim=-1)
        trim_logits = self.trim_head(trim_input)
        outputs["trim_raw"] = trim_logits
        
        if self.training and teacher_forcing_targets is not None and "trim_raw" in teacher_forcing_targets:
            trim_target = teacher_forcing_targets["trim_raw"]
            trim_probs = F.one_hot(trim_target, num_classes=self.target_classes_dict["trim_raw"]).float()
        else:
            trim_probs = F.softmax(trim_logits, dim=-1)
            
        trim_proj = self.trim_proj(trim_probs)
        
        # Stage 4 (bodyType_raw)
        body_input = torch.cat([shared_feature, make_proj, model_proj, trim_proj], dim=-1)
        body_logits = self.body_head(body_input)
        outputs["bodyType_raw"] = body_logits
        
        # Keep track of attention weights
        outputs["attention_weights"] = attn_weights
        
        return outputs
