import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenTransformer(nn.Module):
    def __init__(self, feat_dim=8, embed_dim=128, n_layers=4, n_heads=4, mlp_dim=256, dropout=0.1, max_tokens=200, out_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        # token embedding: linear projection from feat_dim -> embed_dim
        self.token_embed = nn.Linear(feat_dim, embed_dim)
        # positional embedding for token positions (0..max_tokens-1)
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=mlp_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # output head: per-token regression to out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, out_dim)
        )
        self.max_tokens = max_tokens

    def forward(self, feats, mask):
        """
        feats: [B, T, feat_dim]
        mask: [B, T]  (1 -> valid, 0 -> pad)
        returns: preds [B, T, out_dim]
        """
        B, T, _ = feats.shape
        x = self.token_embed(feats)  # [B,T,embed]
        # add pos embedding for first T positions
        x = x + self.pos_embed[:, :T, :]
        # Transformer expects src_key_padding_mask with True for padded positions
        # mask_pad = (mask == 0) -> bool mask
        key_padding_mask = (mask == 0)  # [B, T], True at pad positions
        # pass through encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        preds = self.head(x)
        return preds