"""Video feature encoder utilities."""

import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    """Encode a sequence of video features with optional timestamp inputs."""

    def __init__(self, input_dim, d_model, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.frame_projection = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def _normalize_timestamps(video_timestamps, video_mask):
        timestamps = video_timestamps.float()
        if timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)

        if video_mask is None:
            denom = timestamps.amax(dim=1, keepdim=True).clamp(min=1.0)
        else:
            masked = timestamps.masked_fill(~video_mask.unsqueeze(-1), 0.0)
            denom = masked.amax(dim=1, keepdim=True).clamp(min=1.0)
        return timestamps / denom

    def forward(self, video_features, video_mask=None, video_timestamps=None):
        """Return framewise and pooled video embeddings."""
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(0)

        if video_mask is None:
            video_mask = torch.ones(
                video_features.shape[:2],
                dtype=torch.bool,
                device=video_features.device,
            )

        if video_timestamps is None:
            steps = torch.arange(
                video_features.shape[1],
                device=video_features.device,
                dtype=video_features.dtype,
            )
            video_timestamps = steps.unsqueeze(0).expand(video_features.shape[0], -1)

        normalized_timestamps = self._normalize_timestamps(video_timestamps, video_mask)
        temporal_input = torch.cat([video_features, normalized_timestamps], dim=-1)
        temporal_input = self.frame_projection(temporal_input)

        encoded = self.temporal_encoder(
            temporal_input,
            src_key_padding_mask=~video_mask,
        )

        encoded = encoded * video_mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1) / video_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return encoded, pooled