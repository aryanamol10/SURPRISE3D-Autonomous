"""Video feature encoder utilities."""

import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    """Encode raw video frames or precomputed per-frame features."""

    def __init__(
        self,
        input_dim,
        d_model,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        raw_frame_channels=3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.raw_frame_encoder = nn.Sequential(
            nn.Conv2d(raw_frame_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.raw_frame_projection = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
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

    def _encode_features(self, video_features, video_mask, video_timestamps):
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
        temporal_input = self.feature_projection(temporal_input)
        return temporal_input, video_mask

    def _encode_frames(self, video_frames, video_mask, video_timestamps):
        if video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(0)

        if video_frames.dim() != 5:
            raise ValueError("video_frames must have shape (B, T, C, H, W) or (T, C, H, W)")

        batch_size, num_frames = video_frames.shape[:2]
        if video_mask is None:
            video_mask = torch.ones(
                (batch_size, num_frames),
                dtype=torch.bool,
                device=video_frames.device,
            )

        if video_timestamps is None:
            steps = torch.arange(
                num_frames,
                device=video_frames.device,
                dtype=video_frames.dtype,
            )
            video_timestamps = steps.unsqueeze(0).expand(batch_size, -1)

        frames = video_frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0

        flat_frames = frames.reshape(batch_size * num_frames, *frames.shape[2:])
        frame_embeddings = self.raw_frame_encoder(flat_frames).flatten(1)
        frame_embeddings = frame_embeddings.reshape(batch_size, num_frames, -1)

        normalized_timestamps = self._normalize_timestamps(video_timestamps, video_mask)
        temporal_input = torch.cat([frame_embeddings, normalized_timestamps], dim=-1)
        temporal_input = self.raw_frame_projection(temporal_input)
        return temporal_input, video_mask

    def forward(self, video_features=None, video_frames=None, video_mask=None, video_timestamps=None):
        """Return framewise and pooled video embeddings."""
        if video_frames is not None:
            temporal_input, video_mask = self._encode_frames(video_frames, video_mask, video_timestamps)
        elif video_features is not None:
            temporal_input, video_mask = self._encode_features(video_features, video_mask, video_timestamps)
        else:
            raise ValueError("Either video_frames or video_features must be provided")

        if video_mask is None or not video_mask.any():
            empty = temporal_input.new_zeros(temporal_input.shape)
            pooled = temporal_input.new_zeros(temporal_input.shape[0], temporal_input.shape[-1])
            return empty, pooled

        encoded = self.temporal_encoder(
            temporal_input,
            src_key_padding_mask=~video_mask,
        )

        encoded = encoded * video_mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1) / video_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return encoded, pooled