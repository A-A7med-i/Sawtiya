import torch
import torch.nn as nn


class CNNBiLSTMAttentionClassifier(nn.Module):
    """
    Hybrid CNN + BiLSTM + Attention network for emotion classification from audio features.

    Architecture:
        - CNN layers for spatial feature extraction
        - Bidirectional LSTM for temporal modeling
        - Simplified attention mechanism
        - Fully connected classifier with residual connections

    Args:
        num_emotions (int): Number of emotion classes.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, num_emotions: int = 8, dropout: float = 0.2):
        super().__init__()

        # Convolutional feature extractor
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Bidirectional LSTM for temporal dependencies
        self.temporal_encoder = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(512)

        # Simplified attention mechanism
        self.attention_layer = nn.Linear(512, 1)

        # Fully connected classifier with residual connections
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_emotions)

        # Dropout and activation
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, time, features)

        Returns:
            torch.Tensor: Output logits of shape (batch, num_emotions)
        """

        # CNN feature extraction
        features = self.cnn_encoder(x)

        # Adaptive pooling to preserve temporal resolution
        features = torch.nn.functional.adaptive_avg_pool2d(
            features, (features.size(2), 1)
        )
        features = features.squeeze(-1).permute(0, 2, 1)  # (batch, time, channels)

        # BiLSTM
        lstm_out, _ = self.temporal_encoder(features)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Attention mechanism
        attention_scores = self.attention_layer(lstm_out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)

        # Fully connected classifier with residual connections
        out = self.relu(self.fc1(context_vector))
        out = self.drop(out)
        out = self.relu(self.fc2(out))
        out = self.drop(out)
        logits = self.fc3(out)

        return logits
