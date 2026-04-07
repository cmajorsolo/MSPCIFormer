"""
LSTM Model
"""
import torch
import torch.nn as nn
import numpy as np
from layers.Embed import DataEmbedding, DataEmbeddingLSTM


class Model(nn.Module):
    """
    LSTM Model for Time Series Forecasting

    This model uses a standard LSTM architecture for comparison with Transformer models.

    Architecture:
    1. Input Embedding (Token + Positional + Temporal)
    2. LSTM Layers (Bidirectional or Unidirectional)
    3. Output Projection (Sequence + Feature)

    Parameters:
    - seq_len: Input sequence length
    - pred_len: Output prediction length
    - enc_in: Number of input features
    - c_out: Number of output features
    - d_model: Hidden dimension (LSTM hidden size)
    - n_layers: Number of LSTM layers (maps from e_layers)
    - dropout: Dropout rate
    - bidirectional: Whether to use bidirectional LSTM
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Store configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model

        # Map transformer parameters to LSTM parameters
        self.hidden_size = configs.d_model  # LSTM hidden size
        self.num_layers = getattr(configs, 'e_layers', 2)  # Number of LSTM layers
        self.dropout = configs.dropout
        self.bidirectional = getattr(configs, 'bidirectional', False)  # Optional bidirectional

        # Input embedding - same as Transformer for fair comparison
        self.enc_embedding = DataEmbeddingLSTM(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        # Calculate LSTM output dimension
        lstm_output_dim = self.hidden_size * (2 if self.bidirectional else 1)

        # Add manual dropout for single-layer LSTM
        self.manual_dropout = nn.Dropout(self.dropout)

        # Output projections
        # 1. Sequence projection: seq_len -> pred_len
        self.seq_projection = nn.Linear(self.seq_len, self.pred_len)

        # 2. Feature projection: lstm_hidden -> c_out
        self.feature_projection = nn.Linear(lstm_output_dim, self.c_out)

        # Optional: Additional layers for better feature transformation
        self.use_additional_layers = getattr(configs, 'use_additional_layers', False)
        if self.use_additional_layers:
            self.feature_transform = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(lstm_output_dim // 2, self.c_out)
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (LSTM best practice)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        Forward pass compatible with Transformer interface

        Args:
            x_enc: [Batch, seq_len, enc_in] - input sequences
            x_mark_enc: [Batch, seq_len, mark_dim] - temporal features
            x_dec: Not used in LSTM (kept for interface compatibility)
            x_mark_dec: Not used in LSTM (kept for interface compatibility)

        Returns:
            out: [Batch, pred_len, c_out] - predictions
        """

        # Normalization for price data
        means = x_enc.mean(1, keepdim=True).detach()  # Mean across time dimension
        x_enc = x_enc - means  # Center around 0
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev  # Scale by standard deviation

        # Input embedding (same as Transformer)
        # [Batch, seq_len, enc_in] -> [Batch, seq_len, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # LSTM forward pass
        # [Batch, seq_len, d_model] -> [Batch, seq_len, hidden_size * directions]
        lstm_out, (hidden, cell) = self.lstm(enc_out)

        # Apply manual dropout for single-layer LSTM
        lstm_out = self.manual_dropout(lstm_out)

        # Output projection strategy
        # We need to go from [Batch, seq_len, lstm_hidden] to [Batch, pred_len, c_out]

        # Step 1: Sequence projection (seq_len -> pred_len)
        # Transpose for linear layer: [Batch, lstm_hidden, seq_len]
        lstm_out_transposed = lstm_out.transpose(1, 2)
        # Apply sequence projection: [Batch, lstm_hidden, seq_len] -> [Batch, lstm_hidden, pred_len]
        seq_projected = self.seq_projection(lstm_out_transposed)
        # Transpose back: [Batch, lstm_hidden, pred_len] -> [Batch, pred_len, lstm_hidden]
        seq_projected = seq_projected.transpose(1, 2)

        # Step 2: Feature projection (lstm_hidden -> c_out)
        if self.use_additional_layers:
            # Use additional transformation layers
            out = self.feature_transform(seq_projected)
        else:
            # Direct linear projection
            out = self.feature_projection(seq_projected)

        # Denormalization
        out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        # Output: [Batch, pred_len, c_out]
        return out

    def get_parameter_count(self):
        """Calculate and return the total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Detailed breakdown
        embedding_params = sum(p.numel() for p in self.enc_embedding.parameters() if p.requires_grad)
        lstm_params = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        projection_params = (
            sum(p.numel() for p in self.seq_projection.parameters() if p.requires_grad) +
            sum(p.numel() for p in self.feature_projection.parameters() if p.requires_grad)
        )

        return {
            'total': total_params,
            'embedding': embedding_params,
            'lstm': lstm_params,
            'projections': projection_params
        }
