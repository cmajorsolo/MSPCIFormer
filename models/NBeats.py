"""
N-BEATS Model.
Adapted from https://github.com/ServiceNow/N-BEATS for the MSPCIFormer framework.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, input_size, theta_size: int, basis_function: nn.Module,
                 layers: int, layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)] +
            [nn.Linear(in_features=layer_size, out_features=layer_size)
             for _ in range(layers - 1)]
        )
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(backcast_size, dtype=float) / backcast_size, i)[None, :]
                 for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)
        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(forecast_size, dtype=float) / forecast_size, i)[None, :]
                 for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)

    def forward(self, theta: torch.Tensor):
        backcast = torch.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32) / harmonics
        )[None, :]
        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32), requires_grad=False)
        self.backcast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32), requires_grad=False)
        self.forecast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32), requires_grad=False)
        self.forecast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum(
            'bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic], self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum(
            'bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = torch.einsum(
            'bp,pt->bt', theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum(
            'bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic], self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos
        return backcast, forecast


def _build_generic(input_size: int, output_size: int, stacks: int, layers: int, layer_size: int) -> NBeats:
    '''
    Create N-BEATS generic model
    '''
    return NBeats(nn.ModuleList([
        NBeatsBlock(input_size=input_size,
                    theta_size=input_size + output_size,
                    basis_function=GenericBasis(backcast_size=input_size, forecast_size=output_size),
                    layers=layers,
                    layer_size=layer_size)
        for _ in range(stacks)
    ]))


def _build_interpretable(input_size: int, output_size: int,
                          trend_blocks: int, trend_layers: int, trend_layer_size: int,
                          degree_of_polynomial: int,
                          seasonality_blocks: int, seasonality_layers: int,
                          seasonality_layer_size: int, num_of_harmonics: int) -> NBeats:
    """
    Create N-BEATS interpretable model.
    """
    trend_block = NBeatsBlock(
        input_size=input_size,
        theta_size=2 * (degree_of_polynomial + 1),
        basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                  backcast_size=input_size, forecast_size=output_size),
        layers=trend_layers,
        layer_size=trend_layer_size)
    seasonality_block = NBeatsBlock(
        input_size=input_size,
        theta_size=4 * int(np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
        basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                        backcast_size=input_size, forecast_size=output_size),
        layers=seasonality_layers,
        layer_size=seasonality_layer_size)
    return NBeats(nn.ModuleList(
        [trend_block for _ in range(trend_blocks)] +
        [seasonality_block for _ in range(seasonality_blocks)]
    ))


class Model(nn.Module):
    """
    N-BEATS wrapped for the MSPCIFormer framework.

    Supports multivariate forecasting by applying independent N-BEATS stacks
    per channel. Accepts configs like all other models in the framework.

    Additional optional configs (with defaults):
        nbeats_type (str): 'generic' (default) or 'interpretable'
        -- generic --
        nbeats_stacks (int): number of generic blocks (default 30)
        nbeats_layers (int): FC layers per block (default 4)
        nbeats_layer_size (int): hidden size per FC layer (default 512)
        -- interpretable --
        nbeats_trend_blocks (int): number of trend blocks (default 3)
        nbeats_trend_layers (int): FC layers per trend block (default 4)
        nbeats_trend_layer_size (int): hidden size for trend (default 256)
        nbeats_degree_of_polynomial (int): polynomial degree for trend (default 3)
        nbeats_seasonality_blocks (int): number of seasonality blocks (default 3)
        nbeats_seasonality_layers (int): FC layers per seasonality block (default 4)
        nbeats_seasonality_layer_size (int): hidden size for seasonality (default 2048)
        nbeats_num_of_harmonics (int): harmonics for seasonality (default 1)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        nbeats_type = getattr(configs, 'nbeats_type', 'generic')

        if nbeats_type == 'interpretable':
            self.model = _build_interpretable(
                input_size=configs.seq_len,
                output_size=configs.pred_len,
                trend_blocks=getattr(configs, 'nbeats_trend_blocks', 3),
                trend_layers=getattr(configs, 'nbeats_trend_layers', 4),
                trend_layer_size=getattr(configs, 'nbeats_trend_layer_size', 256),
                degree_of_polynomial=getattr(configs, 'nbeats_degree_of_polynomial', 3),
                seasonality_blocks=getattr(configs, 'nbeats_seasonality_blocks', 3),
                seasonality_layers=getattr(configs, 'nbeats_seasonality_layers', 4),
                seasonality_layer_size=getattr(configs, 'nbeats_seasonality_layer_size', 2048),
                num_of_harmonics=getattr(configs, 'nbeats_num_of_harmonics', 1),
            )
        else:
            self.model = _build_generic(
                input_size=configs.seq_len,
                output_size=configs.pred_len,
                stacks=getattr(configs, 'nbeats_stacks', 30),
                layers=getattr(configs, 'nbeats_layers', 4),
                layer_size=getattr(configs, 'nbeats_layer_size', 512),
            )

    def forward(self, x):
        # x: [Batch, seq_len, enc_in]
        B, L, C = x.shape
        # N-BEATS operates on univariate sequences; apply per channel
        # Reshape to [Batch * enc_in, seq_len] to process all channels in one pass
        x_flat = x.permute(0, 2, 1).reshape(B * C, L)          # [B*C, L]
        mask = torch.ones(B * C, L, device=x.device)            # all valid
        out_flat = self.model(x_flat, mask)                      # [B*C, pred_len]
        out = out_flat.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]
        return out
