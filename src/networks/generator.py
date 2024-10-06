from typing import Optional

import torch
import torch.nn as nn

from src.util.consts import N_FILTERS


class Generator(nn.Module):
    def __init__(self, device: str):
        super(Generator, self).__init__()

        # Encoder
        self.encoders = nn.ModuleList(
            [Generator._encode_layer(N_FILTERS[i], N_FILTERS[i + 1]) for i in range(11)]
        )

        # Decoder
        # Decoder levels have double the number of filters due to skip connections
        self.decoders = nn.ModuleList(
            [
                Generator._decode_layer(N_FILTERS[i + 1] * 2, N_FILTERS[i])
                for i in range(11)
            ]
        )

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        self.to(device)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        if z is None:
            z = nn.init.normal_(torch.Tensor(x.shape[0], 1024, 8).to(x.device))

        # Encoder
        encoder_outputs = []
        for i in range(11):
            x = self.encoders[i](x)
            encoder_outputs.append(x)
            x = self.prelu(x)

        # Combine encoded features with latent variable
        x = torch.cat([x, z], dim=1)

        # Decoder
        x = self.decoders[10](x)
        for i in range(9, -1, -1):
            x = torch.cat([x, encoder_outputs[i]], dim=1)
            x = self.prelu(x)
            x = self.decoders[i](x)

        x = self.tanh(x)

        return x

    @staticmethod
    def _encode_layer(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=31,
                stride=2,
                padding=15,
            ),
        )

    @staticmethod
    def _decode_layer(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=31,
                stride=2,
                padding=15,
                output_padding=1,
            ),
        )
