import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class NyquistFrequencyEmbedding(nn.Module):
    """
    Sine-cosine embedding for timesteps that scales from 1/8 to a (< 1) multiple of
    the Nyquist frequency.

    We choose 1/8 as the slowest frequency so that the slowest-varying embedding varies
    roughly lineary across [0, 2pi] as the relative error between x and sin(x) on [0,
    2pi / 8] is at most 2.5%. The Nyquist frequency is the largest frequency that one
    can sample at T steps without aliasing, so one could assume that to be a great
    choice for the highest frequency but sampling sine and cosine at the Nyquist
    frequency would result in constant (and therefore uninformative) 1 and 0 features,
    so we Nyquist/2 is a better choice. However, Nyquist/2 (which is T/2) leads to the
    evaluation points of the fastest varying points to overlap, so that those features
    would only take a small number of values, such as 2 or 4. In combination with the
    other points, these embeddings would of course still be distinguishable but by
    choosing an irrational fastest frequency, we can get unique embeddings also in the
    fastest-varying dimension for all timepoints. We choose arbitrarily 1/phi where phi
    is the golden ratio.

    Parameters
    ----------
    dim : int
        Number of dimensions of the embedding
    timesteps : int
        Number of timesteps to embed
    """

    def __init__(self, dim: int, timesteps: int | float) -> None:
        super().__init__()

        assert dim % 2 == 0

        T = timesteps
        k = dim // 2

        # Nyquist frequency for T samples per cycle
        nyquist_frequency = T / 2

        golden_ratio = (1 + np.sqrt(5)) / 2
        frequencies = np.geomspace(
            1 / 8, nyquist_frequency / (2 * golden_ratio), num=k
        )

        # Sample every frequency twice, once shifted by pi/2 to get cosine
        scale = np.repeat(2 * np.pi * frequencies / timesteps, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        self.register_buffer(
            "scale",
            torch.from_numpy(scale.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "bias", torch.from_numpy(bias.astype(np.float32)), persistent=False
        )

    def forward(self, t) -> torch.Tensor:
        return torch.addcmul(self.bias, self.scale, t[..., None]).sin()


class N_embed(nn.Module):
    def __init__(self, d_model):
        super(N_embed, self).__init__()
        self.d_model = d_model
        self.encoder_fc1 = nn.Linear(1, d_model)
        self.encoder_fc2 = nn.Linear(d_model, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x1 = F.relu(self.encoder_fc1(x))
        x2 = self.encoder_fc2(x1)
        result = x1 + x2
        return result

    def decode(self, x):
        x = self.decoder(x)
        return x


class Time_embed(nn.Module):
    def __init__(self, d_model):
        super(Time_embed, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        x_time = x.time
        result = x_time.unsqueeze(-1).repeat(1, 1, self.d_model)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        mask = x.mask.unsqueeze(-1).float()
        return result * mask

class MLP_Forecast(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=32, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, context, t):
        x = torch.cat([x, context, t[:, None]], dim=-1)
        return self.net(x)

class Intensity_MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, context, t):
        t = t.view(-1, 1, 1).repeat(1, x.shape[1], 1)
        context = context.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, context, t], dim=-1)

        return self.net(x)

class MLP_Density(torch.nn.Module):

    def __init__(self, dim, out_dim=None, w=32, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, t):
        x = torch.cat([x, t[:, None]], dim=-1)
        return self.net(x)


class Sequence_Embed(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 batch_first,
                 num_layers=3):
        super().__init__()

        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                                 nhead=nhead,
                                                                 dim_feedforward=dim_feedforward,
                                                                 dropout=dropout,
                                                                 batch_first=batch_first)

        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.time_encoder = Time_embed(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.ReLU(),
                                  nn.Linear(d_model, d_model),
                                  nn.ReLU(),
                                  )



    def forward(self, x):

        time_embeds = self.time_encoder(x)
        pad_mask = x.mask.unsqueeze(-1).float()
        b, seq_len, d_model = time_embeds.shape
        mask = torch.zeros((seq_len, seq_len), device=x.time.device)
        pad_mask = pad_mask.squeeze(-1)
        pad_mask[pad_mask == 0] = float('-inf')
        pad_mask[pad_mask == 1] = 0
        transformer_embeds = self.transformer_encoder(time_embeds,
                                                      mask=mask,
                                                      src_key_padding_mask=pad_mask)
        #avg_emb = transformer_embeds.sum(1) / torch.clamp(x.mask.sum(-1)[..., None], min=1)
        return transformer_embeds

