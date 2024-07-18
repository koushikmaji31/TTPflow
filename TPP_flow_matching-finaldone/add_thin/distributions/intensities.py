from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import warnings
from torch.distributions import MixtureSameFamily
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import torch.nn.functional as F
from add_thin.data import Batch
from add_thin.distributions.densities import DISTRIBUTIONS
from add_thin.backbones.embeddings import Intensity_MLP, N_embed
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
import torchdiffeq

patch_typeguard()


@typechecked
class MixtureIntensity(nn.Module):
    """
    Class parameterizing the intensity function as a weighted mixture of distributions.

    Parameters:
    ----------
    n_components : int, optional
        Number of components to use in the mixture, by default 10
    embedding_size : int, optional
        Size of the event embedding, by default 128
    distribution : str, optional
        Distribution to use for the components, by default "normal"

    """

    def __init__(
        self,
        n_components: int = 10,
        embedding_size: int = 128,
        distribution: str = "normal",
        simulation_steps: int = 10,
    ) -> None:
        super().__init__()


        self.rejections_sample_multiple = 20
        self.FM = ConditionalFlowMatcher(sigma=0.0)
        self.mlp = Intensity_MLP(dim= 3 * embedding_size,
                                 w = 6 * embedding_size,
                                 out_dim= embedding_size,
                                 time_varying=True)
        self.time_embed = N_embed(d_model=embedding_size)
        self.simulation_steps = simulation_steps



    def log_likelihood(
        self,
        x_0: Batch,
        x_n_emb,
        flow_time_emb,
        x_n: Batch,
    ):
        """
        Compute the log-likelihood of the event sequences.

        Parameters:
        ----------
        x_0 : Batch
            Batch of event sequences
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        x_n : Batch
            Batch of event sequences to condition on

        Returns:
        -------
        log_likelihood: TensorType[float, "batch"]
            The log-likelihood of the event sequences
        """
        avg_emb = x_n_emb.sum(1) / torch.clamp(x_n.mask.sum(-1)[..., None], min=1)
        combined_emb = torch.cat([avg_emb, flow_time_emb], dim=-1)
        thin_time = x_0.time
        thin_time = thin_time / x_0.tmax
        time_embs = self.time_embed(thin_time)
        x_0_thin_mask = x_0.mask.unsqueeze(-1)
        t_0 = torch.rand_like(time_embs)
        t_1 = time_embs
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0=t_0, x1=t_1)

        vt = self.mlp(xt, combined_emb, t)
        stage_loss = (vt - ut) ** 2
        stage_loss = stage_loss * x_0_thin_mask
        loss_time = torch.mean(stage_loss)
        decoded_time = self.time_embed.decode(time_embs).squeeze()
        decoded_time_loss = F.mse_loss(decoded_time, thin_time.squeeze())
        total_time_loss = loss_time + decoded_time_loss

        return total_time_loss

    def sample(
        self,
        x_n_emb,
        flow_time_emb,
        x_n: Batch,
        seq_lens,
    ) -> Batch:
        """
        Sample event sequences from the intensity function.

        Parameters:
        ----------
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        n_samples : int
            Number of samples to draw
        x_n : Batch
            Batch of event sequences to condition on

        Returns:
        -------
        Batch
            The sampled event sequences
        """
        tmax = x_n.tmax
        avg_emb = x_n_emb.sum(1) / torch.clamp(x_n.mask.sum(-1)[..., None], min=1)
        combined_emb = torch.cat([avg_emb, flow_time_emb], dim=-1)
        sequence_len = seq_lens

        max_seq_len = int(sequence_len.max().detach().item()) + 1
        b = avg_emb.shape[0]
        d = flow_time_emb.shape[-1]

        while True:

            traj = torchdiffeq.odeint(
                lambda t, x: self.mlp.forward(x, combined_emb, t.repeat(x.shape[0], )),
                torch.rand(size=(b, self.rejections_sample_multiple * max_seq_len, d), device=avg_emb.device),
                torch.linspace(0, 1, 5, device=avg_emb.device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )

            out = traj[-1, :, :]
            out = self.time_embed.decode(out).squeeze() * tmax
            times = out.detach()


            # Reject if not in [0, tmax]
            inside = torch.logical_and(times <= tmax, times >= 0)
            sort_idx = torch.argsort(
                inside.int(), stable=True, descending=True, dim=-1
            )
            inside = torch.take_along_dim(inside, sort_idx, dim=-1)[
                :, :max_seq_len
            ]
            times = torch.take_along_dim(times, sort_idx, dim=-1)[
                :, :max_seq_len
            ]

            # Randomly mask out events exceeding the actual sequence length
            mask = (
                torch.arange(0, times.shape[-1], device=times.device)[None, :]
                < sequence_len[:, None]
            )
            mask = mask * inside

            if (mask.sum(-1) == sequence_len).all():
                break
            else:
                self.rejections_sample_multiple += 1
                warnings.warn(
                    f"""
Rejection sampling multiple increased to {self.rejections_sample_multiple}, as not enough event times were inside [0, tmax].
""".strip()
                )
                print("Rejection sampling multiple increased to", self.rejections_sample_multiple)

        times = times * mask

        return Batch.remove_unnescessary_padding(
            time=times, mask=mask, tmax=tmax, kept=None
        )