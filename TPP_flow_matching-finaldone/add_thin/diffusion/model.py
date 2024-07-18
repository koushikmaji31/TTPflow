import math
import torch
import torch.nn as nn

from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import torch.nn.functional as F

from add_thin.data import Batch
from add_thin.backbones.cnn import CNNSeqEmb
from add_thin.backbones.embeddings import NyquistFrequencyEmbedding
from add_thin.processes.hpp import generate_hpp
from add_thin.diffusion.utils import betas_for_alpha_bar
from add_thin.backbones.embeddings import Sequence_Embed, N_embed, MLP_Density, MLP_Forecast
import torchdiffeq
from torchcfm.conditional_flow_matching import *

from torchcfm.models.models import *


patch_typeguard()


@typechecked
class DiffusionModell(nn.Module):
    """
    Base class for diffusion models.

    Parameters
    ----------
    steps : int, optional
        Number of diffusion steps, by default 100
    """

    def __init__(self, steps: int = 10) -> None:
        super().__init__()
        self.steps = steps


@typechecked
class AddThin(DiffusionModell):
    """
    Implementation of AddThin (Add and Thin: Diffusion for Temporal Point Processes).

    Parameters
    ----------
    classifier_model : nn.Module
        Model for predicting the intersection of x_0 and x_n from x_n
    intensity_model : nn.Module
        Model for predicting the intensity of x_0 without x_n
    max_time : float
        T of the temporal point process
    n_max : int, optional
        Maximum number of events, by default 100
    steps : int, optional
        Number of diffusion steps, by default 100
    hidden_dims : int, optional
        Hidden dimensions of the models, by default 128
    emb_dim : int, optional
        Embedding dimensions of the models, by default 32
    encoder_layer : int, optional
        Number of encoder layers, by default 4
    kernel_size : int, optional
        Kernel size of the CNN, by default 16
    forecast : None, optional
        If not None, will turn the model into a conditional one for forecasting
    """

    def __init__(
        self,
        classifier_model,
        intensity_model,
        max_time: float,
        n_max: int = 1,
        n_min: int = 0,
        steps: int = 100,
        hidden_dims: int = 128,
        emb_dim: int = 32,
        encoder_layer: int = 4,
        kernel_size: int = 16,
        forecast=None,
        simulation_steps = 5,
    ) -> None:
        super().__init__(steps)
        # Set models parametrizing the approximate posterior
        self.intensity_model = intensity_model
        self.simulation_steps = simulation_steps
        self.n_max = n_max
        self.n_min = n_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.FM = ConditionalFlowMatcher(sigma=0)
        self.emb_dim = emb_dim
        self.n_max = n_max

        # Init forecast settings
        if forecast:
            self.forecast = True
            self.history_encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim,
                batch_first=True,
            )
            self.history_mlp = nn.Sequential(
                nn.Linear(2 * emb_dim, hidden_dims), nn.ReLU()
            )
            self.forecast_window = forecast
        else:
            self.forecast = False
            self.history = None



        self.set_encoders(
            hidden_dims=hidden_dims,
            max_time=max_time,
            emb_dim=emb_dim,
            encoder_layer=encoder_layer,
            kernel_size=kernel_size,
            steps=steps,
        )

    def set_encoders(
            self,
            hidden_dims: int,
            max_time: float,
            emb_dim: int,
            encoder_layer: int,
            kernel_size: int,
            steps: int,
            ) -> None:
        """
        Set the encoders for the model.

        Parameters
        ----------
        hidden_dims : int
            Hidden dimensions of the models
        max_time : float
            T of the temporal point process
        emb_dim : int
            Embedding dimensions of the models
        encoder_layer : int
            Number of encoder layers
        kernel_size : int
            Kernel size of the CNN
        steps : int
            Number of diffusion steps
        """
        # Event time encoder
        position_emb = NyquistFrequencyEmbedding(
            dim=emb_dim // 2, timesteps=max_time
        )

        self.embedding_matrix = nn.Embedding(2, emb_dim)
        self.time_encoder = nn.Sequential(position_emb)

        # Diffusion time encoder
        position_emb = NyquistFrequencyEmbedding(dim=emb_dim, timesteps=steps)
        self.diffusion_time_encoder = nn.Sequential(
            position_emb,
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Event sequence encoder
        self.sequence_encoder = CNNSeqEmb(
            emb_layer=encoder_layer,
            input_dim=hidden_dims,
            emb_dims=hidden_dims,
            kernel_size=kernel_size,
        )


        self.process_time_embed = N_embed(emb_dim)

        if self.forecast:
            self.mlp = MLP_Forecast(emb_dim, time_varying=True)

        else:
            self.mlp = MLP_Density(emb_dim, time_varying=True)




    def set_history(self, batch: Batch) -> None:
        """
        Set the history to condition the model.

        Parameters
        ----------
        batch : Batch
            Batch of data
        """
        # Compute history embedding
        B, L = batch.time.shape

        # Encode event times
        time_emb = self.time_encoder(
            torch.cat(
                [batch.time.unsqueeze(-1), batch.tau.unsqueeze(-1)], dim=-1
            )
        ).reshape(B, L, -1)

        # Compute history embedding
        embedding = self.history_encoder(time_emb)[0]

        # Index relative to time and set history
        index = (batch.mask.sum(-1).long() - 1).unsqueeze(-1).unsqueeze(-1)
        gather_index = index.repeat(1, 1, embedding.shape[-1])
        self.history = embedding.gather(1, gather_index).squeeze(-2)

    def compute_emb(
        self,
            n,
            x_n,
    ):
        """
        Get the embeddings of x_n.

        Parameters
        ----------
        n : TensorType[torch.long, "batch"]
            Diffusion time step
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType["batch", "embedding"],
            TensorType["batch", "sequence", "embedding"],
            TensorType["batch", "sequence", "embedding"],
        ]
            Diffusion time embedding, event time embedding, event sequence embedding
        """

        B, L = x_n.batch_size, x_n.seq_len

        # embed diffusion and process time
        dif_time_emb = self.diffusion_time_encoder(n)

        # Condition ADD-THIN on history by adding it to the diffusion time embedding
        if self.forecast:
            dif_time_emb = self.history_mlp(
                torch.cat([self.history, dif_time_emb], dim=-1)
            )

        # Embed event and interevent time
        time_emb = self.time_encoder(
            torch.cat([x_n.time.unsqueeze(-1), x_n.tau.unsqueeze(-1)], dim=-1)
        ).reshape(B, L, -1)

        # Embed event sequence and mask out
        kept_indices = x_n.kept.long()
        kept_embs = self.embedding_matrix(kept_indices)
        event_emb = self.sequence_encoder(time_emb)
        event_emb = event_emb + kept_embs

        event_emb = event_emb * x_n.mask[..., None]



        return (
            dif_time_emb,
            time_emb,
            event_emb,
        )

    def thin_from_right_masks(self, x_0, alpha):
        x_0_unpadded = x_0.unpadded_length
        kept_lengths = (alpha * x_0_unpadded).round().long()

        # Create a mask for the indices to keep
        max_sequence_length = x_0.mask.size(1)
        indices_to_keep = torch.arange(max_sequence_length, device=x_0.mask.device)[None, :] >= (x_0_unpadded - kept_lengths)[:, None]
        # Expand the mask to match the shape of x_0.mask
        expanded_mask = indices_to_keep.expand_as(x_0.mask)

        # Apply the mask to x_0.mask
        new_masks = x_0.mask * expanded_mask.float()

        return new_masks.float()

    def thin_from_left_masks(self, x_0, alpha):
        x_0_unpadded = x_0.unpadded_length
        kept_lengths = (alpha * x_0_unpadded).round().long()

        # Create a mask for the indices to keep
        max_sequence_length = x_0.mask.size(1)
        indices_to_keep = torch.arange(max_sequence_length, device=x_0.mask.device)[None, :] < kept_lengths[:, None]

        # Expand the mask to match the shape of x_0.mask
        expanded_mask = indices_to_keep.expand_as(x_0.mask)

        # Apply the mask to x_0.mask
        new_masks = x_0.mask * expanded_mask.float()

        return new_masks.float()

    def noise(
        self, x_0, n
    ):
        """
        Sample x_n from x_0 by applying the noising process.

        Parameters
        ----------
        x_0 : Batch
            Batch of data
        n : TensorType[torch.long, "batch"]
            Number of noise steps

        Returns
        -------
        Tuple[Batch, Batch]
            x_n and thinned x_0
        """
        # Thin


        n_prob = n
        sp = n_prob * self.simulation_steps
        new_masks = self.thin_from_right_masks(x_0, n_prob).float()
        x_0_kept, x_0_thinned = x_0.thin(alpha= new_masks)
        new_left= self.thin_from_right_masks(x_0_thinned, 1/(self.simulation_steps - sp))
        x_0_thinned, _ = x_0_thinned.thin(alpha=new_left)

        # Superposition with HPP (add)
        hpp = generate_hpp(
            tmax=x_0.tmax,
            n_sequences=len(x_0),
        )

        new_masks = self.thin_from_left_masks(hpp, alpha=1 - n_prob)
        hpp_kept, hpp_thinned = hpp.thin(alpha=new_masks)
        x_n = hpp_kept.add_events(x_0_kept)
        x_n.kept = (x_n.kept.float() * x_n.mask.float()).bool()

        return x_n, x_0_thinned, x_0_kept, hpp_thinned, hpp_kept

    def forward(
        self, x_0
    ):
        """
        Forward pass to train the model, i.e., predict x_0 from x_n.

        Parameters
        ----------
        x_0 : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType[float, "batch", "sequence_x_n"],
            TensorType[float, "batch"],
            Batch,
        ]
            classification logits, log likelihood of x_0 without x_n, noised data
        """
        # Uniformly sample n


        #pick 1 timestep between 0 to 1. it can be any value between that.  make it like the n in the docstrings above.
        n = torch.randint(low=0, high=self.simulation_steps, size=(len(x_0),), device=x_0.time.device)
        n = n.float() / self.simulation_steps
        x_n, x_0_thin, x_0_kept, hpp_thinned, hpp_kept = self.noise(x_0=x_0, n=n)

        unpadded_lens = x_0.unpadded_length.float() / self.n_max
        x_1 = self.process_time_embed(unpadded_lens)
        decoded_lens = self.process_time_embed.decode(x_1).squeeze()
        (flow_time_emb, tp_emb,x_n_emb) = self.compute_emb(n=n * self.simulation_steps, x_n=x_n)
        x_0 = torch.rand_like(x_1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0=x_0, x1=x_1)
        t = t.to(self.device)
        xt = xt.to(self.device)
        ut = ut.to(self.device)
        if self.forecast:
            vt = self.mlp(xt, self.history, t)
        else:
            vt = self.mlp(xt, t)
        decode_loss = F.mse_loss(decoded_lens, unpadded_lens)
        fm_loss = torch.mean((vt - ut) ** 2)
        loss = fm_loss + decode_loss



        log_like_x_0 = self.intensity_model.log_likelihood(
            x_n_emb=x_n_emb,
            flow_time_emb=flow_time_emb,
            x_0=x_0_thin,
            x_n=x_n,
        )

        return log_like_x_0, x_n, loss

    def sample(self, n_samples: int, tmax, seqs=None) -> Batch:
        """
        Sample x_0 from ADD-THIN starting from x_N.

        Parameters
        ----------
        n_samples : int
            Number of samples
        tmax : float
            T of the temporal point process
        begin_forecast : None, optional
            Beginning of the forecast, by default None
        end_forecast : None, optional
            End of the forecast, by default None

        Returns
        -------
        Batch
            Sampled x_0s
        """
        # Init x_N by sampling from HPP


        x_0 = generate_hpp(tmax=tmax, n_sequences=n_samples)
        true_x_0 = torch.ones_like(x_0.time).float()
        x_0.kept = (true_x_0 * x_0.mask.float()).bool()
        if self.forecast:
            traj = torchdiffeq.odeint(
                lambda t, x: self.mlp.forward(x, self.history, t.view(-1, ).repeat(x.shape[0],)),
                torch.rand(size=(n_samples, self.history.shape[-1]), device=self.device),
                torch.linspace(0, 1, 5, device=self.device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        else:
            traj = torchdiffeq.odeint(
                lambda t, x: self.mlp.forward(x, t.view(-1, ).repeat(x.shape[0],)),
                torch.rand(size=(n_samples, self.emb_dim), device=self.device),
                torch.linspace(0, 1, 5, device=self.device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )

        out = traj[-1, :, :]
        seq_len = self.process_time_embed.decode(out).squeeze() * self.n_max
        seq_len = seq_len.round().long()
        if seqs is not None:
            seq_len = seqs


        total_steps = self.simulation_steps


        for n_int in range(0, total_steps):

            n = torch.full(
                (n_samples,), n_int, device=tmax.device, dtype=torch.float
            )

            n = n/total_steps

            x_0 = self.sample_x_plus_1(x_n=x_0, n = n, total_steps = total_steps, seq_len = seq_len)


        return x_0

    def sample_x_plus_1(self, x_n: Batch, n, total_steps, seq_len) -> Batch:
        """
        Sample x_n+1 from x_n by predicting x_0 and then sampling from the posterior.

        Parameters
        ----------
        x_n : Batch
            Batch of data
        n : TensorType
            Diffusion time steps

        Returns
        -------
        Batch
            x_n+1
        """
        # Sample x_0 and x_n\x_0

        (flow_time_emb,  tp_emb, x_n_emb) = self.compute_emb(n=n* self.simulation_steps, x_n=x_n)
        #seq_lens = torch.round(seq_len * (1 - n))
        #sl = torch.round(seq_len * (1 - n))
        sl = torch.round(seq_len/self.simulation_steps)

        x_add = self.intensity_model.sample(
            x_n_emb=x_n_emb,
            flow_time_emb=flow_time_emb,
            x_n=x_n,
            seq_lens = sl)

        #seq_lens = torch.round(seq_len * (1 - n)
        #torch.round(seq_len/self.simulation_steps))

        #x_0_right_mask = self.thin_from_right_masks(x_add, alpha=1/(total_steps - (n*total_steps)))

        #x_add, _ = x_add.thin(alpha=x_0_right_mask)


        x_0_plus_one_hpp_mask = x_n.kept.float()

        from_hpp, not_hpp= x_n.thin(alpha = x_0_plus_one_hpp_mask)
        new_masks = self.thin_from_left_masks(from_hpp, alpha=1 - 1/(total_steps-n*total_steps))
        hpp_kept, _ = from_hpp.thin(alpha = new_masks)

        aded = not_hpp.add_events(x_add)

        aded.kept = None

        x_n_plus_1 = hpp_kept.add_events(aded)


        return x_n_plus_1
