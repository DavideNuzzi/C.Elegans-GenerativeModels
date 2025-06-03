import numpy as np
from abc import ABC, abstractmethod

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from typing import List, Tuple


# ---------------------------- utility: KL(q||p) ---------------------------

def kl_diag_gauss(mu_q, logv_q, mu_p, logv_p):
    v_q, v_p = logv_q.exp(), logv_p.exp()
    kl = 0.5 * (logv_p - logv_q + (v_q + (mu_q - mu_p).pow(2)) / v_p - 1)
    return kl.sum(dim=-1)                


# --------------------------- Kalman‑VAE module ----------------------------

class KVAE(nn.Module):

    def __init__(self, features_dim: int, latent_dim: int = 8, encoder=None, decoder=None):
        super().__init__()
        self.features_dim, self.latent_dim = features_dim, latent_dim

        # Gaussian encoder q_φ(z|x)
        if encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(features_dim, 32), nn.Tanh(),
                nn.Linear(32, 2 * latent_dim)
            )
        else:
            self.encoder = encoder

        # Decoder p_θ(x|z)
        if decoder is None:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32), nn.Tanh(),
                nn.Linear(32, features_dim)
            )
        else:
            self.decoder = decoder

        # latent dynamics z_{t+1}=Az_t+ε,   ε~N(0,Q)
        self.A = nn.Parameter(0.5 * torch.eye(latent_dim))
        self.logvar_Q = nn.Parameter(torch.zeros(latent_dim))
        self.logvar_R = nn.Parameter(torch.zeros(features_dim))

        # self.logvar_Q = nn.Parameter(-10*torch.ones(latent_dim))
        # self.logvar_R = nn.Parameter(-10*torch.ones(features_dim))
        # self.logvar_Q.requires_grad_(False)
        # self.logvar_R.requires_grad_(False)


    def forward(self, x: torch.Tensor):

        # If the data is unbatched (T,D), make it a single batch
        if x.ndim == 2: x = x.unsqueeze(0)

        B, T, _ = x.shape

        # Apply the encoder to the observations and split into mean and logvar
        encoder_output = self.encoder(x)
        mu_q, logvar_q = torch.chunk(encoder_output, 2, -1)
        
        # Clamp the logvar
        logvar_q = logvar_q.clamp(-7, 7)

        # Sample gaussian noise and construct the hidden state
        eps  = torch.randn_like(mu_q)
        z_s  = mu_q + eps * (0.5 * logvar_q).exp()

        # Decode back the observations
        x_hat = self.decoder(z_s)

        # Here, if needed, I can squeeze back the variables if batch_dim = 1
        # x_hat, mu_q, lv_q, z_s = [t.squeeze(0) for t in (x_hat, mu_q, logvar_q, z_s)]
        
        return x_hat, mu_q, logvar_q, z_s

    
    # ------------------------------------------------------------------
    def elbo(self, x: torch.Tensor) -> torch.Tensor:

        # If the data is unbatched (T,D), make it a single batch
        if x.ndim == 2: x = x.unsqueeze(0)

        x_hat, mu_q, logvar_q, _ = self.forward(x)  

        # Reconstruction term

        # Reconstructiong error (gaussian)
        recon = -0.5 * (((x_hat - x) ** 2) / self.logvar_R.exp()).sum(dim=-1)
        recon = recon - 0.5 * (self.features_dim * math.log(2 * math.pi) + self.logvar_R.sum())
        
        # Sum over time
        recon = recon.sum(dim=-1)    

        # KL term

        # Compute the first term in the sequence
        kl_sum = kl_diag_gauss(mu_q[:, 0], logvar_q[:, 0],
                               torch.zeros_like(mu_q[:, 0]), torch.zeros_like(logvar_q[:, 0]))
        
        # Compute all other terms
        # Calculate the average for the observation (one step forward in time wrt to z)
        # If I detach the gradients do not flow back in time through the KL loss
        # which is the original KVAE formulation, but maybe it is better in the other way
        # mu_p = (mu_q[:,:-1,:].detach() @ self.A.T)
        mu_p = (mu_q[:,:-1,:] @ self.A.T)        

        # Expand the logvar to match the size
        logvar_p = self.logvar_Q.view(1, 1, -1).expand_as(mu_p)

        # Compute the KL for the rest of the sequence
        kl_sum += kl_diag_gauss(mu_q[:, 1:,:], logvar_q[:, 1:,:], mu_p, logvar_p).sum(axis=-1)    

        # Evidence lower bound, averaged over batches
        elbo = (recon - kl_sum).mean(dim=0)

        return elbo, recon, kl_sum

    # ------------------------------------------------------------------
    def fit(self, dataloader, optimizer, epochs: int = 100, show_progress: bool = True, callback = None) -> List[float]:
        
        stats_history = {'elbo': [], 'recon_loss': [], 'kl_loss': []}

        pbar = tqdm(range(epochs), disable=not show_progress)

        for epoch in pbar:

            stats_epoch = {'elbo': 0, 'recon_loss': 0, 'kl_loss': 0}
            
            for x in dataloader:
                optimizer.zero_grad()
                elbo, recon, kl = self.elbo(x)
                loss = -elbo
                loss.backward()
                optimizer.step()

                stats_epoch['elbo'] += elbo.item()
                stats_epoch['recon_loss'] -= recon.mean().item()
                stats_epoch['kl_loss'] += kl.mean().item()

            for s in stats_epoch:
                stats_epoch[s] /= len(dataloader)
                stats_history[s].append(stats_epoch[s])

            pbar.set_postfix(stats_epoch)            

            if callback is not None:
                if (epoch + 1) % callback[0] == 0:
                    callback[1](epoch)
                    
        return stats_history


    # ------------------------------------------------------------------
    # @torch.no_grad()
    # def generate(self, steps: int, z0=None, burn_in: int = 20):

    #     # Start with a random latent
    #     if z0 is None:
    #         device = next(self.parameters()).device
    #         z = torch.randn(self.latent_dim, device=device)
    #     else:
    #         device = z0.device
    #         z = z0
            
    #     x_output = []

    #     # Start generation
    #     for t in range(steps + burn_in):

    #         # Linear evolution for the latent state
    #         z = self.A @ z + torch.randn_like(z) * self.logvar_Q.exp().sqrt()

    #         # Observation model
    #         x = self.decoder(z) + torch.randn(self.features_dim, device=device) * self.logvar_R.exp().sqrt()

    #         # Save only after the burn-in time
    #         if t >= burn_in:
    #             x_output.append(x)

    #     return torch.stack(x_output)  # (steps,D)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        steps: int,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Roll the trained KVAE forward for `steps` time-steps and return the
        newly generated observations.

        Parameters
        ----------
        steps : int
            Number of observations to generate.
        x0 : torch.Tensor | None, default None
            • If **None**   → start from a random latent  z₀ ~ N(0, I).  
            • If tensor      → conditioning prefix  (shape (..., features)).
            Only the **last** frame is encoded to obtain q(z₀ | x_last).

        Returns
        -------
        torch.Tensor
            Generated sequence of shape (steps, features).  The conditioning
            frames `x0` (if any) are *not* included in the output.
        """
        device = next(self.parameters()).device

        # Obtain initial latent z₀
        if x0 is None:
            z = torch.randn(self.latent_dim, device=device)          # prior draw
        else:
            x0 = x0.to(device)
            # Accept shape (features,) or (T0, features)
            x_last = x0 if x0.ndim == 1 else x0[-1]
            enc_out = self.encoder(x_last.unsqueeze(0))              
            mu, logvar = torch.chunk(enc_out, 2, dim=-1)

            # Could do sampling, but simpler to just take the mean
            # eps = torch.randn_like(mu)
            # z = (mu + eps * (0.5 * logvar).exp()).squeeze(0)         
            z = mu.squeeze(0)

        # Pre-compute stds for speed
        std_Q = self.logvar_Q.exp().sqrt()         # (d,)
        std_R = self.logvar_R.exp().sqrt()         # (features,)

        # Roll forward and collect observations
        generated = []
        for _ in range(steps):
            # latent dynamics
            z = self.A @ z + torch.randn_like(z) * std_Q
            # observation
            x = self.decoder(z) + torch.randn(self.features_dim, device=device) * std_R
            generated.append(x)

        return torch.stack(generated)        # (steps, features)
