
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Optional

# ---------------------------- utility: KL(q||p) ---------------------------

def kl_diag_gauss(mu_q, logv_q, mu_p, logv_p):
    v_q, v_p = logv_q.exp(), logv_p.exp()
    kl = 0.5 * (logv_p - logv_q + (v_q + (mu_q - mu_p).pow(2)) / v_p - 1)
    return kl.sum(dim=-1)


# --------------------------- RNN Encoder module ---------------------------
class RNNEncoder(nn.Module):
    def __init__(self, features_dim: int, rnn_hidden_dim: int, output_dim: int, n_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(
            features_dim, 
            rnn_hidden_dim, 
            num_layers=n_layers,
            batch_first=True, 
            bidirectional=bidirectional
        )
        fc_in_dim = rnn_hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, features_dim)
        rnn_out, _ = self.rnn(x) 
        # rnn_out shape: (B, T, rnn_hidden_dim * num_directions)
        params_out = self.fc(rnn_out)
        # params_out shape: (B, T, output_dim) which is 2 * latent_dim
        return params_out

# --------------------------- Kalman‑VAE module ----------------------------

class KVAE(nn.Module):

    def __init__(self, 
                 features_dim: int, 
                 latent_dim: int = 8, 
                 encoder: Optional[nn.Module] = None, 
                 decoder: Optional[nn.Module] = None,
                 rnn_encoder_hidden_dim: int = 64,
                 max_spectral_radius: Optional[float] = None):
        super().__init__()
        self.features_dim, self.latent_dim = features_dim, latent_dim
        self.max_spectral_radius = max_spectral_radius

        # Gaussian encoder q_φ(z|x)
        if encoder is None:
            # Default RNN encoder
            self.encoder = RNNEncoder(features_dim, rnn_encoder_hidden_dim, 2 * latent_dim)
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

    def _constrain_A_matrix(self):
        if self.max_spectral_radius is not None and self.A.requires_grad:
            with torch.no_grad(): # Important: no gradient tracking for this op
                A_val = self.A.data
                try:
                    eigvals = torch.linalg.eigvals(A_val)
                    current_radius = torch.abs(eigvals).max()
                    # Only renormalize if current_radius is greater and max_spectral_radius is positive
                    if current_radius > self.max_spectral_radius and self.max_spectral_radius > 0:
                        self.A.data = A_val * (self.max_spectral_radius / current_radius)
                except torch.linalg.LinAlgError:
                    # This can happen if A is ill-conditioned, e.g., during early training
                    # print("Warning: Eigenvalue computation failed for matrix A. Skipping constraint.")
                    pass


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # If the data is unbatched (T,D), make it a single batch (1,T,D)
        if x.ndim == 2: x = x.unsqueeze(0)

        # Apply the encoder to the observations and split into mean and logvar
        # For RNN encoder, x is (B, T, features_dim), encoder_output is (B, T, 2*latent_dim)
        encoder_output = self.encoder(x) 
        mu_q, logvar_q = torch.chunk(encoder_output, 2, dim=-1)
        
        # Clamp the logvar
        logvar_q = logvar_q.clamp(-7, 7)

        # Sample gaussian noise and construct the hidden state (reparameterization trick)
        eps  = torch.randn_like(mu_q)
        z_s  = mu_q + eps * (0.5 * logvar_q).exp()

        # Decode back the observations
        x_hat = self.decoder(z_s)
        
        return x_hat, mu_q, logvar_q, z_s

    
    # ------------------------------------------------------------------
    def elbo(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Note: x is assumed to be on the correct device already by `fit` method

        # If the data is unbatched (T,D), make it a single batch
        if x.ndim == 2: x = x.unsqueeze(0) # Already done in forward

        x_hat, mu_q, logvar_q, _ = self.forward(x)  
        B, T, _ = x.shape # Original x shape

        # Reconstruction term E[log p(x|z)]
        # Reconstruction error (gaussian)
        recon_log_likelihood = -0.5 * (((x_hat - x) ** 2) / self.logvar_R.exp()).sum(dim=-1) # Sum over features_dim
        recon_log_likelihood = recon_log_likelihood - 0.5 * (self.features_dim * math.log(2 * math.pi) + self.logvar_R.sum())
        
        # Sum over time
        recon_term_per_batch_item = recon_log_likelihood.sum(dim=-1)  # Shape: (B,)

        # KL term: KL(q(z|x) || p(z))
        # KL_0 = KL(q(z_0|x_0) || p(z_0)) where p(z_0) ~ N(0,I)
        kl_0 = kl_diag_gauss(
            mu_q[:, 0], logvar_q[:, 0],
            torch.zeros_like(mu_q[:, 0]), torch.zeros_like(logvar_q[:, 0])
        )
        
        # KL_t = Sum_{t=1}^{T-1} KL(q(z_t|x_{1:t}) || p(z_t|z_{t-1}))
        # where p(z_t|z_{t-1}) ~ N(A mu_q(z_{t-1}), Q)
        # Using mu_q from encoder as proxy for z_{t-1} in the prior mean
        mu_p_transition = (mu_q[:, :-1, :] @ self.A.T)  

        # Expand the logvar_Q to match the size of mu_p_transition
        logvar_p_transition = self.logvar_Q.view(1, 1, -1).expand_as(mu_p_transition)

        # Compute the KL for the rest of the sequence
        kl_transition = kl_diag_gauss(
            mu_q[:, 1:, :], logvar_q[:, 1:, :], 
            mu_p_transition, logvar_p_transition
        ).sum(dim=-1) # Sum over time steps T-1

        kl_term_per_batch_item = kl_0 + kl_transition # Shape: (B,)

        # Evidence lower bound, averaged over batches
        elbo_mean = (recon_term_per_batch_item - kl_term_per_batch_item).mean(dim=0)

        return elbo_mean, recon_term_per_batch_item, kl_term_per_batch_item

    # ------------------------------------------------------------------
    def fit(self, dataloader, optimizer, epochs: int = 100, show_progress: bool = True, callback = None) -> dict:
        
        stats_history = {'elbo': [], 'recon_loss': [], 'kl_loss': []}
        device = next(self.parameters()).device # Get model's device

        pbar = tqdm(range(epochs), disable=not show_progress, desc="Epochs")

        for epoch in pbar:
            stats_epoch = {'elbo': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
            
            num_batches = 0
            for data_batch in dataloader:
                if isinstance(data_batch, (list, tuple)): # Handles Dataloader yielding (data, label) or similar
                    x = data_batch[0].to(device)
                else: # Handles Dataloader yielding data tensor directly
                    x = data_batch.to(device)
                
                optimizer.zero_grad()
                elbo, recon_batch_terms, kl_batch_terms = self.elbo(x) # elbo is scalar, others are (B,)
                
                loss = -elbo # Maximize ELBO = Minimize -ELBO
                loss.backward()
                optimizer.step()

                self._constrain_A_matrix() # Constrain A after optimizer step

                stats_epoch['elbo'] += elbo.item()
                stats_epoch['recon_loss'] -= recon_batch_terms.mean().item() # Accum NLL = -log p(x|z)
                stats_epoch['kl_loss'] += kl_batch_terms.mean().item()      # Accum KL divergence
                num_batches += 1

            if num_batches > 0:
                for s_key in stats_epoch:
                    stats_epoch[s_key] /= num_batches # Average over batches in the epoch
            
            for s_key in stats_history:
                stats_history[s_key].append(stats_epoch[s_key])

            pbar.set_postfix(stats_epoch)            

            if callback is not None:
                if (epoch + 1) % callback[0] == 0:
                    callback[1](self, epoch) # Pass model and epoch to callback
                    
        return stats_history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        steps: int,
        x0: Optional[torch.Tensor] = None,
        burn_in_steps: int = 0,
    ) -> torch.Tensor:
        """
        Roll the trained KVAE forward for `steps` time-steps and return the
        newly generated observations.

        Parameters
        ----------
        steps : int
            Number of observations to generate.
        x0 : torch.Tensor | None, default None
            • If **None** → start from a random latent  z₀ ~ N(0, I), with optional burn-in.
            • If tensor    → conditioning prefix (shape (T_cond, features) or (features,)).
                              The RNN encoder processes this sequence to get initial z₀.
                              Burn-in is ignored if x0 is provided.
        burn_in_steps : int, default 0
            Number of burn-in steps for the latent dynamics if x0 is None.

        Returns
        -------
        torch.Tensor
            Generated sequence of shape (steps, features). The conditioning
            frames `x0` (if any) are *not* included in the output.
        """
        device = next(self.parameters()).device
        z: torch.Tensor # Declare z type

        if x0 is None:
            z = torch.randn(self.latent_dim, device=device)  # prior draw
            if burn_in_steps > 0:
                current_A = self.A.data # Use constrained A if applicable
                current_logvar_Q = self.logvar_Q.data
                std_Q_burn_in = (0.5 * current_logvar_Q).exp()
                for _ in range(burn_in_steps):
                    z = current_A @ z + torch.randn_like(z) * std_Q_burn_in
        else:
            x0_tensor = x0.to(device)
            # Ensure x0_tensor is (B=1, T_cond, features_dim) for RNN encoder
            if x0_tensor.ndim == 1: # (features_dim) -> (1, 1, features_dim)
                x0_tensor = x0_tensor.unsqueeze(0).unsqueeze(0)
            elif x0_tensor.ndim == 2: # (T_cond, features_dim) -> (1, T_cond, features_dim)
                x0_tensor = x0_tensor.unsqueeze(0)
            
            # x0_tensor is now (1, T_cond, features_dim)
            encoder_params_sequence = self.encoder(x0_tensor) 
            # encoder_params_sequence shape: (1, T_cond, 2*latent_dim)
            
            # Use parameters from the *last* step of the conditioning sequence
            enc_out_params_last_step = encoder_params_sequence[:, -1, :] # (1, 2*latent_dim)
            mu, logvar = torch.chunk(enc_out_params_last_step, 2, dim=-1) # mu, logvar: (1, latent_dim)
            
            # Using the mean for initial z. Could also sample.
            z = mu.squeeze(0) # Shape: (latent_dim)
            # To sample z0 if desired:
            # logvar = logvar.clamp(-7, 7) # Clamp variance before sampling
            # eps = torch.randn_like(mu)
            # z = (mu + eps * (0.5 * logvar).exp()).squeeze(0)

        # Pre-compute stds for generation loop
        std_Q = (0.5 * self.logvar_Q.data).exp() # Using .data to ensure no grad tracking
        std_R = (0.5 * self.logvar_R.data).exp()
        current_A_gen = self.A.data

        generated_observations = []
        for _ in range(steps):
            # Latent dynamics
            z = current_A_gen @ z + torch.randn_like(z) * std_Q
            # Observation model
            x_gen = self.decoder(z) + torch.randn(self.features_dim, device=device) * std_R
            generated_observations.append(x_gen)

        return torch.stack(generated_observations)  # (steps, features)