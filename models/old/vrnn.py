import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any


# ---------------------------- utility: KL(q||p) ---------------------------

def kl_diag_gauss(mu_q: torch.Tensor, logvar_q: torch.Tensor, 
                  mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence KL(q || p) between two diagonal Gaussian distributions.
    Shapes: mu_q, logvar_q, mu_p, logvar_p are all (..., D) where D is latent dim.
    Output shape: (...,)
    """
    v_q = logvar_q.exp()
    v_p = logvar_p.exp()
    
    # Clamp variances for numerical stability (optional, but can be helpful)
    # v_q = torch.clamp(v_q, min=1e-8)
    # v_p = torch.clamp(v_p, min=1e-8)
    # logvar_q = torch.log(v_q)
    # logvar_p = torch.log(v_p)

    kl = 0.5 * (logvar_p - logvar_q + (v_q + (mu_q - mu_p).pow(2)) / v_p - 1)
    return kl.sum(dim=-1)


# ------------------- Variational Recurrent Neural Network (VRNN) -------------------

class VariationalRecurrentNeuralNetwork(nn.Module):
    def __init__(self,
                 features_dim: int,
                 latent_dim: int = 16,
                 main_rnn_hidden_dim: int = 128,
                 x_phi_dim: int = 64,
                 z_phi_dim: int = 32,
                 phi_x_module: Optional[nn.Module] = None,
                 phi_z_module: Optional[nn.Module] = None,
                 prior_net_module: Optional[nn.Module] = None,
                 encoder_net_module: Optional[nn.Module] = None,
                 decoder_net_module: Optional[nn.Module] = None,
                 main_rnn_cell_module: Optional[nn.Module] = None,
                 learn_obs_noise: bool = True):
        super().__init__()

        self.features_dim = features_dim
        self.latent_dim = latent_dim
        self.main_rnn_hidden_dim = main_rnn_hidden_dim
        self.x_phi_dim = x_phi_dim
        self.z_phi_dim = z_phi_dim
        self.learn_obs_noise = learn_obs_noise

        # --- Feature extractors ---
        # For observations x_t
        if phi_x_module is None:
            self.phi_x = nn.Sequential(
                nn.Linear(features_dim, x_phi_dim),
                nn.ReLU(),
                nn.Linear(x_phi_dim, x_phi_dim),
                nn.ReLU()
            )
        else:
            self.phi_x = phi_x_module

        # For latent variables z_t
        if phi_z_module is None:
            self.phi_z = nn.Sequential(
                nn.Linear(latent_dim, z_phi_dim),
                nn.ReLU(),
                nn.Linear(z_phi_dim, z_phi_dim), 
                nn.ReLU()
            )
        else:
            self.phi_z = phi_z_module

        # --- Prior network p(z_t | h_{t-1}) ---
        # Takes h_{t-1} (main_rnn_hidden_dim) -> outputs mu_z_prior, logvar_z_prior (2 * latent_dim)
        if prior_net_module is None:
            self.prior_net = nn.Sequential(
                nn.Linear(main_rnn_hidden_dim, main_rnn_hidden_dim // 2), # Added hidden layer
                nn.ReLU(),
                nn.Linear(main_rnn_hidden_dim // 2, 2 * latent_dim)
            )
        else:
            self.prior_net = prior_net_module

        # --- Encoder network q(z_t | x_t, h_{t-1}) ---
        # Takes cat(phi_x(x_t), h_{t-1}) (x_phi_dim + main_rnn_hidden_dim) -> mu_z_posterior, logvar_z_posterior (2 * latent_dim)
        if encoder_net_module is None:
            self.encoder_net = nn.Sequential(
                nn.Linear(x_phi_dim + main_rnn_hidden_dim, main_rnn_hidden_dim), # Added hidden layer
                nn.ReLU(),
                nn.Linear(main_rnn_hidden_dim, 2 * latent_dim)
            )
        else:
            self.encoder_net = encoder_net_module

        # --- Decoder network p(x_t | z_t, h_{t-1}) ---
        # Takes cat(phi_z(z_t), h_{t-1}) (z_phi_dim + main_rnn_hidden_dim) -> mu_x_reconstruction (features_dim)
        # If learn_obs_noise is True, outputs 2 * features_dim for mu_x and logvar_x
        decoder_out_dim = 2 * features_dim if learn_obs_noise else features_dim
        if decoder_net_module is None:
            self.decoder_net = nn.Sequential(
                nn.Linear(z_phi_dim + main_rnn_hidden_dim, main_rnn_hidden_dim), # Added hidden layer
                nn.ReLU(),
                nn.Linear(main_rnn_hidden_dim, decoder_out_dim)
            )
        else:
            self.decoder_net = decoder_net_module

        # --- Main RNN Cell ---
        # Takes cat(phi_x(x_t), phi_z(z_t)) (x_phi_dim + z_phi_dim) and (h_{t-1}, c_{t-1})
        # Outputs (h_t, c_t)
        if main_rnn_cell_module is None:
            self.main_rnn_cell = nn.LSTMCell(x_phi_dim + z_phi_dim, main_rnn_hidden_dim)
        else:
            self.main_rnn_cell = main_rnn_cell_module

        # Observation noise (log variance)
        if not learn_obs_noise:
            self.logvar_R = nn.Parameter(torch.zeros(features_dim), requires_grad=True) # Still learnable if not part of decoder
        else:
            self.logvar_R = None # Decoder will output it


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process the input sequence x.
        Args:
            x (torch.Tensor): Input sequence of shape (B, T, features_dim).
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing stacked tensors for:
                - "prior_mu": (T, B, latent_dim)
                - "prior_logvar": (T, B, latent_dim)
                - "posterior_mu": (T, B, latent_dim)
                - "posterior_logvar": (T, B, latent_dim)
                - "recon_mu_x": (T, B, features_dim)
                - "recon_logvar_x": (T, B, features_dim) if learn_obs_noise else None
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state for the main RNN
        h_t = torch.zeros(batch_size, self.main_rnn_hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.main_rnn_hidden_dim, device=device)

        all_prior_mu = []
        all_prior_logvar = []
        all_posterior_mu = []
        all_posterior_logvar = []
        all_recon_mu_x = []
        all_recon_logvar_x = [] if self.learn_obs_noise else None

        for t in range(seq_len):
            x_t_slice = x[:, t, :]  # (B, features_dim)

            # --- Prior for z_t based on h_{t-1} (which is current h_t before update) ---
            prior_params_t = self.prior_net(h_t)
            prior_mu_t, prior_logvar_t = torch.chunk(prior_params_t, 2, dim=-1)
            prior_logvar_t = F.hardtanh(prior_logvar_t, min_val=-7., max_val=7.) # Clamp logvar

            # --- Encoder for z_t based on x_t and h_{t-1} ---
            phi_x_t = self.phi_x(x_t_slice)  # (B, x_phi_dim)
            encoder_input_t = torch.cat([phi_x_t, h_t], dim=-1)
            posterior_params_t = self.encoder_net(encoder_input_t)
            posterior_mu_t, posterior_logvar_t = torch.chunk(posterior_params_t, 2, dim=-1)
            posterior_logvar_t = F.hardtanh(posterior_logvar_t, min_val=-7., max_val=7.) # Clamp logvar

            # --- Sample z_t from posterior using reparameterization trick ---
            eps_z = torch.randn_like(posterior_mu_t)
            z_t = posterior_mu_t + eps_z * (0.5 * posterior_logvar_t).exp()
            phi_z_t = self.phi_z(z_t)  # (B, z_phi_dim)

            # --- Decoder for x_t based on z_t and h_{t-1} ---
            decoder_input_t = torch.cat([phi_z_t, h_t], dim=-1)
            decoded_x_params_t = self.decoder_net(decoder_input_t)
            if self.learn_obs_noise:
                recon_mu_x_t, recon_logvar_x_t = torch.chunk(decoded_x_params_t, 2, dim=-1)
                recon_logvar_x_t = F.hardtanh(recon_logvar_x_t, min_val=-7., max_val=7.) # Clamp logvar
                all_recon_logvar_x.append(recon_logvar_x_t)
            else:
                recon_mu_x_t = decoded_x_params_t
            
            # --- Update main RNN state: h_t = f(phi_x(x_t), phi_z(z_t), h_{t-1}) ---
            rnn_input_t = torch.cat([phi_x_t, phi_z_t], dim=-1)
            h_t, c_t = self.main_rnn_cell(rnn_input_t, (h_t, c_t))

            # Collect parameters
            all_prior_mu.append(prior_mu_t)
            all_prior_logvar.append(prior_logvar_t)
            all_posterior_mu.append(posterior_mu_t)
            all_posterior_logvar.append(posterior_logvar_t)
            all_recon_mu_x.append(recon_mu_x_t)

        results = {
            "prior_mu": torch.stack(all_prior_mu), # (T, B, latent_dim)
            "prior_logvar": torch.stack(all_prior_logvar),
            "posterior_mu": torch.stack(all_posterior_mu),
            "posterior_logvar": torch.stack(all_posterior_logvar),
            "recon_mu_x": torch.stack(all_recon_mu_x), # (T, B, features_dim)
            "recon_logvar_x": torch.stack(all_recon_logvar_x) if self.learn_obs_noise else None
        }
        return results

    def elbo(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the Evidence Lower Bound (ELBO) for the given batch.
        Args:
            x (torch.Tensor): Input sequence of shape (B, T, features_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - elbo: Scalar, the mean ELBO over the batch.
                - recon_loss: Scalar, the mean reconstruction loss (negative log-likelihood).
                - kl_div: Scalar, the mean KL divergence.
        """
        params = self.forward(x) # Tensors are (T, B, Dim)
        
        # Transpose x to match params: (T, B, features_dim)
        x_transposed = x.transpose(0, 1)

        # --- Reconstruction Loss (Negative Log-Likelihood) ---
        recon_mu_x = params["recon_mu_x"]
        if self.learn_obs_noise:
            recon_logvar_x = params["recon_logvar_x"]
        else:
            recon_logvar_x = self.logvar_R.unsqueeze(0).unsqueeze(0).expand_as(recon_mu_x)
        
        # Sum over features_dim, then sum over time_dim
        # recon_nll_per_step_and_batch = - (
        #     -0.5 * math.log(2 * math.pi) 
        #     - 0.5 * recon_logvar_x
        #     - 0.5 * ((x_transposed - recon_mu_x)**2 / recon_logvar_x.exp())
        # ).sum(dim=-1) # Sum over features
        
        # Simpler calculation for Gaussian NLL
        # log p(x|z) = -0.5 * [ (x-mu)^2 / sigma^2 + log(sigma^2) + log(2pi) ]
        # Sum over features dimension
        recon_nll_terms = 0.5 * (
            ((x_transposed - recon_mu_x)**2 / recon_logvar_x.exp()) + 
            recon_logvar_x + 
            math.log(2 * math.pi)
        )
        recon_nll_per_step_and_batch = recon_nll_terms.sum(dim=-1) # (T, B)

        # Sum over time steps for each batch item
        recon_nll_per_batch_item = recon_nll_per_step_and_batch.sum(dim=0)  # (B,)
        mean_recon_loss = recon_nll_per_batch_item.mean() # Scalar

        # --- KL Divergence KL(q(z|x,h) || p(z|h)) ---
        kl_div_per_step_and_batch = kl_diag_gauss(
            params["posterior_mu"], params["posterior_logvar"],
            params["prior_mu"], params["prior_logvar"]
        ) # (T, B)

        # Sum over time steps for each batch item
        kl_div_per_batch_item = kl_div_per_step_and_batch.sum(dim=0)  # (B,)
        mean_kl_div = kl_div_per_batch_item.mean() # Scalar
        
        elbo = -mean_recon_loss - mean_kl_div
        return elbo, mean_recon_loss, mean_kl_div

    @torch.no_grad()
    def generate(self,
                 steps: int,
                 x0_context: Optional[torch.Tensor] = None,
                 burn_in_steps: int = 0) -> torch.Tensor:
        """
        Generate new sequences.
        Args:
            steps (int): Number of time steps to generate.
            x0_context (Optional[torch.Tensor]): Conditioning context (B, T_ctx, features_dim).
                                                 If None, starts from scratch. Batch size B=1 for generation.
            burn_in_steps (int): Number of burn-in steps if x0_context is None.
        Returns:
            torch.Tensor: Generated sequence of shape (B, steps, features_dim).
        """
        self.eval() # Set to evaluation mode
        device = next(self.parameters()).device
        
        batch_size = 1 # Generation is typically done one sequence at a time
        if x0_context is not None:
            if x0_context.ndim == 2: # (T_ctx, features_dim) -> (1, T_ctx, features_dim)
                x0_context = x0_context.unsqueeze(0)
            batch_size = x0_context.shape[0]
            x0_context = x0_context.to(device)

        # Initialize hidden state
        h_t = torch.zeros(batch_size, self.main_rnn_hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.main_rnn_hidden_dim, device=device)
        
        # Process context if provided
        if x0_context is not None:
            context_len = x0_context.shape[1]
            for t in range(context_len):
                x_ctx_t_slice = x0_context[:, t, :]
                phi_x_ctx_t = self.phi_x(x_ctx_t_slice)
                
                # During context processing, we'd ideally use the posterior to get z_t
                # but for simplicity in generation setup, we can just run the RNN
                # or use the prior for z_t if we don't want to sample from posterior here.
                # Let's use the posterior mean for z_t from context.
                encoder_input_t = torch.cat([phi_x_ctx_t, h_t], dim=-1)
                posterior_params_t = self.encoder_net(encoder_input_t)
                posterior_mu_t, _ = torch.chunk(posterior_params_t, 2, dim=-1)
                z_t_ctx = posterior_mu_t # Use mean for context processing
                phi_z_ctx_t = self.phi_z(z_t_ctx)
                
                rnn_input_t = torch.cat([phi_x_ctx_t, phi_z_ctx_t], dim=-1)
                h_t, c_t = self.main_rnn_cell(rnn_input_t, (h_t, c_t))
        
        # Burn-in phase (if no context)
        if x0_context is None and burn_in_steps > 0:
            # Initial z_t for burn-in (e.g., from a fixed prior or zeros)
            # Sample z_0 from prior p(z_0 | h_{-1}=0)
            prior_params_t = self.prior_net(h_t) # h_t is h_{-1} (zeros)
            prior_mu_t, prior_logvar_t = torch.chunk(prior_params_t, 2, dim=-1)
            eps_z = torch.randn_like(prior_mu_t)
            z_t_burn = prior_mu_t + eps_z * (0.5 * prior_logvar_t).exp()

            for _ in range(burn_in_steps):
                phi_z_t_burn = self.phi_z(z_t_burn)
                
                # Decode x_t (not stored, just for RNN update)
                decoder_input_t = torch.cat([phi_z_t_burn, h_t], dim=-1)
                decoded_x_params_t = self.decoder_net(decoder_input_t)
                if self.learn_obs_noise:
                    recon_mu_x_t_burn, _ = torch.chunk(decoded_x_params_t, 2, dim=-1)
                else:
                    recon_mu_x_t_burn = decoded_x_params_t
                
                # Use mean of decoded x for feature extraction
                phi_x_t_burn = self.phi_x(recon_mu_x_t_burn)
                
                # Update RNN state
                rnn_input_t = torch.cat([phi_x_t_burn, phi_z_t_burn], dim=-1)
                h_t, c_t = self.main_rnn_cell(rnn_input_t, (h_t, c_t))
                
                # Sample next z_t for burn-in from prior
                prior_params_next_t = self.prior_net(h_t)
                prior_mu_next_t, prior_logvar_next_t = torch.chunk(prior_params_next_t, 2, dim=-1)
                eps_z_next = torch.randn_like(prior_mu_next_t)
                z_t_burn = prior_mu_next_t + eps_z_next * (0.5 * prior_logvar_next_t).exp()


        # Generation loop
        generated_x_sequence = []
        
        # Initial z_t for generation loop (if no context and no burn-in, or after burn-in)
        # Sample z_0 from prior p(z_0 | h_{final_context_or_burn_in})
        prior_params_t = self.prior_net(h_t)
        prior_mu_t, prior_logvar_t = torch.chunk(prior_params_t, 2, dim=-1)
        eps_z = torch.randn_like(prior_mu_t)
        current_z_t = prior_mu_t + eps_z * (0.5 * prior_logvar_t).exp()

        for _ in range(steps):
            phi_z_gen_t = self.phi_z(current_z_t)
            
            # Decode x_t
            decoder_input_gen_t = torch.cat([phi_z_gen_t, h_t], dim=-1)
            decoded_x_params_gen_t = self.decoder_net(decoder_input_gen_t)
            
            if self.learn_obs_noise:
                recon_mu_x_gen_t, recon_logvar_x_gen_t = torch.chunk(decoded_x_params_gen_t, 2, dim=-1)
                # Sample from p(x_t | z_t, h_{t-1})
                eps_x = torch.randn_like(recon_mu_x_gen_t)
                x_gen_t = recon_mu_x_gen_t + eps_x * (0.5 * recon_logvar_x_gen_t).exp()
            else:
                recon_mu_x_gen_t = decoded_x_params_gen_t
                # Sample from p(x_t | z_t, h_{t-1})
                eps_x = torch.randn_like(recon_mu_x_gen_t)
                x_gen_t = recon_mu_x_gen_t + eps_x * (0.5 * self.logvar_R.expand_as(recon_mu_x_gen_t)).exp()

            generated_x_sequence.append(x_gen_t)
            
            # Use the *generated* x_gen_t (or its mean if preferred) for RNN input
            phi_x_gen_t = self.phi_x(x_gen_t) # or self.phi_x(recon_mu_x_gen_t)
            
            # Update RNN state
            rnn_input_gen_t = torch.cat([phi_x_gen_t, phi_z_gen_t], dim=-1)
            h_t, c_t = self.main_rnn_cell(rnn_input_gen_t, (h_t, c_t))
            
            # Sample next z_t from prior for the next iteration
            prior_params_next_t = self.prior_net(h_t)
            prior_mu_next_t, prior_logvar_next_t = torch.chunk(prior_params_next_t, 2, dim=-1)
            eps_z_next = torch.randn_like(prior_mu_next_t)
            current_z_t = prior_mu_next_t + eps_z_next * (0.5 * prior_logvar_next_t).exp()
            
        return torch.stack(generated_x_sequence, dim=1) # (B, steps, features_dim)

    def fit(self, dataloader: Any, optimizer: torch.optim.Optimizer, 
            epochs: int = 100, show_progress: bool = True, 
            callback: Optional[Tuple[int, callable]] = None,
            grad_clip_norm: Optional[float] = None) -> Dict[str, List[float]]:
        """
        Train the VRNN model.
        """
        stats_history = {'elbo': [], 'recon_loss': [], 'kl_loss': []}
        device = next(self.parameters()).device

        pbar = tqdm(range(epochs), disable=not show_progress, desc="Epochs")

        for epoch in pbar:
            self.train() # Set to training mode
            epoch_elbo = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            num_batches = 0

            for data_batch in dataloader:

                if isinstance(data_batch, (list, tuple)):
                    x = data_batch[0].to(device)
                else:
                    x = data_batch.to(device)

                if x.ndim == 2: x = x.unsqueeze(0)
                
                optimizer.zero_grad()
                elbo, recon_loss, kl_div = self.elbo(x)
                
                loss = -elbo  # Maximize ELBO = Minimize -ELBO
                loss.backward()

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
                
                optimizer.step()

                epoch_elbo += elbo.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_div.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_elbo = epoch_elbo / num_batches
                avg_recon_loss = epoch_recon_loss / num_batches
                avg_kl_loss = epoch_kl_loss / num_batches
            else: # Should not happen if dataloader is not empty
                avg_elbo, avg_recon_loss, avg_kl_loss = 0.0, 0.0, 0.0

            stats_history['elbo'].append(avg_elbo)
            stats_history['recon_loss'].append(avg_recon_loss)
            stats_history['kl_loss'].append(avg_kl_loss)

            pbar.set_postfix({
                'ELBO': f"{avg_elbo:.4f}", 
                'ReconL': f"{avg_recon_loss:.4f}", 
                'KL': f"{avg_kl_loss:.4f}"
            })

            if callback is not None:
                if (epoch + 1) % callback[0] == 0:
                    callback[1](self, epoch, stats_history) # Pass model, epoch, and stats
                    
        return stats_history