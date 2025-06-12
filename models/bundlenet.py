import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
from torch.optim import Optimizer


class BundleNET(nn.Module):
    """
    The model described in:
    https://www.biorxiv.org/content/10.1101/2023.08.08.551978v4.full
    It is basically an autoregressive model (predicting the residuals)
    The acts at the latent level and not in the original timeseries space.
    It uses additional (behavioural) modules to shape the latent space itself.
    I will implement so that it works even without the behavior.
    I don't think it will work well as a generative model, as it is not
    trained to do it, it is just an embedding model that uses the temporal
    prediction as a mean of shaping the latent space in a better way.
    """
    def __init__(self,
                 encoder: Optional[nn.Module] = None,
                 temporal_predictor: Optional[nn.Module] = None,
                 behavior_decoder: Optional[nn.Module] = None,
                 noise_std: int = 0,
                 gamma: float = 0.9
                 ):
        
        super().__init__()

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = nn.Sequential(
                nn.LazyLinear(50), nn.ReLU(),
                nn.LazyLinear(30), nn.ReLU(),
                nn.LazyLinear(25), nn.ReLU(),
                nn.LazyLinear(10), nn.ReLU(),
                nn.LazyLinear(3),
            )

        if temporal_predictor:
            self.temporal_predictor = temporal_predictor
        else:
            self.temporal_predictor = nn.Sequential(
                nn.LazyLinear(3),
            )

        self.behavior_decoder = behavior_decoder
        self.noise_std = noise_std
        self.gamma = gamma

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input x into the latent space. Then predicts the next step.
        Args:
            x: Neural state (B, T, features_dim).
        Returns:
            z: Encoded state (B, T, latent_dim)
        """

        z = self.encoder(x)
        z = z + torch.randn_like(z) * self.noise_std
        return z

    
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predicts the next latent state
        Args:
            z: Latent state (B, T, latent_dim)
        Returns:
            z_next: Predicted next state (B, T, latent_dim)
        """
        z_next = self.temporal_predictor(z) + z
        return z_next

    # ------------------------------------------------------------------ #
    def loss(self, x: torch.Tensor, behavior: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Encode the whole sequence
        z = self.encode(x)

        # Predict the next latent state
        z_next = self.predict(z)

        # Compute the "Markov loss"
        loss_markov = ((z_next[:, :-1, :] - z[:, 1:, :])**2).sum(dim=-1).mean()
        
        # Compute the behavior loss (if needed)
        if behavior is not None:
            behavior_pred = self.behavior_decoder(z)
            
            behavior = behavior[:, 1:]
            behavior_pred = torch.transpose(behavior_pred[:, 1:, :], 1, 2)
            
            loss_behavior = nn.functional.cross_entropy(behavior_pred, behavior)
            loss = (1-self.gamma) * loss_markov + self.gamma * loss_behavior
        else:
            loss = loss_markov
            loss_behavior = 0

        return loss, loss_markov, loss_behavior 


    # ------------------------------------------------------------------ #
    def fit(
        self,
        x, 
        optimizer: Optimizer,
        behavior: Optional[torch.Tensor] = None,
        epochs: int = 100,
        grad_clip: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[float]:
        
        loss_history = {'Markov': [], 'Behavior': [], 'Total': []}
        pbar = tqdm(range(epochs), disable=not show_progress)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        if behavior is not None: 
            if behavior.ndim == 1:
                behavior = behavior.unsqueeze(0)
                
        for _ in pbar:
            
            optimizer.zero_grad()
            loss, loss_markov, loss_behavior = self.loss(x, behavior)
            loss.backward()
            
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            optimizer.step()

            pbar.set_postfix({'Loss markov': loss_markov.item(), 'Loss behavior': loss_behavior.item(), 'Loss total': loss.item()})
            loss_history['Markov'].append(loss_markov.item())
            loss_history['Behavior'].append(loss_behavior.item())
            loss_history['Total'].append(loss.item())

        return loss_history

    
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, x_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Generates sequences autoregressively starting from x_initial.
        It first encodes to a latent state and then generates both
        future latent states and behaviors
        Args:
            x_init: Initial state, shape (T, features_dim) or (features_dim).
            num_steps: Number of steps to generate.
        Returns:
            Generated latent states (z_2, ..., z_num_steps+1)
            Generated behaviors.
        """

        self.eval()
        
        # If the user passes a sequence and not a single point take the last one
        if x_init.ndim == 2:
            x = x_init[-1, :]
        else:
            x = x_init

        # Encode in the latent space
        z = self.encode(x)
        
        states_predictions = []
        behavior_predictions = []
        
        for _ in range(num_steps):
            
            # Predict new state and predict corresponding behavior
            z = self.predict(z)
            B = self.behavior_decoder(z)

            states_predictions.append(z)
            behavior_predictions.append(B)

        states_predictions = torch.stack(states_predictions, dim=0)
        behavior_predictions = torch.stack(behavior_predictions, dim=0)
        behavior_predictions = torch.argmax(behavior_predictions, dim=-1)

        return states_predictions, behavior_predictions

