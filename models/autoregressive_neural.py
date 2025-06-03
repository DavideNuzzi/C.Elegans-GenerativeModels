import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
from torch.optim import Optimizer


class NeuralAutoregressiveResidual(nn.Module):
    """
    A neural autoregressive model that predicts residuals.
    Model: x_hat_{t+1} = x_hat_t + f(x_hat_t) + sigma * epsilon_t
    The network f(x_hat_t) is trained to predict the "de-noised" true residual:
    (x_{t+1} - x_t) - sigma * epsilon_t, where epsilon_t is the noise
    instance used in the model's prediction step for x_hat_{t+1}.
    """
    def __init__(self,
                 features_dim: int,
                 f_net: Optional[nn.Module] = None,
                 sigma: float = 0.01):
        super().__init__()
        self.features_dim = features_dim

        if f_net is None:
            # Default MLP for the residual function f
            self.f_net = nn.Sequential(
                nn.Linear(features_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, features_dim)
            )
        else:
            self.f_net = f_net

        self.sigma = sigma # Fixed noise standard deviation, not learned

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs autoregressive rollout prediction.
        Args:
            x: States (B, L, features_dim).
        Returns:
            predicted_states: (x_hat_1, ..., x_hat_L) of shape (B, L, features_dim).
        """

        batch_size, L, _ = x.shape
        device = x.device

        # The prediction starts from x_hat_1 = x_1
        x_current = x[:, 0, :] 

        residual_predictions = [] # Will store f(x_hat_1), ..., f(x_hat_{L})
        noises = []               # Will store the noise used at each pass

        for _ in range(L-1):
            
            # Compute residual prediction
            f_pred = self.f_net(x_current) # f(x_hat_t)

            # Save the prediction (it is used in the loss)
            residual_predictions.append(f_pred)

            # Sample noise
            noise_t = torch.randn(batch_size, self.features_dim, device=device) # epsilon_t
            
            # Save noise
            noises.append(noise_t)

            # Next step prediction
            x_current = x_current + f_pred + self.sigma * noise_t # x_hat_{t+1}

        residual_predictions = torch.stack(residual_predictions, dim=1)
        noises = torch.stack(noises, dim=1)

        return residual_predictions,noises

    # ------------------------------------------------------------------ #
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for a batch of true sequences.
        The loss is the mean squared error of:
        f(x_hat_t) - ( (x_{t+1} - x_t) + sigma * epsilon_t )
        Args:
            x: Ground truth sequences (B, L, features_dim).
        Returns:
            loss: Scalar loss value (averaged over batch and time).
        """

        # Residual predictions f(x_hat_1), ..., f(x_hat_L)
        prediction_residuals, noises= self.forward(x)

        # true_residuals are (x_1-x_0), ..., (x_L-x_{L-1})
        true_residuals = x[:, 1:, :] - x[:, :-1, :] # (B, L-1, features_dim)

        # Target for f_predictions: (true_residuals + self.sigma * noise)
        loss_vector_components = prediction_residuals - true_residuals + self.sigma * noises

        # Sum of squared L2 norms of the difference vectors, then averaged.
        # ||v||^2 = sum(v_i^2) over feature dimension
        squared_l2_norms_per_step = (loss_vector_components**2).sum(dim=-1) # (B, L)

        # Average these squared L2 norms over batch and L time steps
        loss = squared_l2_norms_per_step.mean()

        return loss

    # ------------------------------------------------------------------ #
    def fit(
        self,
        dataloader: Any,               # torch.utils.data.DataLoader
        optimizer: Optimizer,
        epochs: int = 100,
        grad_clip: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[float]:
        
        """One-pass training loop; returns per-epoch losses."""
        device = next(self.parameters()).device
        loss_history: List[float] = []

        pbar = tqdm(range(epochs), disable=not show_progress)

        for _ in pbar:
            epoch_loss = 0.0

            for x in dataloader:
                x = x[0] if isinstance(x, (list, tuple)) else x

                optimizer.zero_grad()
                loss = self.loss(x)
                loss.backward()
                
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

                optimizer.step()
                epoch_loss += loss.item()

            pbar.set_postfix({'Loss': f"{epoch_loss:.4e}"})
            loss_history.append(epoch_loss / len(dataloader))

        return loss_history
    
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, x_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Generates sequences autoregressively starting from x_initial.
        Args:
            x_init: Initial state, shape (T, features_dim) or (features_dim).
            num_steps: Number of steps to generate.
        Returns:
            Generated states (x_hat_2, ..., x_hat_num_steps+1)
        """
        self.eval() # Set model to evaluation mode
        
        # If the user passes a sequence and not a single point take the last one
        if x_init.ndim == 2:
            x = x_init[-1, :]
        else:
            x = x_init

        states_predictions = []
        
        for _ in range(num_steps):

            # Predict residual
            f_pred = self.f_net(x) # f(x_hat_t)

            # Next step prediction
            x = x + f_pred + self.sigma * torch.randn_like(f_pred) # x_hat_{t+1}

            states_predictions.append(x)

        states_predictions = torch.stack(states_predictions, dim=0)

        return states_predictions

