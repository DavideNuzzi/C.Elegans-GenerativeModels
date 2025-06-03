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
    def generate(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Generates sequences autoregressively starting from x_initial.
        Args:
            x: Initial state, shape (T, features_dim) or (features_dim).
            num_steps: Number of steps to generate.
        Returns:
            Generated states (x_hat_2, ..., x_hat_num_steps+1)
        """
        self.eval() # Set model to evaluation mode
        
        # If the user passes a sequence and not a single point take the last one
        if x.ndim == 2:
            x = x[-1, :]

        states_predictions = []
        
        for _ in range(num_steps):

            # Predict residual
            f_pred = self.f_net(x) # f(x_hat_t)

            # Next step prediction
            x = x + f_pred + self.sigma * torch.randn_like(f_pred) # x_hat_{t+1}

            states_predictions.append(x)

        states_predictions = torch.stack(states_predictions, dim=0)

        return states_predictions




    # def fit(self,
    #         dataloader: Any, # Should be torch.utils.data.DataLoader
    #         optimizer: torch.optim.Optimizer,
    #         epochs: int,
    #         show_progress: bool = True,
    #         grad_clip_norm: Optional[float] = None) -> Dict[str, List[float]]:
    #     """
    #     Trains the model.
    #     """
    #     stats_history: Dict[str, List[float]] = {'loss': []}
    #     device = next(self.parameters()).device # Assumes model is already on a device

    #     # Progress bar for epochs
    #     epoch_pbar = tqdm(range(epochs), desc="Epochs", disable=not show_progress, leave=True)

    #     for epoch in epoch_pbar:
    #         self.train() # Set model to training mode
    #         epoch_loss_sum = 0.0
    #         num_batches = 0

    #         # Iterate over batches without a separate progress bar for batches
    #         for x_batch in dataloader:
    #             # Ensure data is on the correct device
    #             if isinstance(x_batch, (list, tuple)): # e.g. [data, labels]
    #                 x_batch_device = x_batch[0].to(device)
    #             else:
    #                 x_batch_device = x_batch.to(device)
                
    #             if x_batch_device.shape[1] <= 1: # Need at least 2 time steps for a residual
    #                 continue

    #             optimizer.zero_grad()
    #             current_loss = self.loss(x_batch_device)

    #             if torch.isnan(current_loss) or torch.isinf(current_loss):
    #                 print(f"Warning: NaN/Inf loss at epoch {epoch+1}, batch {num_batches+1}. Skipping update.")
    #                 if num_batches == 0 and epoch == 0 and len(dataloader) == 1 : # Critical failure early
    #                     print("Stopping due to NaN/Inf loss on first batch of first epoch.")
    #                     stats_history['loss'].append(float('nan'))
    #                     # Close progress bar before early exit
    #                     if show_progress:
    #                         epoch_pbar.close()
    #                     return stats_history 
    #                 continue # Skip this batch update

    #             current_loss.backward() # Gradients propagate through the unrolled sequence

    #             if grad_clip_norm is not None:
    #                 torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

    #             optimizer.step()

    #             epoch_loss_sum += current_loss.item()
    #             num_batches += 1
            
    #         if num_batches > 0:
    #             avg_epoch_loss = epoch_loss_sum / num_batches
    #         else: # Handles case where all batches were skipped (e.g. due to sequence length)
    #             avg_epoch_loss = float('nan') if not stats_history['loss'] else stats_history['loss'][-1]
            
    #         stats_history['loss'].append(avg_epoch_loss)

    #         if show_progress:
    #             # Update the epoch progress bar with the average loss for the completed epoch
    #             epoch_pbar.set_postfix({'Avg Loss': f"{avg_epoch_loss:.4e}"})
    #         # Fallback print if progress bar disabled but epoch updates are desired
    #         elif not show_progress and ((epoch + 1) % 10 == 0 or (epoch + 1) == epochs) : 
    #              print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4e}")
        
    #     if show_progress: # Ensure the progress bar is closed upon completion
    #         epoch_pbar.close()
            
    #     return stats_history