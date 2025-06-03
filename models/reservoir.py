import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
from utils.utils import ridge_regression

# ---------------------------------------------------------------------------- #
#                               Classic reservoir                              #
# ---------------------------------------------------------------------------- #
class ReservoirModel(nn.Module):
    
    """
    An Echo-State / reservoir computer that learns a *linear read-out*
    from fixed, randomly-connected nonlinear states.

    Parameters
    ----------
    features_dim            - dimensionality of the input / output time-series x(t)
    N_nodes                 - number of reservoir neurons
    connectivity_density    - probability of a non-zero recurrent edge
    spectral_radius         - desired spectral radius ρ(A) after rescaling
    alpha_regularizer       - ridge (L2) regularisation strength
    input_weight_strength   - scale of input weights W_in
    predict_mode            - "next"     : read-out targets x(t + 1)
                              "residual" : targets x(t + 1) - x(t)
    """

    def __init__(self,
                 features_dim: int,
                 N_nodes: int = 100,
                 connectivity_density: float = 0.2,
                 spectral_radius: float = 0.1,
                 alpha_regularizer: float = 1e-6,
                 input_weight_strength: float = 0.1,
                 predict_mode: str = "next"):
        
        super().__init__()
        assert predict_mode in {"next", "residual"}

        self.features_dim = features_dim
        self.N_nodes = N_nodes
        self.alpha = alpha_regularizer
        self.predict_mode = predict_mode

        # Reservoir matrix
        A = torch.rand(N_nodes, N_nodes)
        mask = (torch.rand_like(A) < connectivity_density).float()
        A *= mask

        # Rescale to requested spectral radius ρ
        eigvals = torch.linalg.eigvals(A)          
        rho = eigvals.abs().max().real + 1e-12
        A *= spectral_radius / rho

        # Input matrix
        W_in = (torch.rand(N_nodes, features_dim) - 0.5) * 2.0 * input_weight_strength

        # Register for automatic device/ dtype handling but freeze grads
        self.register_buffer("A", A)
        self.register_buffer("W_in", W_in)
        self.register_buffer("W_out", torch.empty(N_nodes, features_dim))   # filled at fit()


    # --------------------------------------------------------------------- #
    #  Private utilities                                                    #
    # --------------------------------------------------------------------- #
    def _reservoir_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rolls the reservoir across the input sequence (T, F) ➜ states (T-1, N)
        """
        T = x.size(0)
        states = torch.zeros(T - 1, self.N_nodes, device=x.device, dtype=x.dtype)
        r_t = torch.zeros(self.N_nodes, device=x.device, dtype=x.dtype)

        for t in range(T - 1):                        # 0 … T-2
            r_t = torch.tanh(self.A @ r_t + self.W_in @ x[t])
            states[t] = r_t                           # <-- overwrite row t

        return states
    
    
    # --------------------------------------------------------------------- #
    #  Core API                                                             #
    # --------------------------------------------------------------------- #
    def fit(self, x: torch.Tensor) -> Dict[str, List[float]]:
        """
        Learns the ridge read-out on the *entire* 1-D sequence x (T, F).
        No batching is needed for echo-state networks.
        """
        if x.ndim != 2:
            raise ValueError("Input must be of shape (T, features_dim)")

        with torch.no_grad():

            # ------------- targets -------------
            if self.predict_mode == "next":
                Y = x[1:]                              # (T-1, F)
            else:   # residual mode
                Y = x[1:] - x[:-1]

            # ------------- reservoir states -------------
            R = self._reservoir_states(x)              # (T-1, N)

            # ------------- ridge regression -------------
            self.W_out.copy_(ridge_regression(R, Y, self.alpha))
            
            loss = self.loss(x).item()

        return loss

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One-shot forward pass over a *known* trajectory
        Returns predictions for t = 0 … T-2  (length T-1).
        """
        if not hasattr(self, "W_out"):
            raise AttributeError("Model not yet fitted. Call `fit` first.")

        x = x.to(self.A.dtype).to(self.A.device)
        R = self._reservoir_states(x)
        preds = R @ self.W_out                         # (T-1, F)
        return preds

    # --------------------------------------------------------------------- #
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSE on the supplied ground-truth sequence.
        """
        preds = self.forward(x)
        if self.predict_mode == "next":
            target = x[1:]
        else:
            target = x[1:] - x[:-1]

        return ((preds - target)**2).sum(dim=-1).sqrt().mean()

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(self,
                 x_init: torch.Tensor,
                 num_steps: int) -> torch.Tensor:
        """
        Autonomously rolls the reservoir for `num_steps` steps.

        *If `x_init` is a sequence*, the final sample seeds the internal state
        (recommended).  If it is a single vector, the reservoir is started cold
        with r(0) = 0.
        """
        if x_init.ndim == 1:
            x_history = x_init.unsqueeze(0)            # (1, F)
        else:
            x_history = x_init                         # (T, F)

        x_history = x_history.to(self.A.dtype).to(self.A.device)
        r = torch.zeros(self.N_nodes, device=x_history.device, dtype=x_history.dtype)

        # Warm-up if we have multiple initial steps
        for t in range(x_history.size(0) - 1):
            r = torch.tanh(self.A @ r + self.W_in @ x_history[t])

        x_prev = x_history[-1]
        preds = []

        for _ in range(num_steps):
            r = torch.tanh(self.A @ r + self.W_in @ x_prev)
            out = r @ self.W_out                       # predicted next / residual

            if self.predict_mode == "residual":
                x_new = x_prev + out
            else:
                x_new = out

            preds.append(x_new)
            x_prev = x_new

        return torch.stack(preds, dim=0)               # (num_steps, F)

