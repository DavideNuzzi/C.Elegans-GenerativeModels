import torch
import torch.nn as nn
from typing import List, Dict
from utils.utils import ridge_regression


# --------------------------------------------------------------------------- #
#                         Polynomial delay-embedding model                    #
# --------------------------------------------------------------------------- #
class PolynomialEmbeddingModel(nn.Module):
    """
    Delay embedding with quadratic terms followed by ridge regression.

    Parameters
    ----------
    features_dim      – input dimension F
    k_steps           – number of past points in the window
    skip              – lag between successive points inside the window
    alpha_regularizer – ridge (L2) regularisation strength
    predict_mode      – "next":  predict x(t+1)
                         "residual": predict x(t+1) − x(t)
    """

    def __init__(self,
                 features_dim: int,
                 k_steps: int = 2,
                 skip: int = 1,
                 alpha_regularizer: float = 1e-3,
                 predict_mode: str = "next"):

        super().__init__()
        assert predict_mode in {"next", "residual"}

        self.F = features_dim
        self.k = k_steps
        self.skip = skip
        self.alpha_regularizer = alpha_regularizer
        self.predict_mode = predict_mode

        L = self.F * self.k                 # linear part size
        Q = L * (L + 1) // 2                # quadratic upper-triangular part
        self.embed_dim = 1 + L + Q

        self.register_buffer("W_out", torch.empty(self.embed_dim, self.F))

        tri = torch.triu_indices(L, L)
        self.register_buffer("_tri_i", tri[0])
        self.register_buffer("_tri_j", tri[1])

    # --------------------------------------------------------------------- #
    #  Private utilities                                                    #
    # --------------------------------------------------------------------- #
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Φ(x_t,…,x_{t−(k−1)skip}) → [1 | linear | quadratic]."""
        T = x.size(0) - self.k * self.skip
        win = torch.stack([x[i * self.skip:i * self.skip + T]          # (T, k, F)
                           for i in range(self.k)], dim=1)
        lin = win.reshape(T, -1)                                       # (T, L)
        outer = lin.unsqueeze(2) * lin.unsqueeze(1)                    # (T, L, L)
        quad = outer[:, self._tri_i, self._tri_j]                      # (T, Q)
        const = torch.ones(T, 1, device=x.device, dtype=x.dtype)
        return torch.cat((const, lin, quad), 1)                        # (T, D)
    
    def _embed_window(self, window: torch.Tensor) -> torch.Tensor:
        """Embedding for a single window of length k·skip (used in generate)."""
        vec = window[:: self.skip].reshape(-1)          # (L,)
        outer = vec.unsqueeze(1) * vec.unsqueeze(0)     # (L, L)
        quad = outer[self._tri_i, self._tri_j]          # (Q,)
        const = torch.tensor([1.0], device=window.device, dtype=window.dtype)
        return torch.cat((const, vec, quad))            # (D,)

    # --------------------------------------------------------------------- #
    #  Core API                                                             #
    # --------------------------------------------------------------------- #
    def fit(self, x: torch.Tensor) -> float:
        """Fit the linear read-out on the whole sequence and return RMSE."""
        φ = self._embed(x)
        t0 = self.k * self.skip

        if self.predict_mode == "next":
            Y = x[t0:]
        else:
            Y = x[t0:] - x[t0 - 1:-1]

        self.W_out.copy_(ridge_regression(φ, Y, self.alpha_regularizer))
        return self.loss(x).item()

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict over a known trajectory."""
        if not hasattr(self, "W_out"):
            raise AttributeError("Model not fitted.")
        Φ = self._embed(x)
        return Φ @ self.W_out

    # --------------------------------------------------------------------- #
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """RMSE on the supplied sequence."""
        preds = self.forward(x)
        t0 = self.k * self.skip
        target = x[t0:] if self.predict_mode == "next" else x[t0:] - x[t0 - 1:-1]
        return ((preds - target) ** 2).sum(-1).sqrt().mean()

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, x_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Autoregressive rollout starting from the last window of `x_init`.
        `x_init` must contain at least k·skip consecutive samples.
        """
        win_size = self.k * self.skip
        if x_init.size(0) < win_size:
            raise ValueError("x_init too short for generation.")
        
        hist = x_init[-win_size:].clone()
        preds = []

        for _ in range(num_steps):
            φ = self._embed_window(hist)               # (D,)
            pred = φ @ self.W_out                      # (F,)
            x_new = hist[-1] + pred if self.predict_mode == "residual" else pred
            preds.append(x_new)
            hist = torch.cat((hist[1:], x_new.unsqueeze(0)), 0)
            
        return torch.stack(preds)