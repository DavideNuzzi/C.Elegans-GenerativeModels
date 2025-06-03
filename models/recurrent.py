import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, List, Dict
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#                                Recurrent model                              #
# --------------------------------------------------------------------------- #
class RecurrentModel(nn.Module):
    """
    Simple encoder-RNN-decoder fitted with MSE and teacher forcing.

    Parameters
    ----------
    features_dim      - dimension of each time-step x(t)
    hidden_dim        - size of the latent state h(t) for the encoder / RNN
    recurrent_layers  - number of consecutive RNN layers
    encoder_model     - maps x → h   (defaults to Linear(features_dim, hidden_dim))
    recurrent_model   - maps h_t → h_{t+1} (defaults to one-layer RNN)
    decoder_model     - maps h → x̂   (defaults to Linear(hidden_dim, features_dim))
    predict_mode      - "next"     : learn x(t+1)
                         "residual": learn x(t+1) - x(t)
    """

    def __init__(
        self,
        features_dim: int,
        hidden_dim: int = 64,
        recurrent_layers: int = 1,
        encoder_model: Optional[nn.Module] = None,
        recurrent_model: Optional[nn.Module] = None,
        decoder_model: Optional[nn.Module] = None,
        predict_mode: str = "next",
    ):
        super().__init__()
        assert predict_mode in {"next", "residual"}

        # Defaults
        self.encoder_model = (
            encoder_model
            if encoder_model is not None
            else nn.Linear(features_dim, hidden_dim)
        )
        self.recurrent_model = (
            recurrent_model
            if recurrent_model is not None
            else nn.RNN(hidden_dim, hidden_dim, num_layers=recurrent_layers)
        )
        self.decoder_model = (
            decoder_model
            if decoder_model is not None
            else nn.Linear(hidden_dim, features_dim)
        )

        self.predict_mode = predict_mode

    # --------------------------------------------------------------------- #
    #  Private utilities                                                    #
    # --------------------------------------------------------------------- #
    def _step(self, x: torch.Tensor) -> torch.Tensor:
        """Full pass encoder → RNN → decoder for a sequence (T, F)."""
        h = self.encoder_model(x)               # (T, H)
        h_out, _ = self.recurrent_model(h)      # (T, H)
        y = self.decoder_model(h_out)           # (T, F)
        return y

    # --------------------------------------------------------------------- #
    #  Core API                                                             #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions for each input step."""
        return self._step(x)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """RMSE over the sequence."""
        y_pred = self._step(x[:-1])
        target = x[1:] if self.predict_mode == "next" else x[1:] - x[:-1]
        return ((y_pred - target) ** 2).sum(-1).sqrt().mean()

    def fit(
        self,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer = None,
        epochs: int = 1000,
        epoch_callback_fn=None
    ) -> List[float]:
        
        """Fit on the full sequence; returns list of per-epoch losses."""
        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=3e-4)

        loss_history: List[float] = []
        bar = tqdm(range(epochs))

        for epoch in bar:
            optimizer.zero_grad()

            loss = self.loss(x)
            loss.backward()
            optimizer.step()

            l = loss.item()
            loss_history.append(l)
            bar.set_postfix({"loss": l})

            if epoch_callback_fn is not None:
                epoch_callback_fn(epoch)

        return loss_history

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, x_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Autoregressive rollout. `x_init` may be a single vector (F,)
        or a short warm-up sequence (T₀, F).
        """
        seq = x_init.clone() if x_init.ndim == 2 else x_init.unsqueeze(0)

        for _ in range(num_steps):
            pred = self._step(seq)[-1,:]                  # last prediction
            next_x = seq[-1,:] + pred if self.predict_mode == "residual" else pred
            seq = torch.cat((seq, next_x.unsqueeze(0)), dim=0)

        return seq[len(x_init) :]                       # (num_steps, F)

