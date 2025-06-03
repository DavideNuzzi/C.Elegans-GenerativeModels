# lorenz_dataset.py
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Union, Tuple


# ------------------------------------------------------------------------- #
#  Internal helpers                                                         #
# ------------------------------------------------------------------------- #
def _lorenz_rhs(t: float, x: np.ndarray,
                rho: float, sigma: float, beta: float) -> np.ndarray:
    """dx/dt for the classic Lorenz system."""
    return np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


def _simulate_lorenz(x0: np.ndarray,
                     t: np.ndarray,
                     rho: float,
                     sigma: float,
                     beta: float) -> np.ndarray:
    """Numerically integrate the Lorenz equations over `t`."""
    sol = solve_ivp(
        fun=lambda _t, _x: _lorenz_rhs(_t, _x, rho, sigma, beta),
        t_span=(float(t[0]), float(t[-1])),
        y0=x0,
        t_eval=t,
        method="RK45",
    )
    return sol.y.T         # shape (T, 3)


# ------------------------------------------------------------------------- #
#  Dataset                                                                  #
# ------------------------------------------------------------------------- #
class LorenzDataset(Dataset):
    r"""
    On-the-fly Lorenz‐attractor simulator yielding **either**
    sliding windows *or* the whole trajectory.

    Parameters
    ----------
    x0 : array-like, shape (3,)
        Initial condition.
    t_span : tuple(float, float), default=(0.0, 50.0)
        Start/end time of the simulation.
    dt : float, default=0.01
        Fixed step for the time grid (`np.arange` is used).
    rho, sigma, beta : float
        Lorenz parameters (defaults are the classic chaotic values).
    mode : {"window", "full"}, default="window"
        • "window": serve fixed-length windows (see below).  
        • "full"  : dataset length == 1, returns the complete series.
    window_length : int | None
        Mandatory if `mode=="window"`.  Number of consecutive samples
        per item.
    window_shift : int, default=1
        Stride between the **starts** of successive windows.
    standardize : bool, default=False
        Z-score the whole simulation before slicing.
    device : str | torch.device | None, default=None
        • None → keep the tensor on CPU.  
        • something like "cuda:0" → move the *entire* data tensor once.
    dtype : torch.dtype, default=torch.float32
        Data type of the stored/returned tensor.
    """

    def __init__(
        self,
        x0: Union[np.ndarray, Tuple[float, float, float]],
        *,
        t_span: Tuple[float, float] = (0.0, 50.0),
        dt: float = 0.01,
        rho: float = 28.0,
        sigma: float = 10.0,
        beta: float = 8.0 / 3.0,
        mode: str = "window",
        window_length: Optional[int] = None,
        window_shift: int = 1,
        standardize: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if mode not in {"window", "full"}:
            raise ValueError('`mode` must be "window" or "full"')

        self.mode = mode
        self.window_length = window_length
        self.window_shift = window_shift

        # Generate the time-series (NumPy)                               
        t0, t1 = map(float, t_span)
        if dt <= 0 or t1 <= t0:
            raise ValueError("Need t1 > t0 and dt > 0")

        times = np.arange(t0, t1 + dt * 0.5, dt, dtype=np.float64)
        series_np = _simulate_lorenz(np.asarray(x0, dtype=np.float64), times, rho, sigma, beta)  

        # Torch conversion & optional standardisation
        series = torch.as_tensor(series_np, dtype=dtype)
        if standardize:
            mean = series.mean(0, keepdim=True)
            std = series.std(0, keepdim=True).clamp(min=1e-8)
            series = (series - mean) / std

        # Move to device once
        if device is not None:
            series = series.to(device)

        self._series = series                                  
        T = series.size(0)

        # Compute length depending on mode
        if self.mode == "window":
            if self.window_length is None:
                raise ValueError("`window_length` must be set in window mode.")
            if self.window_length > T:
                raise ValueError("window_length exceeds generated series length.")
            self._n_items = (T - self.window_length) // self.window_shift + 1
        else:  # full
            self._n_items = 1

    # --------------------------------------------------------------------- #
    #  PyTorch Dataset API                                                  #
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self._n_items

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._n_items:
            raise IndexError("Index out of range for LorenzDataset")

        if self.mode == "full":
            # Always return the complete trajectory
            return self._series

        # Window mode
        start = idx * self.window_shift
        end = start + self.window_length
        return self._series[start:end]                        # view, no copy

    # --------------------------------------------------------------------- #
    #  Convenience getters                                                  #
    # --------------------------------------------------------------------- #
    @property
    def series(self) -> torch.Tensor:
        """Full (T, 3) tensor – useful if you need direct access."""
        return self._series

    @property
    def times(self) -> torch.Tensor:
        """1-D tensor of simulation times (assumes constant dt)."""
        return torch.arange(
            0, len(self._series), device=self._series.device, dtype=self._series.dtype
        )

    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        T, F = self._series.shape
        extra = (
            f"windows={self._n_items}, win_len={self.window_length}, shift={self.window_shift}"
            if self.mode == "window"
            else "full_sequence"
        )
        return (
            f"{self.__class__.__name__}({extra}, time_steps={T}, "
            f"features={F}, device={self._series.device})"
        )
