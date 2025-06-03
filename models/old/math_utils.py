import torch

def ridge_regression(R: torch.Tensor,
                     Y: torch.Tensor,
                     alpha: float) -> torch.Tensor:
    """
    Solves   W = argmin ‖R W - Y‖²_F  + α ‖W‖²_F
    = (RᵀR + α I)⁻¹ RᵀY      
    """

    # Promote to float64 for numerical parity with scikit-learn
    # If I don't do it the Cholesky decomposition does not work
    R, Y = R.double(), Y.double()

    # Bbuild the Gram matrix  G = RᵀR + αI   (SPD by construction)
    G = R.T @ R
    G.diagonal().add_(alpha)          # in-place: G_ii += alpha

    # Solve G W = RᵀY  via Cholesky (fast & stable)
    L = torch.linalg.cholesky(G)                      # G = L Lᵀ
    W = torch.cholesky_solve(R.T @ Y, L)        # (n_f, n_t)

    # Cast back to the original dtype and return
    return W.to(dtype=R.dtype)
