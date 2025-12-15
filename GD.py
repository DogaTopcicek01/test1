import torch
import torch.optim as optim
from functions import ConeBeamProjectorFn


def adam_reconstruction_autograd(sino, A, img_params, sino_params,
                                 num_iter=100, lr=0.01, device='cpu'):
    """
    Reconstruction using Adam with your autograd Function.
    """
    Nz, Ny, Nx = img_params.N_z, img_params.N_y, img_params.N_x

    # Create reconstruction as learnable parameter
    x = torch.nn.Parameter(torch.zeros((Nz, Ny, Nx), device=device))

    # Create optimizer
    optimizer = optim.Adam([x], lr=lr)

    loss_history = []

    print(f"\nAdam with autograd: {num_iter} iterations, lr={lr}")

    for iteration in range(num_iter):
        optimizer.zero_grad()

        # Use your autograd function
        Ax = ConeBeamProjectorFn.apply(x, A, img_params, sino_params, device)

        # Compute loss
        loss = 0.5 * torch.sum((sino - Ax) ** 2)

        # Backward
        loss.backward()

        # Optional: print gradient stats
        if iteration == 0:
            grad_norm = torch.norm(x.grad).item() if x.grad is not None else 0
            print(f"Initial gradient norm: {grad_norm:.6e}")

        # Update
        optimizer.step()

        loss_history.append(loss.item())

        if iteration % 10 == 0:
            print(f"Iter {iteration:3d}: loss = {loss.item():.6e}")

    return x.detach(), loss_history
