import torch
from typing import Optional, Dict, Any


# ============================================================
#  Forward projector: y = A x (FRONTIER-OPTIMIZED)
# ============================================================
def forward_project_fast(
        x: torch.Tensor,
        A: Dict[str, torch.Tensor],
        img_params: Any,
        sino_params: Any,
        device: Optional[torch.device] = None,
        show_progress: bool = False,
) -> torch.Tensor:
    if device is None:
        device = x.device

    Nz, Ny, Nx = x.shape
    N_beta = sino_params.N_beta
    N_dv = sino_params.N_dv
    N_dw = sino_params.N_dw

    Ax = torch.zeros((N_beta, N_dv, N_dw), dtype=torch.float32, device=device)

    B = A["B"].to(device).contiguous()
    C = A["C"].to(device).contiguous()
    j_u = A["j_u"].to(device).contiguous()

    i_vstart = A["i_vstart"].to(device).contiguous()
    i_vstride = A["i_vstride"].to(device).contiguous()
    i_vstride_max = int(A["i_vstride_max"])

    i_wstart = A["i_wstart"].to(device).contiguous()
    i_wstride = A["i_wstride"].to(device).contiguous()
    i_wstride_max = int(A["i_wstride_max"])

    B_ij_scaler = float(A["B_ij_scaler"])
    C_ij_scaler = float(A["C_ij_scaler"])

    x_vol = x.to(device).permute(2, 1, 0).contiguous()

    # ========================================================
    # ✅ FIXED: ADD CACHE FOR iw_range (CRITICAL FOR PERFORMANCE)
    # ========================================================
    iw_range_cache: Dict[tuple, torch.Tensor] = {}

    # =========================================================
    # OPTIMIZATION 1: Remove XY loops with torch.where
    # =========================================================
    beta_iter = range(N_beta)

    for i_beta in beta_iter:
        ju_grid = j_u[:, :, i_beta]
        iv0 = i_vstart[:, :, i_beta]
        ivlen = i_vstride[:, :, i_beta]

        base_B_idx = i_beta * i_vstride_max

        # =====================================================
        # OPTIMIZATION 2: Process all valid XY at once
        # =====================================================
        # Create mask of valid (x,y) positions
        xy_mask = (ju_grid >= 0) & (ivlen > 0)
        valid_jx, valid_jy = torch.where(xy_mask)

        if valid_jx.numel() == 0:
            continue

        # =====================================================
        # OPTIMIZATION 3: Process XY in batches for better GPU utilization
        # =====================================================
        XY_BATCH = 256  # Tune for Frontier (AMD MI250X)
        n_xy = valid_jx.numel()

        for batch_start in range(0, n_xy, XY_BATCH):
            batch_end = min(batch_start + XY_BATCH, n_xy)

            batch_jx = valid_jx[batch_start:batch_end]
            batch_jy = valid_jy[batch_start:batch_end]

            # Process this XY batch
            for idx in range(batch_end - batch_start):
                j_x = batch_jx[idx]
                j_y = batch_jy[idx]

                ju = int(ju_grid[j_x, j_y])
                v0 = int(iv0[j_x, j_y])
                vl = int(ivlen[j_x, j_y])

                if vl <= 0:
                    continue

                # V-direction vectorized
                iv_range = torch.arange(v0, v0 + vl, device=device, dtype=torch.int64)
                Bij_idx = base_B_idx + (iv_range - v0)
                B_vec = B_ij_scaler * B[j_x, j_y, Bij_idx]

                if torch.all(B_vec == 0):
                    continue

                x_line = x_vol[j_x, j_y, :]
                iw0_line = i_wstart[ju]
                iwlen_line = i_wstride[ju]

                # Get all z indices with non-zero x AND valid w-range
                z_mask = (x_line != 0) & (iwlen_line > 0)
                valid_z = torch.where(z_mask)[0]

                if valid_z.numel() == 0:
                    continue

                # Process each z-slice (geometry-safe)
                for z_idx in range(valid_z.numel()):
                    j_z = valid_z[z_idx]
                    x_val = x_line[j_z]

                    w0 = int(iw0_line[j_z])
                    wl = int(iwlen_line[j_z])

                    if wl <= 0:
                        continue

                    # =================================================
                    # ✅ FIXED: USE CACHE INSTEAD OF NEW torch.arange
                    # =================================================
                    key = (w0, wl)
                    if key not in iw_range_cache:
                        iw_range_cache[key] = torch.arange(
                            w0, w0 + wl, device=device, dtype=torch.int64
                        )
                    iw_range = iw_range_cache[key]

                    C_idx = j_z * i_wstride_max + (iw_range - w0)
                    C_vec = C_ij_scaler * C[ju, C_idx]

                    if torch.all(C_vec == 0):
                        continue

                    # Outer product accumulation
                    contrib = torch.outer(B_vec * x_val, C_vec)
                    Ax[i_beta, iv_range[:, None], iw_range[None, :]] += contrib

    return Ax


# ============================================================
#  Back projector: x = A^T y (FRONTIER-OPTIMIZED)
# ============================================================
def back_project_fast(
        Ax: torch.Tensor,
        A: Dict[str, torch.Tensor],
        img_params: Any,
        sino_params: Any,
        device: Optional[torch.device] = None,
        show_progress: bool = False,
) -> torch.Tensor:
    if device is None:
        device = Ax.device

    Ax = Ax.to(device).contiguous()

    N_beta = sino_params.N_beta
    Nx = img_params.N_x
    Ny = img_params.N_y
    Nz = img_params.N_z

    x_vol = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=device)

    B = A["B"].to(device).contiguous()
    C = A["C"].to(device).contiguous()
    j_u = A["j_u"].to(device).contiguous()

    i_vstart = A["i_vstart"].to(device).contiguous()
    i_vstride = A["i_vstride"].to(device).contiguous()
    i_vstride_max = int(A["i_vstride_max"])

    i_wstart = A["i_wstart"].to(device).contiguous()
    i_wstride = A["i_wstride"].to(device).contiguous()
    i_wstride_max = int(A["i_wstride_max"])

    B_ij_scaler = float(A["B_ij_scaler"])
    C_ij_scaler = float(A["C_ij_scaler"])

    # ========================================================
    # ✅ FIXED: ADD CACHE FOR iw_range (CRITICAL FOR PERFORMANCE)
    # ========================================================
    iw_range_cache: Dict[tuple, torch.Tensor] = {}

    beta_iter = range(N_beta)

    for i_beta in beta_iter:
        ju_grid = j_u[:, :, i_beta]
        iv0 = i_vstart[:, :, i_beta]
        ivlen = i_vstride[:, :, i_beta]

        base_B_idx = i_beta * i_vstride_max

        # Process valid XY positions
        xy_mask = (ju_grid >= 0) & (ivlen > 0)
        valid_jx, valid_jy = torch.where(xy_mask)

        if valid_jx.numel() == 0:
            continue

        # XY batching
        XY_BATCH = 256
        n_xy = valid_jx.numel()

        for batch_start in range(0, n_xy, XY_BATCH):
            batch_end = min(batch_start + XY_BATCH, n_xy)

            batch_jx = valid_jx[batch_start:batch_end]
            batch_jy = valid_jy[batch_start:batch_end]

            for idx in range(batch_end - batch_start):
                j_x = batch_jx[idx]
                j_y = batch_jy[idx]

                ju = int(ju_grid[j_x, j_y])
                v0 = int(iv0[j_x, j_y])
                vl = int(ivlen[j_x, j_y])

                if vl <= 0:
                    continue

                iv_range = torch.arange(v0, v0 + vl, device=device, dtype=torch.int64)
                Bij_idx = base_B_idx + (iv_range - v0)
                B_vec = B_ij_scaler * B[j_x, j_y, Bij_idx]

                if torch.all(B_vec == 0):
                    continue

                iw0_line = i_wstart[ju]
                iwlen_line = i_wstride[ju]

                # Get all valid z at once
                z_mask = (iwlen_line > 0)
                valid_z = torch.where(z_mask)[0]

                if valid_z.numel() == 0:
                    continue

                # Process each z-slice (geometry-safe)
                for z_idx in range(valid_z.numel()):
                    j_z = valid_z[z_idx]

                    w0 = int(iw0_line[j_z])
                    wl = int(iwlen_line[j_z])

                    if wl <= 0:
                        continue

                    # =================================================
                    # ✅ FIXED: USE CACHE INSTEAD OF NEW torch.arange
                    # =================================================
                    key = (w0, wl)
                    if key not in iw_range_cache:
                        iw_range_cache[key] = torch.arange(
                            w0, w0 + wl, device=device, dtype=torch.int64
                        )
                    iw_range = iw_range_cache[key]

                    C_idx = j_z * i_wstride_max + (iw_range - w0)
                    C_vec = C_ij_scaler * C[ju, C_idx]

                    if torch.all(C_vec == 0):
                        continue

                    sino_block = Ax[i_beta, iv_range[:, None], iw_range[None, :]]
                    contrib = (B_vec[:, None] * C_vec[None, :]) * sino_block
                    x_vol[j_x, j_y, j_z] += contrib.sum()

    return x_vol.permute(2, 1, 0).contiguous()


# ============================================================
#  Autograd wrapper
# ============================================================
class ConeBeamProjectorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, img_params, sino_params, device=None):
        Ax = forward_project_fast(
            x, A, img_params, sino_params, device=device, show_progress=False
        )
        ctx.A = A
        ctx.img_params = img_params
        ctx.sino_params = sino_params
        ctx.device = device if device is not None else x.device
        return Ax

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = back_project_fast(
            grad_output,
            ctx.A,
            ctx.img_params,
            ctx.sino_params,
            device=ctx.device,
            show_progress=False,
        )
        return grad_x, None, None, None, None


def cone_beam_projector(x, A, img_params, sino_params, device=None):
    return ConeBeamProjectorFn.apply(x, A, img_params, sino_params, device)


# ============================================================
#  FRONTIER-READY UTILITIES
# ============================================================
def check_adjoint(x, y, A, img_params, sino_params, device=None):
    """
    Verify that <Ax, y> == <x, A^T y> (should be ~1e-10)
    """
    if device is None:
        device = x.device

    Ax = forward_project_fast(x, A, img_params, sino_params, device)
    ATy = back_project_fast(y, A, img_params, sino_params, device)

    dot1 = torch.dot(Ax.flatten(), y.flatten())
    dot2 = torch.dot(x.flatten(), ATy.flatten())

    rel_error = abs(dot1 - dot2) / max(abs(dot1), abs(dot2), 1e-12)
    return rel_error < 1e-8, float(rel_error)


def memory_footprint(A, img_params, sino_params):
    """
    Estimate memory usage for Frontier planning
    """
    Nx, Ny, Nz = img_params.N_x, img_params.N_y, img_params.N_z
    N_beta = sino_params.N_beta

    # System matrix memory
    B_size = A["B"].numel() * 4  # float32 = 4 bytes
    C_size = A["C"].numel() * 4
    indices_size = (A["j_u"].numel() + A["i_vstart"].numel() +
                    A["i_vstride"].numel() + A["i_wstart"].numel() +
                    A["i_wstride"].numel()) * 4  # int32 = 4 bytes

    total_A = (B_size + C_size + indices_size) / 1e9  # GB

    # Working memory (worst case)
    x_size = Nx * Ny * Nz * 4 / 1e9
    Ax_size = N_beta * sino_params.N_dv * sino_params.N_dw * 4 / 1e9

    return {
        "system_matrix_GB": total_A,
        "volume_GB": x_size,
        "sinogram_GB": Ax_size,
        "total_working_GB": total_A + x_size + Ax_size
    }
