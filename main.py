import os
import numpy as np
import torch
import time
from pathlib import Path

# Your modules
from matrix import compute_sys_matrix
from phantom_gpu import forward_project_fast, back_project_fast
from GD import adam_reconstruction_autograd

# ============================================================
# HARDCODED PARAMETERS (change these as needed)
# ============================================================
DET_ROWS = 64        # Number of detector rows
DET_CHANNELS = 64    # Number of detector channels
NUM_VIEWS = 64       # Number of projection views
ITERATIONS = 50      # Adam iterations
LEARNING_RATE = 0.01 # Learning rate

# ============================================================
# HELPER CLASSES
# ============================================================
class ImgParams:
    def __init__(self, Nx, Ny, Nz, delta_pixel_image):
        self.N_x = Nx; self.N_y = Ny; self.N_z = Nz
        self.Delta_xy = delta_pixel_image
        self.Delta_z = delta_pixel_image
        self.x_0 = -self.N_x * self.Delta_xy / 2.0
        self.y_0 = -self.N_y * self.Delta_xy / 2.0
        self.z_0 = -self.N_z * self.Delta_z / 2.0

class SinoParams:
    def __init__(self, dist_source_detector, magnification, num_views, num_det_rows, num_det_channels):
        self.N_dv = num_det_rows
        self.N_dw = num_det_channels
        self.N_beta = num_views
        self.Delta_dv = 1.0
        self.Delta_dw = 1.0
        self.u_s = -dist_source_detector / magnification
        self.u_r = 0.0
        self.v_r = 0.0
        self.u_d0 = dist_source_detector - dist_source_detector / magnification
        self.v_d0 = -self.N_dv * self.Delta_dv / 2.0
        self.w_d0 = -self.N_dw * self.Delta_dw / 2.0

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("=" * 60)
    print("3D Cone-Beam CT Reconstruction")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parameters
    Nz = DET_ROWS
    Ny = DET_CHANNELS
    Nx = DET_CHANNELS
    magnification = 2.0
    dist_source_detector = 3.0 * DET_CHANNELS
    delta_pixel_image = 1.0 / magnification
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    
    # Create output directory
    output_dir = Path("frontier_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    print(f"\nProblem size:")
    print(f"  Volume: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} voxels")
    print(f"  Sinogram: {NUM_VIEWS} views x {DET_ROWS} rows x {DET_CHANNELS} channels")
    
    # ========== 1. GENERATE PHANTOM ==========
    print("\n" + "-" * 40)
    print("1. Generating phantom...")
    start_time = time.time()
    
    # Simple cube phantom
    phantom = torch.zeros((Nz, Ny, Nx), device=device)
    size = int(min(Nz, Ny, Nx) * 0.6)  # 60% of smallest dimension
    z0, y0, x0 = (Nz-size)//2, (Ny-size)//2, (Nx-size)//2
    phantom[z0:z0+size, y0:y0+size, x0:x0+size] = 1.0
    
    phantom_time = time.time() - start_time
    print(f"   ✓ Created {size}x{size}x{size} cube in {phantom_time:.2f}s")
    print(f"   Phantom range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    
    # ========== 2. SETUP PARAMETERS ==========
    print("\n2. Setting up parameters...")
    img_params = ImgParams(Nx, Ny, Nz, delta_pixel_image)
    sino_params = SinoParams(dist_source_detector, magnification, NUM_VIEWS, DET_ROWS, DET_CHANNELS)
    
    # ========== 3. COMPUTE SYSTEM MATRIX ==========
    print("\n3. Computing system matrix...")
    start_time = time.time()
    
    A = compute_sys_matrix(sino_params, img_params, angles, device=device)
    
    matrix_time = time.time() - start_time
    print(f"   ✓ System matrix computed in {matrix_time:.2f}s")
    print(f"   i_vstride_max = {A['i_vstride_max']}, i_wstride_max = {A['i_wstride_max']}")
    
    # ========== 4. FORWARD PROJECTION ==========
    print("\n4. Forward projection...")
    start_time = time.time()
    
    sino = forward_project_fast(phantom, A, img_params, sino_params, device=device)
    
    proj_time = time.time() - start_time
    print(f"   ✓ Forward projection in {proj_time:.2f}s")
    print(f"   Sinogram shape: {sino.shape}")
    
    # Save phantom and sinogram
    np.save(output_dir / 'phantom.npy', phantom.cpu().numpy())
    np.save(output_dir / 'sinogram.npy', sino.cpu().numpy())
    print(f"   ✓ Saved phantom.npy and sinogram.npy")
    
    # ========== 5. RECONSTRUCTION ==========
    print(f"\n5. ADAM Reconstruction ({ITERATIONS} iterations)...")
    start_time = time.time()
    
    recon, loss_history = adam_reconstruction_autograd(
        sino=sino,
        A=A,
        img_params=img_params,
        sino_params=sino_params,
        num_iter=ITERATIONS,
        lr=LEARNING_RATE,
        device=device
    )
    
    recon_time = time.time() - start_time
    print(f"   ✓ Reconstruction completed in {recon_time:.2f}s")
    print(f"   Final loss: {loss_history[-1]:.6e}")
    
    # ========== 6. SAVE RESULTS ==========
    print("\n6. Saving results...")
    
    # Save reconstruction
    recon_np = recon.cpu().numpy()
    np.save(output_dir / 'reconstruction.npy', recon_np)
    np.save(output_dir / 'loss_history.npy', np.array(loss_history))
    
    # Summary
    print("\n" + "=" * 60)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Total time: {phantom_time + matrix_time + proj_time + recon_time:.2f}s")
    print(f"\nFiles saved to {output_dir}/:")
    print(f"  phantom.npy          - {phantom.shape} original phantom")
    print(f"  sinogram.npy         - {sino.shape} projection data")
    print(f"  reconstruction.npy   - {recon.shape} reconstructed volume")
    print(f"  loss_history.npy     - {len(loss_history)} loss values")
    
    # Show memory usage
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory usage: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("\nTo download results to your local machine:")
    print(f"  scp $USER@frontier.olcf.ornl.gov:{output_dir.absolute()}/*.npy .")
    print("=" * 60)

if __name__ == '__main__':
    main()
