import numpy as np
from numba import jit, prange

# Add Numba JIT compilation for critical functions
@jit(nopython=True, parallel=True)
def _gen_ellipsoid_fast(x_coords, y_coords, z_coords, x0, y0, z0, 
                        a, b, c, gray_level, alpha=0, beta=0, gamma=0):
    """
    Numba-accelerated ellipsoid generation.
    """
    # Precompute rotation matrix
    cos_a, sin_a = np.cos(-alpha), np.sin(-alpha)
    cos_b, sin_b = np.cos(-beta), np.sin(-beta)
    cos_g, sin_g = np.cos(-gamma), np.sin(-gamma)
    
    # Rotation matrix R = Rz * Ry * Rx
    r11 = cos_b * cos_g
    r12 = sin_a * sin_b * cos_g - cos_a * sin_g
    r13 = cos_a * sin_b * cos_g + sin_a * sin_g
    
    r21 = cos_b * sin_g
    r22 = sin_a * sin_b * sin_g + cos_a * cos_g
    r23 = cos_a * sin_b * sin_g - sin_a * cos_g
    
    r31 = -sin_b
    r32 = sin_a * cos_b
    r33 = cos_a * cos_b
    
    # Precompute squared radii
    a2 = a * a
    b2 = b * b
    c2 = c * c
    
    nz, ny, nx = x_coords.shape
    image = np.zeros((nz, ny, nx), dtype=np.float32)
    
    # Parallel loop over all voxels
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                dx = x_coords[k, j, i] - x0
                dy = y_coords[k, j, i] - y0
                dz = z_coords[k, j, i] - z0
                
                # Apply rotation
                x_rot = r11*dx + r12*dy + r13*dz
                y_rot = r21*dx + r22*dy + r23*dz
                z_rot = r31*dx + r32*dy + r33*dz
                
                # Check if inside ellipsoid
                if (x_rot*x_rot/a2 + y_rot*y_rot/b2 + z_rot*z_rot/c2) <= 1.0:
                    image[k, j, i] = gray_level
    
    return image


def gen_shepp_logan_3d_memory_efficient(num_rows, num_cols, num_slices, 
                                        block_size=(2,2,2), scale=1.0, 
                                        offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Memory-efficient 3D Shepp-Logan phantom generator.
    Generates phantom in chunks to avoid large meshgrids.
    """
    # High resolution for anti-aliasing
    hr_rows = num_rows * block_size[1]
    hr_cols = num_cols * block_size[2]
    hr_slices = num_slices * block_size[0]
    
    # Generate coordinate grids in chunks
    chunk_size = 64  # Process 64 slices at a time to save memory
    phantom_hr = np.zeros((hr_slices, hr_rows, hr_cols), dtype=np.float32)
    
    # Shepp-Logan 3D parameters
    sl3d_paras = [
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 
         'alpha': 0, 'beta': 0, 'gamma': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88,
         'alpha': 0, 'beta': 0, 'gamma': 0, 'gray_level': -0.98},
        # ... rest of parameters (truncated for brevity)
    ]
    
    shift_x = offset_x * 2.0
    shift_y = offset_y * 2.0
    shift_z = offset_z * 2.0
    
    # Process in chunks to save memory
    for start_slice in range(0, hr_slices, chunk_size):
        end_slice = min(start_slice + chunk_size, hr_slices)
        
        # Create coordinate grids only for this chunk
        z_coords = np.linspace(-1.0, 1.0, hr_slices)[start_slice:end_slice]
        y_coords = np.linspace(1.0, -1.0, hr_rows)
        x_coords = np.linspace(-1.0, 1.0, hr_cols)
        
        z_grid, y_grid, x_grid = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Add each ellipsoid to the chunk
        for params in sl3d_paras:
            chunk = _gen_ellipsoid_fast(
                x_grid, y_grid, z_grid,
                x0=params['x0']*scale - shift_x,
                y0=params['y0']*scale - shift_y,
                z0=params['z0']*scale - shift_z,
                a=params['a']*scale,
                b=params['b']*scale,
                c=params['c']*scale,
                alpha=params['alpha'],
                beta=params['beta'],
                gamma=params['gamma'],
                gray_level=params['gray_level']
            )
            phantom_hr[start_slice:end_slice] += chunk
    
    # Downsample with block averaging
    phantom = phantom_hr.reshape(
        hr_slices//block_size[0], block_size[0],
        hr_rows//block_size[1], block_size[1],
        hr_cols//block_size[2], block_size[2]
    ).sum(axis=(1, 3, 5)) / (block_size[0]*block_size[1]*block_size[2])
    
    return phantom
