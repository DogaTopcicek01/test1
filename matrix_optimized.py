import torch
import math
import time
import numpy as np
from numba import jit, prange, float32, int32
import numba

# ======================================================================
#   Numba-optimized core functions
# ======================================================================

@jit(nopython=True, parallel=True)
def compute_amatrix_params_numba(sino_params_dict, img_params_dict, 
                                 view_angles, AMATRIX_RHO=1.0):
    """
    Numba-optimized version of computeAMatrixParameters.
    Returns all parameters as a tuple.
    """
    # Extract parameters from dicts
    N_x = img_params_dict['N_x']
    N_y = img_params_dict['N_y']
    N_z = img_params_dict['N_z']
    Delta_xy = img_params_dict['Delta_xy']
    x_0 = img_params_dict['x_0']
    y_0 = img_params_dict['y_0']
    z_0 = img_params_dict['z_0']
    Delta_z = img_params_dict['Delta_z']
    
    N_dv = sino_params_dict['N_dv']
    N_dw = sino_params_dict['N_dw']
    Delta_dv = sino_params_dict['Delta_dv']
    Delta_dw = sino_params_dict['Delta_dw']
    u_s = sino_params_dict['u_s']
    u_d0 = sino_params_dict['u_d0']
    u_r = sino_params_dict['u_r']
    v_r = sino_params_dict['v_r']
    v_d0 = sino_params_dict['v_d0']
    w_d0 = sino_params_dict['w_d0']
    
    N_beta = len(view_angles)
    
    i_vstride_max = 0
    i_wstride_max = 0
    u_0 = np.inf
    u_1 = -np.inf
    B_ij_max = 0.0
    C_ij_max = 0.0
    
    # Precompute view angles
    cos_beta = np.cos(view_angles)
    sin_beta = np.sin(view_angles)
    
    # -------------------------------------------------
    # Part 1: Compute over x, y, beta
    # -------------------------------------------------
    for j_x in prange(N_x):
        x_v = j_x * Delta_xy + (x_0 + Delta_xy / 2.0)
        for j_y in range(N_y):
            y_v = j_y * Delta_xy + (y_0 + Delta_xy / 2.0)
            
            for i_beta in range(N_beta):
                beta = view_angles[i_beta]
                cosine = cos_beta[i_beta]
                sine = sin_beta[i_beta]
                
                # Scanner coordinates
                u_v = cosine * x_v - sine * y_v + u_r
                v_v = sine * x_v + cosine * y_v + v_r
                
                # Magnification
                M = (u_d0 - u_s) / (u_v - u_s)
                
                # Geometry
                theta = math.atan2(v_v, u_v - u_s)
                alpha_xy = beta - theta
                # Fold alpha into [-pi/4, pi/4]
                alpha_xy = ((alpha_xy + math.pi/4.0) % (math.pi/2.0)) - math.pi/4.0
                
                # Eq. for W_pv
                W_pv = M * Delta_xy * math.cos(alpha_xy) / math.cos(theta)
                
                # i_vstart
                i_vstart = round(
                    (M * v_v - W_pv / 2.0 - (v_d0 + Delta_dv / 2.0)) / Delta_dv
                )
                i_vstart = max(0, i_vstart)
                
                # temp_stop
                temp_stop = round(
                    (M * v_v + W_pv / 2.0 - (v_d0 + Delta_dv / 2.0)) / Delta_dv
                )
                temp_stop = min(temp_stop, N_dv - 1)
                
                # stride in v
                i_vstride = max(temp_stop - i_vstart + 1, 0)
                if i_vstride > i_vstride_max:
                    i_vstride_max = i_vstride
                
                # Update u_0, u_1
                voxel_u_min = u_v - Delta_xy / 2.0
                if voxel_u_min < u_0:
                    u_0 = voxel_u_min
                if voxel_u_min > u_1:
                    u_1 = voxel_u_min
    
    # Compute Delta_u, N_u
    Delta_u = Delta_xy / AMATRIX_RHO
    N_u = math.ceil((u_1 - u_0) / Delta_u) + 1
    u_1 = u_0 + N_u * Delta_u
    
    # -------------------------------------------------
    # Part 2: Compute over u, z
    # -------------------------------------------------
    for j_u in prange(N_u):
        u_v = j_u * Delta_u + (u_0 + Delta_xy / 2.0)
        M = (u_d0 - u_s) / (u_v - u_s)
        W_pw = M * Delta_z
        
        for j_z in range(N_z):
            w_v = j_z * Delta_z + (z_0 + Delta_z / 2.0)
            
            i_wstart = int(
                (M * w_v - (w_d0 + Delta_dw / 2.0) - W_pw / 2.0) / Delta_dw + 0.5
            )
            i_wstart = max(i_wstart, 0)
            
            temp_stop = int(
                (M * w_v - (w_d0 + Delta_dw / 2.0) + W_pw / 2.0) / Delta_dw + 0.5
            )
            temp_stop = min(temp_stop, N_dw - 1)
            
            i_wstride = max(temp_stop - i_wstart + 1, 0)
            if i_wstride > i_wstride_max:
                i_wstride_max = i_wstride
    
    return (i_vstride_max, i_wstride_max, u_0, u_1, 
            Delta_u, N_u, B_ij_max, C_ij_max)


@jit(nopython=True, parallel=True)
def compute_bmatrix_numba(sino_params_dict, img_params_dict, view_angles,
                         u_0, Delta_u, i_vstride_max, B_ij_scaler,
                         output_dict):
    """
    Compute B matrix with Numba acceleration.
    """
    # Extract parameters
    N_x = img_params_dict['N_x']
    N_y = img_params_dict['N_y']
    Delta_xy = img_params_dict['Delta_xy']
    x_0 = img_params_dict['x_0']
    y_0 = img_params_dict['y_0']
    
    N_dv = sino_params_dict['N_dv']
    Delta_dv = sino_params_dict['Delta_dv']
    u_s = sino_params_dict['u_s']
    u_d0 = sino_params_dict['u_d0']
    u_r = sino_params_dict['u_r']
    v_r = sino_params_dict['v_r']
    v_d0 = sino_params_dict['v_d0']
    
    N_beta = len(view_angles)
    
    # Precompute trig functions
    cos_beta = np.cos(view_angles)
    sin_beta = np.sin(view_angles)
    
    # Get output arrays
    B = output_dict['B']
    i_vstart = output_dict['i_vstart']
    i_vstride = output_dict['i_vstride']
    j_u_array = output_dict['j_u']
    
    # Main computation loop
    for j_x in prange(N_x):
        x_v = j_x * Delta_xy + (x_0 + Delta_xy / 2.0)
        for j_y in range(N_y):
            y_v = j_y * Delta_xy + (y_0 + Delta_xy / 2.0)
            
            for i_beta in range(N_beta):
                cosine = cos_beta[i_beta]
                sine = sin_beta[i_beta]
                
                # Scanner coordinates
                u_v = cosine * x_v - sine * y_v + u_r
                v_v = sine * x_v + cosine * y_v + v_r
                
                # Magnification
                M = (u_d0 - u_s) / (u_v - u_s)
                
                # Geometry
                theta = math.atan2(v_v, u_v - u_s)
                alpha_xy = view_angles[i_beta] - theta
                alpha_xy = ((alpha_xy + math.pi/4.0) % (math.pi/2.0)) - math.pi/4.0
                
                W_pv = M * Delta_xy * math.cos(alpha_xy) / math.cos(theta)
                
                # Detector indices
                i_vstart_val = int(
                    (M * v_v - W_pv/2.0 - (v_d0 + Delta_dv/2.0)) / Delta_dv + 0.5
                )
                i_vstart_val = max(i_vstart_val, 0)
                
                temp_stop = int(
                    (M * v_v + W_pv/2.0 - (v_d0 + Delta_dv/2.0)) / Delta_dv + 0.5
                )
                temp_stop = min(temp_stop, N_dv - 1)
                
                i_vstride_val = max(temp_stop - i_vstart_val + 1, 0)
                
                # Store results
                i_vstart[j_x, j_y, i_beta] = i_vstart_val
                i_vstride[j_x, j_y, i_beta] = i_vstride_val
                
                # j_u index
                j_u_val = int((u_v - (u_0 + Delta_xy/2.0)) / Delta_u + 0.5)
                j_u_array[j_x, j_y, i_beta] = j_u_val
                
                # Compute B values for this footprint
                if i_vstride_val > 0:
                    cos_alpha = math.cos(alpha_xy)
                    for i_v_offset in range(i_vstride_val):
                        i_v = i_vstart_val + i_v_offset
                        v_d = v_d0 + Delta_dv/2.0 + i_v * Delta_dv
                        
                        delta_v = abs(v_d - M * v_v)
                        
                        # L_v calculation
                        L_v = (W_pv - Delta_dv) / 2.0
                        L_v = abs(L_v)
                        L_v = max(L_v, delta_v)
                        L_v = (W_pv + Delta_dv) / 2.0 - L_v
                        L_v = max(L_v, 0.0)
                        
                        B_ij = Delta_xy * L_v / (cos_alpha * Delta_dv)
                        
                        # Store compressed value
                        idx = i_beta * i_vstride_max + i_v_offset
                        B[j_x, j_y, idx] = B_ij / B_ij_scaler + 0.5


# ======================================================================
#   Main wrapper function (compatible with your existing code)
# ======================================================================

def compute_sys_matrix_optimized(sino_params, img_params, view_angle_list,
                                 device='cpu',
                                 ISBIJCOMPRESSED=True,
                                 ISCIJCOMPRESSED=True,
                                 AMATRIX_RHO=1.0):
    """
    Optimized version with Numba acceleration.
    """
    start_time = time.time()
    
    # Convert parameters to dicts for Numba
    sino_dict = {
        'N_dv': sino_params.N_dv,
        'N_dw': sino_params.N_dw,
        'Delta_dv': sino_params.Delta_dv,
        'Delta_dw': sino_params.Delta_dw,
        'u_s': sino_params.u_s,
        'u_d0': sino_params.u_d0,
        'u_r': sino_params.u_r,
        'v_r': sino_params.v_r,
        'v_d0': sino_params.v_d0,
        'w_d0': sino_params.w_d0
    }
    
    img_dict = {
        'N_x': img_params.N_x,
        'N_y': img_params.N_y,
        'N_z': img_params.N_z,
        'Delta_xy': img_params.Delta_xy,
        'Delta_z': img_params.Delta_z,
        'x_0': img_params.x_0,
        'y_0': img_params.y_0,
        'z_0': img_params.z_0
    }
    
    view_angles = np.array(view_angle_list, dtype=np.float32)
    
    # Step 1: Compute parameters with Numba
    print("Computing system matrix parameters...")
    params = compute_amatrix_params_numba(
        sino_dict, img_dict, view_angles, AMATRIX_RHO
    )
    
    i_vstride_max, i_wstride_max, u_0, u_1, Delta_u, N_u, B_ij_max, C_ij_max = params
    
    # Step 2: Allocate arrays
    N_x, N_y, N_z = img_params.N_x, img_params.N_y, img_params.N_z
    N_beta = len(view_angle_list)
    
    A = allocateSysMatrix(N_x, N_y, N_z, N_beta,
                         i_vstride_max, i_wstride_max, N_u,
                         device=device)
    
    # Add computed parameters
    A['i_vstride_max'] = i_vstride_max
    A['i_wstride_max'] = i_wstride_max
    A['u_0'] = u_0
    A['u_1'] = u_1
    A['Delta_u'] = Delta_u
    A['N_u'] = N_u
    
    # Scaler values
    if ISBIJCOMPRESSED:
        A['B_ij_scaler'] = B_ij_max / 255.0 if B_ij_max > 0 else 1.0
    else:
        A['B_ij_scaler'] = 1.0
    
    if ISCIJCOMPRESSED:
        A['C_ij_scaler'] = C_ij_max / 255.0 if C_ij_max > 0 else 1.0
    else:
        A['C_ij_scaler'] = 1.0
    
    # Step 3: Compute B matrix with Numba
    print("Computing B matrix...")
    output_dict = {
        'B': A['B'].cpu().numpy(),
        'i_vstart': A['i_vstart'].cpu().numpy(),
        'i_vstride': A['i_vstride'].cpu().numpy(),
        'j_u': A['j_u'].cpu().numpy()
    }
    
    compute_bmatrix_numba(
        sino_dict, img_dict, view_angles,
        u_0, Delta_u, i_vstride_max, A['B_ij_scaler'],
        output_dict
    )
    
    # Copy back to GPU if needed
    if device != 'cpu':
        for key in ['B', 'i_vstart', 'i_vstride', 'j_u']:
            A[key] = torch.from_numpy(output_dict[key]).to(device)
    
    # Step 4: Compute C matrix (could also be optimized with Numba)
    print("Computing C matrix...")
    computeCMatrix(sino_params, img_params, A, ISCIJCOMPRESSED)
    
    end_time = time.time()
    print(f"System matrix computed in {end_time - start_time:.2f} seconds")
    
    return A


# Keep your existing allocateSysMatrix and computeCMatrix functions
# They can be optimized similarly if needed
