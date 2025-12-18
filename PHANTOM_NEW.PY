def gen_shepp_logan(num_rows, num_cols, modified=True):
    """
    Generate a Shepp Logan phantom
    
    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        modified: bool, if True use modified (visible) values, if False use original values
    
    Return:
        out_image: 2D array, num_rows*num_cols
    """
    
    if modified:
        # MODIFIED SHEPP-LOGAN VALUES (commonly used - much better visibility!)
        sl_paras = [
            {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': -0.8},
            {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': -0.2},
            {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': -0.2},
            {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.1},
            {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.1},
            {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.1},
            {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.1},
            {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.1},
            {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.1}
        ]
    else:
        # ORIGINAL SHEPP-LOGAN VALUES (poor visibility)
        sl_paras = [
            {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 2.0},
            {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': -0.98},
            {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': -0.02},
            {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': -0.02},
            {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.01},
            {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
            {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
            {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
            {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
            {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.01}
        ]

    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)

    x_grid, y_grid = np.meshgrid(axis_x, axis_y)
    image = x_grid * 0.0

    for el_paras in sl_paras:
        image += _gen_ellipse(x_grid=x_grid, y_grid=y_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                             a=el_paras['a'], b=el_paras['b'], theta=el_paras['theta'] / 180.0 * np.pi,
                             gray_level=el_paras['gray_level'])

    return image


def gen_shepp_logan_3d_raw(num_rows, num_cols, num_slices, scale=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0, modified=True):
    """
    Generate a 3D Shepp Logan phantom based on below reference.
    
    Kak AC, Slaney M. Principles of computerized tomographic imaging. Page.102. IEEE Press, New York, 1988.
    
    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        num_slices: int, number of slices.
        scale: (scalar, optional), scaling factor of phantom within the image. from 0 to 1
        offset_x: scalar, proportion of x-axis that is added to x-coordinate of phantom within image
        offset_y: scalar, proportion of y-axis that is added to y-coordinate of phantom within image
        offset_z: scalar, proportion of z-axis that is added to z-coordinate of phantom within image
        modified: bool, if True use modified (visible) values
    
    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
    """
    
    if modified:
        # MODIFIED 3D SHEPP-LOGAN VALUES (better visibility!)
        sl3d_paras = [
            {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.8},
            {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.2},
            {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.2},
            {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.1},
            {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.1},
            {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.1},
            {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.1},
            {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.1},
            {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.2}
        ]
    else:
        # ORIGINAL 3D SHEPP-LOGAN VALUES
        sl3d_paras = [
            {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 2.0},
            {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.98},
            {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.02},
            {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.02},
            {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.02},
            {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.02},
            {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.01},
            {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.01},
            {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.02},
            {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.02}
        ]

    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    axis_z = np.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = np.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    shift_x = offset_x * 2.0
    shift_y = offset_y * 2.0
    shift_z = offset_z * 2.0

    for el_paras in sl3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, 
                               x0=el_paras['x0']*scale - shift_x, 
                               y0=el_paras['y0']*scale - shift_y,
                               z0=el_paras['z0']*scale - shift_z,
                               a=el_paras['a']*scale, b=el_paras['b']*scale, c=el_paras['c']*scale,
                               gamma=el_paras['gamma'] / 180.0 * np.pi,
                               gray_level=el_paras['gray_level'])

    return np.transpose(image, (2, 0, 1))


def gen_shepp_logan_3d(num_rows, num_cols, num_slices, block_size=(2,2,2), scale=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0, modified=True):
    """
    Generate a 3D Shepp Logan phantom.
    
    Args:
        num_rows: int, number of rows in generated phantom.
        num_cols: int, number of columns in generated phantom.
        num_slices: int, number of slices in generated phantom.
        block_size: (int, int, int): size of block average.
        scale: scaling factor of phantom within the image.
        offset_x: proportion of x-axis offset
        offset_y: proportion of y-axis offset
        offset_z: proportion of z-axis offset
        modified: bool, if True use modified (visible) values
    
    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
    """
    phantom_raw = gen_shepp_logan_3d_raw(num_rows*block_size[1], num_cols*block_size[2], num_slices*block_size[0],
                                         scale=scale, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                                         modified=modified)
    phantom = phantom_raw.reshape(phantom_raw.shape[0]//block_size[0], block_size[0],
                                  phantom_raw.shape[1]//block_size[1], block_size[1],
                                  phantom_raw.shape[2]//block_size[2], block_size[2]).sum((1, 3, 5)) / (block_size[0]*block_size[1]*block_size[2])
    return phantom


def gen_demo_shepp_logan(num_rows, num_cols, demo_type='high_contrast'):
    """
    Generate a demonstration Shepp-Logan phantom with highly visible features
    
    Args:
        num_rows: int, number of rows
        num_cols: int, number of cols
        demo_type: str, 'high_contrast', 'binary', or 'gradient'
    
    Return:
        out_image: 2D array
    """
    
    if demo_type == 'high_contrast':
        # High contrast demo - clearly visible features
        sl_paras = [
            {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': 0.2},
            {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': 0.8},
            {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': 0.8},
            {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.6},
            {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.7},
            {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.7},
            {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.6},
            {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.6},
            {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.6}
        ]
    elif demo_type == 'binary':
        # Binary demo - only 0 and 1 values
        sl_paras = [
            {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': 0.0},
            {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': 1.0},
            {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 1.0},
            {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 1.0}
        ]
    else:  # gradient
        # Gradient demo - different values for each feature
        sl_paras = [
            {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 1.0},
            {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': 0.3},
            {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': 0.9},
            {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': 0.9},
            {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.7},
            {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.6},
            {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.6},
            {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.5},
            {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.5},
            {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.5}
        ]

    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)

    x_grid, y_grid = np.meshgrid(axis_x, axis_y)
    image = x_grid * 0.0

    for el_paras in sl_paras:
        image += _gen_ellipse(x_grid=x_grid, y_grid=y_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                             a=el_paras['a'], b=el_paras['b'], theta=el_paras['theta'] / 180.0 * np.pi,
                             gray_level=el_paras['gray_level'])

    return image



    plt.show()
