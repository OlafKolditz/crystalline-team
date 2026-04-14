import numpy as np
from scipy import special as sc 

import meshio

#-----------------------------------------------------------------------------
def pd_lsrf_nb(t, ddict, k, r):
    """
    Pd Line-Source Radial-flow with no boundary (infinite reservoir)
    Supports ONE vectorized parameter at a time (k or r).
    Dimensionless variables (tD, rD) are calculated relative to r_w.
    Inputs.
        t: time
        ddict: dictionary with parameters/properties (mu, por, ct, rw)
        k: permeability (scalar or array)
        r: radius (scalar or array)
    Ouputs.
        pd: dimensionless pressure (array)
    """
    t = np.atleast_1d(t).reshape(-1, 1)
    k = np.atleast_1d(k).flatten()[None, :]
    r = np.atleast_1d(r).flatten()[None, :]

    mu = ddict['mu']
    por = ddict['por']
    c_t = ddict['c_t']
    r_w = ddict['r_w'] 

    t_safe = np.maximum(t, 1e-12) #avoiding zero division
    td = (k * t_safe) / (por * mu * c_t * (r_w**2))
    rd = r / r_w  
    
    pd = -(1 / 2) * sc.expi(-(rd**2) / (4 * td))
    pd = np.where(t <= 0, 0.0, pd)
    
    if pd.shape[1] == 1:
        return pd.ravel() #flatten 1D arrays
    return pd

#---------------------------------------------------------------------------
def step_rate_r(func, delta_t, tp, q_array, rd_dict, k_val, r_val, *args):
    """
    2D Optimized Step Rate for radial-flow family. 
    Supports ONE vectorized parameter (k, r, or an arg).
    Inputs.
        func: function-call to evaluate the steps 
        delta_t: time value of test
        tp: T time of step protocol change
        q_array: flowrate of test in every step T
        rd_dict: dictionary with parameters/properties (pi,mu,b,h)
        k_val: permeability value (scalar or array)
        r_val: radii of evaluation (scalar or array)
        *arg: ordered argument of function kernel used
    Ouput:
        p_ws: pressure of the step rate in consistent units. 
    """
    pi, mu, B = rd_dict['p_i'], rd_dict['mu'], rd_dict.get('B', 1.0)
    h = rd_dict.get('h', 1.0) #if not h or unsuccesful retrive, then h=1.0
    k_arr = np.atleast_1d(k_val).flatten()[None, :] 
    r_arr = np.atleast_1d(r_val).flatten()[None, :] 
   
    dt_matrix = delta_t[:, None] - tp[None, :]    # delta_t (N, 1), tp (1, M) -> dt_matrix (N, M)
    mask = dt_matrix > 1e-12
    dt_safe = np.where(mask, dt_matrix, 0.0)
    
    pwd_raw = func(dt_safe.ravel(), rd_dict, k_arr, r_arr, *args)
    num_times, num_events = dt_matrix.shape
    num_scenarios = pwd_raw.size // (num_times * num_events) #automatic escenarios
    
    pwd_3d = pwd_raw.reshape(num_times, num_events, num_scenarios)
    dq = np.diff(q_array, prepend=0)
    
    summation = np.einsum('j,ijk->ik', dq, pwd_3d) # Superposition via Einstein Summation:
    
    C_const = (mu * B) / (2 * np.pi * k_arr * h)
    p_ws = pi - (C_const * summation)
    
    p_ws = np.where(delta_t[:, None] <= 1e-12, pi, p_ws)
    
    offset = p_ws[0, :] - pi
    p_ws = p_ws - offset
    
    return p_ws.squeeze()

#----------------------------------------------------------------------

def save_combined_mesh(msh_file, output_path, fracture_label="fracture"):
    """
    Reads MSH and saves a single VTU with all elements and MaterialIDs.
    Focus on changing the ID of the fracture from domain
    Inputs.
        msh_file: Mesh file (path/file.msh)
        output_path: New vtu file (path/file.msh)
        fracture_label: name of element to change
    Output.
        vtu file with combined elements (rock+fracture) with different IDs
    """
   
    msh = meshio.read(msh_file)
    try:
        target_id = msh.field_data[fracture_label][0]
    except KeyError:
        print(f"Warning: '{fracture_label}' not found. Combined mesh may lack IDs.")
        return

    valid_cells = []
    valid_material_ids = []

    for i, cell_block in enumerate(msh.cells):
        if cell_block.type in ["line", "triangle", "quad"]:
            n_cells = len(cell_block.data)
            block_ids = np.zeros(n_cells, dtype=np.int32)
            
            if i < len(msh.cell_data.get("gmsh:physical", [])):
                ids_in_block = msh.cell_data["gmsh:physical"][i]
                block_ids[ids_in_block == target_id] = 1
            
            valid_cells.append(cell_block)
            valid_material_ids.append(block_ids)

    combined_mesh = meshio.Mesh(
        points=msh.points,
        cells=valid_cells,
        cell_data={"MaterialIDs": valid_material_ids}
    )
    combined_mesh.write(output_path)
    print(f"Combined mesh saved to: {output_path}")

