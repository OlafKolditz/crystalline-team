from pathlib import Path
import gmsh
import math

def create_scylindre_mesh(
    filepath: Path,
    radius: float,
    thickness: float,
    mesh_size: float,
    r_well: float = 0.0,      # Starting radius (0 for line, >0 for finite)
    refine_size: float=0.1, 
    center_y: float = 0.0
) -> None:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.add(filepath.stem)

    z = center_y
    r_start = r_well 
    r_end = r_well + radius

    p1 = gmsh.model.occ.addPoint(r_start,      z+thickness/2, 0, mesh_size)
    p2 = gmsh.model.occ.addPoint(r_end,   z+thickness/2, 0, mesh_size)
    p3 = gmsh.model.occ.addPoint(r_end,   z-thickness/2, 0, mesh_size)
    p4 = gmsh.model.occ.addPoint(r_start,      z-thickness/2, 0,  mesh_size)

    l1 = gmsh.model.occ.addLine(p2, p1)
    l2 = gmsh.model.occ.addLine(p3, p2)
    l3 = gmsh.model.occ.addLine(p4, p3)
    l4 = gmsh.model.occ.addLine(p1, p4)
    gmsh.model.occ.synchronize()

    cl   = gmsh.model.occ.addCurveLoop([l4, l3, l2, l1])
    surf = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.synchronize()

    # --- local refinement ---
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [l4])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", refine_size)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r_well + refine_size * 5) 
    gmsh.model.mesh.field.setNumber(2, "DistMax", radius*0.25) 

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    # -------------------------

    pg_domain = gmsh.model.addPhysicalGroup(2, [surf])
    gmsh.model.setPhysicalName(2, pg_domain, "domain")
    bcs = [("top", l1), ("boundary_R", l2), ("bottom", l3), ("well", l4)]
    for name, line in bcs:
        pg = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, pg, name)

    gmsh.model.mesh.generate(2)
    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()

#------------------------------------------------------------------------
def create_rectangle_frac_mesh(
    filepath: Path,
    radius: float,
    height: float,
    mesh_size: float,
    center_z: float = 0.0,
    r_well: float = 0.01,
    length:float=8.,
    mode="domain",
) -> None:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.add(filepath.stem)
    
    z = center_z
    r_start = r_well
    r_end=radius+r_start

    p1 = gmsh.model.occ.addPoint(r_start, z + height / 2, 0.0, mesh_size)
    p2 = gmsh.model.occ.addPoint(r_end, z + height / 2, 0.0, mesh_size/10)
    p3 = gmsh.model.occ.addPoint(r_end, z - height / 2, 0.0, mesh_size/10)
    p4 = gmsh.model.occ.addPoint(r_start, z - height / 2, 0.0, mesh_size)
    
    p5 = gmsh.model.occ.addPoint(r_start, z, 0.0, mesh_size/10)
    p6 = gmsh.model.occ.addPoint(length, z, 0.0, mesh_size/10)
    p7= gmsh.model.occ.addPoint(r_end, z, 0.0, mesh_size)
    gmsh.model.occ.synchronize()
    
    l1 = gmsh.model.occ.addLine(p2, p1)
    l2 = gmsh.model.occ.addLine(p1, p5)
    l3 = gmsh.model.occ.addLine(p5, p4)
    l4 = gmsh.model.occ.addLine(p4, p3)
    l5 = gmsh.model.occ.addLine(p3, p7)
    l6 = gmsh.model.occ.addLine(p7, p2)

    l7 = gmsh.model.occ.addLine(p5, p6)
    l8 = gmsh.model.occ.addLine(p6, p7)
    gmsh.model.occ.synchronize()
    
    cl1 = gmsh.model.occ.addCurveLoop([l1,l2,l7,l8,l6])
    surf1 = gmsh.model.occ.addPlaneSurface([cl1])
    cl2 = gmsh.model.occ.addCurveLoop([l3,l4,l5,-l8,-l7])
    surf2 = gmsh.model.occ.addPlaneSurface([cl2])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    if mode == "BC":
        bcs = [("top", l1), ("bottom", l4)]
        for name, line in bcs:
            pg = gmsh.model.addPhysicalGroup(1, [line])
            gmsh.model.setPhysicalName(1, pg, name)
        gmsh.model.addPhysicalGroup(1, [l5, l6], name="boundary_R")
        gmsh.model.addPhysicalGroup(1, [l2, l3], name="well")
      
    elif mode == "domain":
        gmsh.model.addPhysicalGroup(2, [surf1, surf2], name="surf")
        gmsh.model.addPhysicalGroup(1, [l7], name="fracture")
        
      
    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()

#---------------------------------------------------------------------------------
def create_rectangle_frac_mesh_v2(
    filepath: Path,
    radius: float,
    height: float,
    mesh_size: float,
    center_z: float = 0.0,
    r_well: float = 0.01,
    length: float = 8.0,
) -> None:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    
    gmsh.model.add(filepath.stem)
    
    z = center_z
    r_start = r_well
    r_end = radius + r_start

    p1 = gmsh.model.occ.addPoint(r_start, z + height / 2, 0.0, mesh_size) # Top-Well
    p2 = gmsh.model.occ.addPoint(r_end, z + height / 2, 0.0, mesh_size)   # Top-Right
    p3 = gmsh.model.occ.addPoint(r_end, z - height / 2, 0.0, mesh_size)   # Bot-Right
    p4 = gmsh.model.occ.addPoint(r_start, z - height / 2, 0.0, mesh_size) # Bot-Well
    
    p5 = gmsh.model.occ.addPoint(r_start, z, 0.0, 0.0)              # INTERSECTION
    p6 = gmsh.model.occ.addPoint(length, z, 0.0, mesh_size)               # Frac-Tip
    p7 = gmsh.model.occ.addPoint(r_end, z, 0.0, mesh_size)                # Far-Mid
    
    gmsh.model.occ.synchronize()
    
    l_well_top = gmsh.model.occ.addLine(p1, p5)
    l_well_bot = gmsh.model.occ.addLine(p5, p4)
    
    l_frac = gmsh.model.occ.addLine(p5, p6)
    
    l_top = gmsh.model.occ.addLine(p2, p1)
    l_bot = gmsh.model.occ.addLine(p4, p3)
    l_right_top = gmsh.model.occ.addLine(p7, p2)
    l_right_bot = gmsh.model.occ.addLine(p3, p7)
    l_connector = gmsh.model.occ.addLine(p6, p7)
    
    gmsh.model.occ.synchronize()
    
    cl1 = gmsh.model.occ.addCurveLoop([l_top, l_right_top, l_connector, l_frac, l_well_top])
    surf1 = gmsh.model.occ.addPlaneSurface([cl1])
    
    cl2 = gmsh.model.occ.addCurveLoop([-l_frac, -l_connector, l_right_bot, l_bot, l_well_bot])
    surf2 = gmsh.model.occ.addPlaneSurface([cl2])
    
    #-------------------------------
    all_entities = (gmsh.model.occ.getEntities(2) + 
                gmsh.model.occ.getEntities(1) + 
                gmsh.model.occ.getEntities(0))
    gmsh.model.occ.fragment(all_entities, []) 
    #------------------------------
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.model.addPhysicalGroup(2, [surf1, surf2], name="bulk_mesh")
    gmsh.model.addPhysicalGroup(1, [l_well_top, l_well_bot], name="well")
    gmsh.model.addPhysicalGroup(1, [l_frac], name="fracture")
    gmsh.model.addPhysicalGroup(1, [l_top], name="top")
    gmsh.model.addPhysicalGroup(1, [l_bot], name="bottom")
    gmsh.model.addPhysicalGroup(1, [l_right_top, l_right_bot], name="boundary_R")
    
    gmsh.model.addPhysicalGroup(0, [p5], name="intersection_point")
    gmsh.model.addPhysicalGroup(0, [p6], name="fracture_tip")
    
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()

