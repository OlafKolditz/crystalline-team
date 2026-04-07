import gmsh
import numpy as np
import sys
from pathlib import Path

import ogstools as ot
import pyvista as pv

class CubicDomainWithFault:
    def __init__(self, wide, height, thickness, z_center, aperture):
        """
        Initialize the cubic domain with fault subdomain.
        
        Parameters:
        wide: width of the domain in x-direction (m)
        height: height of the domain in y-direction (m)
        thickness: thickness of the domain in z-direction (m)
        z_center: z-coordinate of the domain center (m)
        aperture: aperture (thickness) of the fault subdomain in y-direction (m)
        """
        self.wide = wide
        self.height = height
        self.thickness = thickness
        self.z_center = z_center
        self.aperture = aperture
        
        # Calculate domain boundaries
        self.x_min = -wide/2
        self.x_max = wide/2
        self.y_min = -height/2
        self.y_max = height/2
        self.z_min = z_center - thickness/2
        self.z_max = z_center + thickness/2
        
        # Fault subdomain boundaries (vertical strip at back surface)
        self.fault_y_min = self.y_max - aperture
        self.fault_y_max = self.y_max
        
        # Vertical line parameters: starts from center of fault domain to top surface
        self.line_start_z = self.z_center  # Center of fault domain in z-direction
        self.line_end_z = self.z_max       # Top surface
        self.line_x = 0                    # Center in x-direction
        self.line_y = self.fault_y_max     # Back surface (fault surface)
        
        # Physical group tags
        self.rock_volume_tag = None
        self.fault_volume_tag = None
        self.surface_tags = {}
        self.line_tag = None
        
    def create_geometry(self):
        """Create the geometry with two subdomains and vertical line."""
        
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.model.add("cubic_domain_with_fault")
        
        # Create the full rock box
        rock_box = gmsh.model.occ.addBox(self.x_min, self.y_min, self.z_min,
                                         self.wide, self.height, self.thickness)
        
        # Create the fault box (at the back)
        fault_box = gmsh.model.occ.addBox(self.x_min, self.fault_y_min, self.z_min,
                                          self.wide, self.aperture, self.thickness)
        
        # First, fragment the two boxes to create the rock and fault volumes
        print("Creating rock and fault volumes...")
        gmsh.model.occ.fragment([(3, rock_box)], [(3, fault_box)])
        gmsh.model.occ.synchronize()
        
        # Get the volumes after fragmentation
        volumes = gmsh.model.getEntities(dim=3)
        
        # Identify which volume is rock and which is fault based on centroid y-coordinate
        rock_volumes = []
        fault_volumes = []
        
        for vol in volumes:
            com = gmsh.model.occ.getCenterOfMass(vol[0], vol[1])
            if com[1] < self.fault_y_min + self.aperture/2:
                rock_volumes.append(vol[1])
            else:
                fault_volumes.append(vol[1])
        
        # Store volume tags
        self.rock_volume_tag = rock_volumes[0] if rock_volumes else None
        self.fault_volume_tag = fault_volumes[0] if fault_volumes else None
        
        print(f"Rock volume tag: {self.rock_volume_tag}")
        print(f"Fault volume tag: {self.fault_volume_tag}")
        
        # Now create the vertical line
        print(f"\nCreating vertical line from z={self.line_start_z} to z={self.line_end_z}")
        
        # Create points for the line
        start_point = gmsh.model.occ.addPoint(self.line_x, self.line_y, self.line_start_z)
        end_point = gmsh.model.occ.addPoint(self.line_x, self.line_y, self.line_end_z)
        
        # Create the line
        line = gmsh.model.occ.addLine(start_point, end_point)
        gmsh.model.occ.synchronize()
        
        # Fragment the line with the volumes
        # This will split the line at volume boundaries
        print("Fragmenting line with volumes...")
        all_volumes = [(3, self.rock_volume_tag), (3, self.fault_volume_tag)]
        gmsh.model.occ.fragment(all_volumes, [(1, line)])
        gmsh.model.occ.synchronize()
        
        # Get all lines after fragmentation
        lines = gmsh.model.getEntities(dim=1)
        
        # Find the line segments that are inside the fault volume and on the back surface
        line_tags = []
        for line_entity in lines:
            # Get the bounding box of the line to check if it's on the back surface
            com = gmsh.model.occ.getCenterOfMass(line_entity[0], line_entity[1])
            # Check if the line is on the back surface (y = fault_y_max) and within z range
            if (abs(com[1] - self.fault_y_max) < 1e-6 and 
                self.line_start_z - 1e-6 <= com[2] <= self.line_end_z + 1e-6 and
                abs(com[0] - self.line_x) < 1e-6):
                # Verify this line is within the fault volume's x-range
                if self.x_min - 1e-6 <= com[0] <= self.x_max + 1e-6:
                    line_tags.append(line_entity[1])
        
        # Store line tags (only the vertical borehole line)
        if line_tags:
            self.line_tag = line_tags
            print(f"Found {len(line_tags)} line segment(s) for the vertical borehole")
        else:
            self.line_tag = [line]
            print("Using original line")
        
        # Get all surfaces for physical groups after all fragmentations
        surfaces = gmsh.model.getEntities(dim=2)
        
        # Categorize surfaces by their location and which volume they belong to
        for surf in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
            tol = 1e-6
            
            # Get the adjacent volumes to determine if this surface is on the boundary
            # For bottom surface (z = z_min)
            if abs(com[2] - self.z_min) < tol:
                if 'bottom' not in self.surface_tags:
                    self.surface_tags['bottom'] = []
                self.surface_tags['bottom'].append(surf[1])
            
            # For top surface (z = z_max)
            elif abs(com[2] - self.z_max) < tol:
                if 'top' not in self.surface_tags:
                    self.surface_tags['top'] = []
                self.surface_tags['top'].append(surf[1])
            
            # For left surface (x = x_min)
            elif abs(com[0] - self.x_min) < tol:
                if 'left' not in self.surface_tags:
                    self.surface_tags['left'] = []
                self.surface_tags['left'].append(surf[1])
            
            # For right surface (x = x_max)
            elif abs(com[0] - self.x_max) < tol:
                if 'right' not in self.surface_tags:
                    self.surface_tags['right'] = []
                self.surface_tags['right'].append(surf[1])
            
            # For front surface (y = y_min)
            elif abs(com[1] - self.y_min) < tol:
                if 'front' not in self.surface_tags:
                    self.surface_tags['front'] = []
                self.surface_tags['front'].append(surf[1])
            
            # For back surface (y = fault_y_max)
            elif abs(com[1] - self.fault_y_max) < tol:
                if 'back' not in self.surface_tags:
                    self.surface_tags['back'] = []
                self.surface_tags['back'].append(surf[1])
            
            # For interface between rock and fault (y = fault_y_min)
            #elif abs(com[1] - self.fault_y_min) < tol:
            #    if 'interface' not in self.surface_tags:
            #        self.surface_tags['interface'] = []
            #    self.surface_tags['interface'].append(surf[1])
        
        # Print surface statistics
        print("\nSurface groups found:")
        for key, tags in self.surface_tags.items():
            print(f"  {key}: {len(tags)} surface(s)")
        
        return True
    
    def assign_physical_groups(self):
        """Assign physical groups to volumes, surfaces, and line."""
        
        # Physical groups for volumes (subdomains)
        if self.rock_volume_tag:
            gmsh.model.addPhysicalGroup(3, [self.rock_volume_tag], 1)
            gmsh.model.setPhysicalName(3, 1, "Rock_mass")
            print(f"Assigned Rock_mass physical group to volume {self.rock_volume_tag}")
        
        if self.fault_volume_tag:
            gmsh.model.addPhysicalGroup(3, [self.fault_volume_tag], 2)
            gmsh.model.setPhysicalName(3, 2, "Fault_zone")
            print(f"Assigned Fault_zone physical group to volume {self.fault_volume_tag}")
        
        # Physical groups for surfaces
        surface_physical_ids = {
            'left': 10,
            'right': 11,
            'front': 12,
            'back': 13,
            'bottom': 14,
            'top': 15,
            #'interface': 16
        }
        
        for surface_name, tags in self.surface_tags.items():
            if tags:
                phys_id = surface_physical_ids[surface_name]
                gmsh.model.addPhysicalGroup(2, tags, phys_id)
                gmsh.model.setPhysicalName(2, phys_id, f"Surface_{surface_name}")
                print(f"Assigned Surface_{surface_name} physical group to {len(tags)} surface(s)")
        
        # Physical group for the vertical line (only the borehole line)
        if self.line_tag:
            # Filter out any line segments that are not the vertical borehole
            # The line should be at x=0, y=fault_y_max, and between line_start_z and line_end_z
            filtered_line_tags = []
            for line_tag in self.line_tag:
                # Get the line's points
                line_entity = (1, line_tag)
                com = gmsh.model.occ.getCenterOfMass(1, line_tag)
                # Check if this line is at the correct position
                if (abs(com[0] - self.line_x) < 1e-6 and 
                    abs(com[1] - self.fault_y_max) < 1e-6 and
                    self.line_start_z - 1e-6 <= com[2] <= self.line_end_z + 1e-6):
                    filtered_line_tags.append(line_tag)
            
            if filtered_line_tags:
                gmsh.model.addPhysicalGroup(1, filtered_line_tags, 20)
                gmsh.model.setPhysicalName(1, 20, "Vertical_line_in_fault")
                print(f"Assigned Vertical_line_in_fault physical group to {len(filtered_line_tags)} line segment(s)")
            else:
                print("Warning: No valid line segments found for Vertical_line_in_fault")
        
        # Also create physical groups for all surfaces combined (optional)
        #all_surface_tags = []
        #for tags in self.surface_tags.values():
        #    all_surface_tags.extend(tags)
        
        #if all_surface_tags:
        #    gmsh.model.addPhysicalGroup(2, all_surface_tags, 100)
        #    gmsh.model.setPhysicalName(2, 100, "All_boundaries")
    
    def generate_mesh(self, mesh_size_factor=1.0, optimize=True):
        """Generate the mesh with specified parameters."""
        
        # Set mesh options
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_size_factor)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        
        # Set mesh size - use a reasonable size for all elements
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5)
        
        # Generate 3D mesh
        print("\nGenerating 3D mesh...")
        gmsh.model.mesh.generate(3)
        
        # Check if mesh generation was successful
        element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
        total_elements = 0
        for i in range(len(element_types)):
            elem_tags = element_tags[i]
            total_elements += len(elem_tags)
        
        # Mesh statistics
        print("\nMesh Statistics:")
        nodes = gmsh.model.mesh.getNodes()
        print(f"  Number of nodes: {len(nodes[1])}")
        print(f"  Number of elements: {total_elements}")
        
        # Element type distribution
        if total_elements > 0:
            print("  Element type distribution:")
            for i in range(len(element_types)):
                elem_type = element_types[i]
                elem_tags = element_tags[i]
                if len(elem_tags) > 0:
                    try:
                        # Get element name
                        elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
                        print(f"    {elem_name} (type {elem_type}): {len(elem_tags)} elements")
                    except:
                        print(f"    Element type {elem_type}: {len(elem_tags)} elements")
        else:
            print("  WARNING: No elements were generated!")
        
        return total_elements > 0
    
    def visualize_mesh(self):
        """Launch Gmsh GUI to visualize the mesh."""
        # Show the mesh
        gmsh.fltk.run()
    
    def export_mesh(self, filename="cubic_domain_with_fault.msh", out_dir=""):
        """Export the mesh to file."""
        # Export in MSH format (version 4.1)
        gmsh.write(filename)
        print(f"\nMesh exported to {filename}")
        
        meshes = ot.meshes_from_gmsh(
            filename=str(filename), reindex=True, log=False
        )

        vtu_names = []
        for name, mesh in meshes.items():
            if name == "Rock_mass" or name == "Fault_zone":
                continue
            print(f"{name}: {mesh.n_cells} cells")
            if "physical_group_" in name:
                vtu_name = Path(
                    out_dir, f"{name.replace('physical_group_', '')}.vtu"
                )
            else:
                vtu_name = Path(out_dir, f"{name}.vtu")
            vtu_names.append(vtu_name)
            pv.save_meshio(vtu_name, mesh)
        
        
        ## Also export to VTK format for ParaView
        #vtk_filename = filename.replace('.msh', '.vtk')
        #gmsh.write(vtk_filename)
        #print(f"Mesh also exported to {vtk_filename} for ParaView")
    
    def get_mesh_info(self):
        """Print detailed information about the mesh and physical groups."""
        print("\n" + "="*60)
        print("MESH INFORMATION")
        print("="*60)
        
        # Get all physical groups
        print("\nPhysical Groups:")
        for dim in range(4):
            groups = gmsh.model.getPhysicalGroups(dim)
            if groups:
                print(f"\n  Dimension {dim}:")
                for group in groups:
                    name = gmsh.model.getPhysicalName(dim, group[1])
                    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, group[1])
                    print(f"    {group[1]}: {name} - {len(entities)} entities")
        
        # Get mesh quality statistics
        print("\nMesh Quality:")
        try:
            quality = gmsh.model.mesh.getQuality()
            if len(quality) >= 3:
                print(f"  Minimum element quality: {quality[0]:.4f}")
                print(f"  Maximum element quality: {quality[1]:.4f}")
                print(f"  Average element quality: {quality[2]:.4f}")
            else:
                print("  Quality statistics not fully available")
        except Exception as e:
            print(f"  Quality statistics not available: {e}")
        print("="*60)
    
    def cleanup(self):
        """Clean up Gmsh."""
        gmsh.finalize()

def main():
    # Define domain parameters
    wide = 50.0      # Width in x-direction (m)
    height = 50.0     # Height in y-direction (m)
    thickness = 50.0  # Thickness in z-direction (m)
    z_center = -600   # Z-coordinate of domain center (m)
    aperture = 0.5   # Aperture (thickness) of fault subdomain (m)
    
    # Create the cubic domain with fault
    domain = CubicDomainWithFault(wide, height, thickness, z_center, aperture)
    
    # Print domain information
    print("="*60)
    print("CUBIC DOMAIN WITH FAULT SUBDOMAIN")
    print("="*60)
    print(f"Domain dimensions:")
    print(f"  Width (X): {wide} m")
    print(f"  Height (Y): {height} m")
    print(f"  Thickness (Z): {thickness} m")
    print(f"  Z-center: {z_center} m")
    print(f"\nFault subdomain (vertical strip at back):")
    print(f"  Aperture (thickness in Y): {aperture} m")
    print(f"  X-range: [{domain.x_min:.2f}, {domain.x_max:.2f}] m")
    print(f"  Y-range: [{domain.fault_y_min:.2f}, {domain.fault_y_max:.2f}] m")
    print(f"  Z-range: [{domain.z_min:.2f}, {domain.z_max:.2f}] m")
    print(f"\nVertical line inside fault:")
    print(f"  Position: X = 0 m, Y = {domain.fault_y_max:.2f} m")
    print(f"  Starts at Z = {domain.line_start_z:.2f} m (center of fault domain)")
    print(f"  Ends at Z = {domain.line_end_z:.2f} m (top surface)")
    print("="*60)
    
    try:
        # Create geometry
        print("\nCreating geometry...")
        domain.create_geometry()
        
        # Assign physical groups
        print("\nAssigning physical groups...")
        domain.assign_physical_groups()
        
        # Generate mesh
        print("\nGenerating mesh...")
        success = domain.generate_mesh(mesh_size_factor=0.8, optimize=True)
        
        if success:
            # Get mesh information
            domain.get_mesh_info()
            
            # Export mesh
            domain.export_mesh("cubic_domain_with_fault.msh")
            
            # Ask user if they want to visualize
            response = input("\nDo you want to visualize the mesh? (y/n): ").lower()
            if response == 'y':
                print("Launching Gmsh GUI for visualization...")
                print("Close the Gmsh window to continue.")
                domain.visualize_mesh()
        else:
            print("\nERROR: Mesh generation failed. No elements were created.")
            print("Please check the geometry and try again.")
        
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        domain.cleanup()

if __name__ == "__main__":
    main()
