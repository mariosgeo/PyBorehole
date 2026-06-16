import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.physics.ert as ert
import warnings


@dataclass
class Geometry:
    """
    A utility class for performing common geometric calculations.

    Attributes:
        borehole_length (float): The total depth/length of the borehole.
        borehole_diameter (float): The diameter of the borehole.
        layer_1d_geometry (np.ndarray, optional): A 2D array defining the layered soil geometry, typically [top_depth, bottom_depth, resistivity] for each layer.
        background_resistivity (float, optional): The default resistivity applied to the background/unspecified areas. Defaults to 5.0.
        rhomap (list, optional): A list of [marker, resistivity] pairs generated after geometry creation to map regions to resistivity.
        area_xy (tuple, optional): The (x, y) dimensions of the inner core modeling domain. Defaults to (10.0, 10.0).
        world_xy (tuple, optional): The (x, y) dimensions of the outer boundary domain. Defaults to (50.0, 50.0).
        world_area (float, optional): The maximum area/volume constraint for tetrahedra in the outer domain. Defaults to 0.0 (unconstrained).
        core_area (float, optional): The maximum area/volume constraint for tetrahedra in the core domain. Defaults to None (calculated from borehole_diameter).
        borehole_area (float, optional): The maximum area/volume constraint for tetrahedra in the borehole. Defaults to None (calculated from core_area).
    """
    borehole_length: float 
    borehole_diameter: float
    layer_1d_geometry: np.ndarray = None
    background_resistivity: float = 5.0
    rhomap: list = None
    area_xy: tuple = (10.0, 10.0)
    world_xy: tuple = (50.0, 50.0)
    world_area: float = 0.0
    core_area: float = None
    borehole_area: float = None
    

    def __post_init__(self):
        if self.layer_1d_geometry is not None and len(self.layer_1d_geometry) > 0:
            if self.layer_1d_geometry[0, 0] != 0:
                self.layer_1d_geometry[0, 0] = 0.0
                warnings.warn("The top value of the first layer did not start from 0. It has been forced to 0.")
            
            if self.layer_1d_geometry[-1, 1] > self.borehole_length:
                self.borehole_length = float(self.layer_1d_geometry[-1, 1])
                warnings.warn(f"The bottom value of the last layer is larger than the borehole length. borehole_length has been updated to {self.borehole_length}.")

        if self.core_area is None:
            self.core_area = self.borehole_diameter / 2.0
        
        if self.borehole_area is None:
            self.borehole_area = self.core_area / 2.0


    def make_basic_geometry(self, vtk_filename: str = "plc.vtk") -> pg.core.Mesh:
        """
        Create a basic 3D geometry representation of the borehole and layered domain.
        """
        if self.layer_1d_geometry is None or len(self.layer_1d_geometry) == 0:
            raise ValueError("layer_1d_geometry must be defined to create the geometry.")

        x_dim, y_dim = self.area_xy
        wx_dim, wy_dim = self.world_xy if self.world_xy else (x_dim, y_dim)

        # 1. Create the outer boundary domain (world)
        # We use marker=-1 for the boundary soil, allowing it to grow (area=0)
        # We stagger the heights (0.5, 0.3, 0.1) to avoid intersecting coplanar facets for robust meshing.
        world_depth = self.borehole_length * 2.5
        world = pg.meshtools.createCube(
            size=[wx_dim, wy_dim, world_depth + 0.5],
            pos=[0.0, 0.0, -world_depth / 2.0 + 0.25],
            marker=-1,
            area=self.world_area
        )
        
        # Shift the world region marker to be safely outside the core domain
        # Add a small offset to Y and Z to avoid aligning with symmetry planes or layer interfaces
        safe_wx = x_dim / 2.0 + (wx_dim / 2.0 - x_dim / 2.0) / 2.0
        for m in world.regionMarkers():
            m.setPos([safe_wx, 0.011, -world_depth / 2.0 + 0.0123])

        # 2. Create the inner core block for fine resolution
        core = pg.meshtools.createCube(
            size=[x_dim, y_dim, self.borehole_length + 0.3],
            pos=[0.0, 0.0, -self.borehole_length / 2.0],
            marker=2,
            area=self.core_area
        )
        
        # Shift the core region marker to be safely outside the borehole cylinder
        # Add a small offset to Y and Z to avoid aligning with symmetry planes or layer interfaces
        safe_x = self.borehole_diameter / 2.0 + (x_dim / 2.0 - self.borehole_diameter / 2.0) / 2.0
        for m in core.regionMarkers():
            m.setPos([safe_x, 0.011, -self.borehole_length / 2.0 + 0.0123])

        # 3. Create the borehole cylinder
        borehole = pg.meshtools.createCylinder(
            radius=self.borehole_diameter / 2.0,
            height=self.borehole_length + 0.1,
            pos=[0.0, 0.0, -self.borehole_length / 2.0],
            marker=1,
            area=self.borehole_area
        )
        
        # Shift the borehole region marker to ensure it doesn't fall exactly on layer interfaces
        for m in borehole.regionMarkers():
            m.setPos([0.007, 0.011, -self.borehole_length / 2.0 + 0.0123])

        geom = world + core + borehole

        # In PyGIMLi, combining geometries with '+' can discard the area constraints 
        # on the region markers. We re-apply them directly to the final PLC markers.
        for m in geom.regionMarkers():
            if m.marker() == -1:
                m.setArea(self.world_area)
            elif m.marker() == 2:
                m.setArea(self.core_area)
            elif m.marker() == 1:
                m.setArea(self.borehole_area)

        if vtk_filename:
            geom.exportVTK(vtk_filename)

        return geom

    def make_array(
        self, 
        geom: pg.core.Mesh, 
        abmn_order: tuple, 
        short_spacing: float, 
        long_spacing: float, 
        measuring_spacing: float, 
        protocol_filename: str = "protocol.dat"
    ) -> pg.DataContainerERT:
        """
        Generate an electrode array, move it along the borehole, create a protocol, and add nodes to the PLC.
        
        Args:
            geom (pg.core.Mesh): The PLC geometry where electrode nodes will be added.
            abmn_order (tuple): The order of A, B, M, N electrodes (e.g., (1, 4, 2, 3) for Wenner).
            short_spacing (float): The minimum 'a' spacing for the array.
            long_spacing (float): The maximum 'a' spacing for the array.
            measuring_spacing (float): The step size to move the array down the borehole.
            protocol_filename (str): Output filename for the protocol.
            
        Returns:
            pg.DataContainerERT: The generated data container with the protocol.
        """
        if not (long_spacing > short_spacing):
            raise ValueError("long_spacing must be greater than short_spacing")
            
        # Generate all unique electrode positions along the borehole
        num_electrodes = int(round(self.borehole_length / measuring_spacing)) + 1
        z_positions = -np.arange(num_electrodes) * measuring_spacing
        
        # Add electrodes as nodes to the PLC (marker 99 is standard for electrodes)
        for z in z_positions:
            node = geom.createNode([0.0, 0.0, z])
            node.setMarker(99)
            
        # Create the ERT DataContainer
        data = pg.DataContainerERT()
        for z in z_positions:
            data.createSensor([0.0, 0.0, z])
            
        # Map A, B, M, N to their relative index positions (0-based)
        a_pos, b_pos, m_pos, n_pos = [idx - 1 for idx in abmn_order]
        
        # Use only the short and long spacings
        a_spacings = [short_spacing, long_spacing]
        
        # Calculate total measurements to correctly resize the container
        total_measurements = 0
        for a in a_spacings:
            a_steps = int(round(a / measuring_spacing))
            if a_steps == 0:
                continue
            max_rel_idx = max(a_pos, b_pos, m_pos, n_pos) * a_steps
            total_measurements += max(0, len(z_positions) - max_rel_idx)
            
        data.resize(total_measurements)
        
        data_index = 0
        for a in a_spacings:
            # Number of index steps between adjacent logical array positions
            a_steps = int(round(a / measuring_spacing))
            if a_steps == 0:
                continue
                
            max_rel_idx = max(a_pos, b_pos, m_pos, n_pos) * a_steps
            
            # Move the array from top to bottom
            for start_idx in range(len(z_positions) - max_rel_idx):
                data.createFourPointData(
                    data_index,
                    int(start_idx + a_pos * a_steps),
                    int(start_idx + b_pos * a_steps),
                    int(start_idx + m_pos * a_steps),
                    int(start_idx + n_pos * a_steps)
                )
                data_index += 1
                
        if protocol_filename:
            data.save(protocol_filename)
            
        return data

    def make_mesh(self, geom: pg.core.Mesh, vtk_filename: str = "mesh.vtk") -> pg.core.Mesh:
        """
        Generate a mesh from the provided geometry, paint layers, and create rhomap.
        
        Args:
            geom (pg.core.Mesh): The geometry to mesh.
            vtk_filename (str, optional): The filename for exporting the mesh to VTK format. Defaults to 'mesh.vtk'.
        """
        # Generate the mesh (omit global area constraint to respect region areas)
        mesh = pg.meshtools.createMesh(geom)

        # 4. Paint the layers onto the soil cells
        for cell in mesh.cells():
            if cell.marker() == 2:  # Only update the soil cells
                z_center = cell.center()[2]
                for i, row in enumerate(self.layer_1d_geometry):
                    depth_top = float(row[0])
                    depth_bottom = float(row[1])
                    if i == len(self.layer_1d_geometry) - 1:
                        depth_bottom = max(depth_bottom, self.borehole_length)
                    
                    z_top = -depth_top
                    z_bottom = -depth_bottom
                    
                    # Increased epsilon to 0.15 to safely catch the vertically expanded core cells
                    if z_bottom - 0.15 <= z_center <= z_top + 0.15:
                        cell.setMarker(i + 2)
                        break

        # 5. Create rhomap according to PyGIMLi tutorial standard
        # A list of [marker, resistivity] pairs. Marker 1 (Borehole) defaults to background_resistivity.
        rhomap = [[1, self.background_resistivity]]
        
        total_thickness = 0.0
        sum_res_thickness = 0.0

        for i, row in enumerate(self.layer_1d_geometry):
            res_value = float(row[2]) if len(row) > 2 else np.nan
            rhomap.append([i + 2, res_value])
            
            if not np.isnan(res_value):
                thickness = float(row[1]) - float(row[0])
                total_thickness += thickness
                sum_res_thickness += res_value * thickness
                
        avg_resistivity = sum_res_thickness / total_thickness if total_thickness > 0 else self.background_resistivity
        rhomap.append([-1, avg_resistivity])

        # Store the rhomap on the instance for future ERT simulation
        self.rhomap = rhomap

        # Map the rhomap to the mesh cells to create the Resistivity array for VTK export
        rhomap_dict = dict(rhomap)
        mesh["Resistivity"] = np.array([rhomap_dict.get(cell.marker(), np.nan) for cell in mesh.cells()])

        if vtk_filename:
            mesh.exportVTK(vtk_filename)

        return mesh

    def run(self, mesh: pg.core.Mesh, data: pg.DataContainerERT, output_filename: str = "simulated_data.dat", noise_level: float = 0.0, noise_abs: float = 0.0) -> pg.DataContainerERT:
        """
        Run the ERT forward model using the generated mesh and data protocol.
        
        Args:
            mesh (pg.core.Mesh): The generated mesh.
            data (pg.DataContainerERT): The measurement protocol.
            output_filename (str, optional): The filename for exporting the simulated data. Defaults to 'simulated_data.dat'.
            noise_level (float, optional): Relative noise level to add to the data (e.g. 0.05 for 5%). Defaults to 0.0.
            noise_abs (float, optional): Absolute noise level in Ohms. Defaults to 0.0.
            
        Returns:
            pg.DataContainerERT: The simulated data container.
        """
        if self.rhomap is None:
            raise ValueError("rhomap is not defined. Please run make_mesh first to generate the rhomap.")
            
        sim_data = ert.simulate(
            mesh=mesh,
            scheme=data,
            res=self.rhomap,
            noiseLevel=noise_level,
            noiseAbs=noise_abs
        )
        
        if output_filename:
            sim_data.save(output_filename)
            
        return sim_data


# Example usage:
if __name__ == "__main__":
    # Create a dummy layer configuration: [top, bottom, resistivity]
    layers = np.array([
        [0.0, 5.0, 100.0],
        [5.0, 10.0, 50.0],
        [10.0, 15.0, 200.0]
    ])
    geometry = Geometry(borehole_length=15.0, borehole_diameter=0.2, layer_1d_geometry=layers)
    geom = geometry.make_basic_geometry()
    
    # Generate the electrode array and protocol prior to meshing
    data = geometry.make_array(
        geom=geom, 
        abmn_order=(1, 4, 2, 3), 
        short_spacing=1.0, 
        long_spacing=3.0, 
        measuring_spacing=0.5
    )
    print(f"ERT Data generated with {data.size()} measurements.")
    
    mesh = geometry.make_mesh(geom)
    print(f"Mesh created: {mesh}")
    print(f"Rhomap generated: {geometry.rhomap}")
    
    # Run the forward simulation
    simulated_data = geometry.run(mesh, data)
    print(f"Forward simulation complete. Simulated data saved to 'simulated_data.dat' with {simulated_data.size()} measurements.")
    # pg.show(mesh) # Uncomment to visualize using PyGIMLi
