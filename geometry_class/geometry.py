import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pygimli as pg
import pygimli.physics.ert as ert
import warnings
import pyvista as pv
from pygimli.viewer.pv import drawSlice

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
    area_xy: tuple = (20.0, 20.0)
    world_xy: tuple = (100.0, 100.0)
    world_area: float = 0.0
    core_area: float = None
    borehole_area: float = None
    long_spacing: float = 1.0
    

    def __post_init__(self):
        if self.layer_1d_geometry is not None and len(self.layer_1d_geometry) > 0:
            if self.layer_1d_geometry[0, 0] != 0:
                self.layer_1d_geometry[0, 0] = 0.0
                warnings.warn("The top value of the first layer did not start from 0. It has been forced to 0.")
            
            if self.layer_1d_geometry[-1, 1] > self.borehole_length:
                self.borehole_length = float(self.layer_1d_geometry[-1, 1])
                warnings.warn(f"The bottom value of the last layer is larger than the borehole length. borehole_length has been updated to {self.borehole_length}.")

        if self.core_area is None:
            self.core_area = self.borehole_diameter / 10.0
        
        if self.borehole_area is None:
            self.borehole_area = self.core_area / 5.0


    def make_basic_geometry(self, vtk_filename: str = "plc.vtk") -> pg.core.Mesh:
        """
        Create a basic 3D geometry representation of the borehole and layered domain.
        """
        if self.layer_1d_geometry is None or len(self.layer_1d_geometry) == 0:
            raise ValueError("layer_1d_geometry must be defined to create the geometry.")

        x_dim, y_dim = self.area_xy
        wx_dim, wy_dim = self.world_xy if self.world_xy else (x_dim, y_dim)

        last_layer_boundary = float(self.layer_1d_geometry[-1, 1])
        core_depth = last_layer_boundary + 10.0 * self.long_spacing

        # 1. Create the outer boundary domain (world)
        world_depth = max(self.borehole_length * 2.5, core_depth * 1.5)
        world = pg.meshtools.createCube(
            size=[wx_dim, wy_dim, world_depth],
            pos=[0.0, 0.0, -world_depth / 2.0],
            marker=100,
            area=self.world_area
        )
        
        for m in world.regionMarkers():
            m.setPos([wx_dim / 2.0 - 0.1, 0.011, -world_depth / 2.0 + 0.0123])

        # 2. Create the inner core block for fine resolution
        core = pg.meshtools.createCube(
            size=[x_dim, y_dim, core_depth],
            pos=[0.0, 0.0, -core_depth / 2.0],
            marker=2,
            area=self.core_area
        )
        
        for m in core.regionMarkers():
            m.setPos([x_dim / 2.0 - 0.1, 0.011, -core_depth / 2.0 + 0.0123])

        # 3. Create the borehole cylinder
        borehole = pg.meshtools.createCylinder(
            radius=self.borehole_diameter / 2.0,
            height=self.borehole_length,
            pos=[0.0, 0.0, -self.borehole_length / 2.0],
            marker=1,
            area=self.borehole_area
        )
        
        for m in borehole.regionMarkers():
            m.setPos([0.007, 0.011, -self.borehole_length / 2.0 + 0.0123])
            

        geom = world + core + borehole

        # In PyGIMLi, combining geometries with '+' can discard the area constraints 
        # on the region markers. We re-apply them directly to the final PLC markers.
        for m in geom.regionMarkers():
            if m.marker() == 100:
                m.setArea(self.world_area)
            elif m.marker() == 2:
                m.setArea(self.core_area)
            elif m.marker() == 1:
                m.setArea(self.borehole_area)


        #pg.show(geom)

        geom.exportVTK('before_boundaries.vtk')
        # Set strict boundary conditions for ERT and avoid conflicts with region markers.
        # createCube assigns boundary markers 1-6 which conflict with our region markers 1, 2, 3...
        for bound in geom.boundaries():
            center = bound.center()
            # Top surface must be -1 (Free surface / Neumann)
            if np.isclose(center[2], 0.0, atol=1e-3):
                bound.setMarker(-1)
            # Outer walls and bottom must be -2 (Mixed / Robin boundary)
            elif (np.isclose(abs(center[0]), wx_dim / 2.0, atol=1e-3) or
                  np.isclose(abs(center[1]), wy_dim / 2.0, atol=1e-3) or
                  np.isclose(center[2], -world_depth, atol=1e-3)):
                bound.setMarker(-2)
            else:
                # Internal boundaries should be 0 so they are not treated as Dirichlet BCs
                bound.setMarker(0)
        #pg.show(geom)
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
            
        num_electrodes = int(round(self.borehole_length / measuring_spacing)) + 1
        z_positions = -np.arange(num_electrodes) * measuring_spacing
            
        # Inject the electrode nodes structurally into the PLC before meshing.
        # Connect them with edges to form a constrained 1D line inside the borehole.
        # This mathematically forces TetGen to preserve every single electrode exactly as a mesh node.
        nodes = []
        for z in z_positions:
            node = geom.createNode([0.0, 0.0, z])
            node.setMarker(99)
            nodes.append(node)
            
        for i in range(len(nodes) - 1):
            geom.createEdge(nodes[i], nodes[i+1])

        # Create the ERT DataContainer
        data = pg.DataContainerERT()
        data.setSensorPositions([[0.0, 0.0, z] for z in z_positions])
            
        # Map A, B, M, N to their relative index positions (0-based)
        a_pos, b_pos, m_pos, n_pos = [idx - 1 for idx in abmn_order]
        
        # Use only the short and long spacings
        a_spacings = [short_spacing, long_spacing]
        
        a_idx, b_idx, m_idx, n_idx = [], [], [], []
        
        for a in a_spacings:
            # Number of index steps between adjacent logical array positions
            a_steps = int(round(a / measuring_spacing))
            if a_steps == 0:
                continue
                
            max_rel_idx = max(a_pos, b_pos, m_pos, n_pos) * a_steps
            
            # Move the array from top to bottom
            for start_idx in range(len(z_positions) - max_rel_idx):
                a_idx.append(int(start_idx + a_pos * a_steps))
                b_idx.append(int(start_idx + b_pos * a_steps))
                m_idx.append(int(start_idx + m_pos * a_steps))
                n_idx.append(int(start_idx + n_pos * a_steps))
                
        # Register standard arrays for pyGIMLi
        data.resize(len(a_idx))
        data.set("a", a_idx)
        data.set("b", b_idx)
        data.set("m", m_idx)
        data.set("n", n_idx)
        data.set("valid", np.ones(len(a_idx), dtype=int))
                
        if protocol_filename:
            data.save(protocol_filename)
            
        return data

    def make_mesh(self, geom: pg.core.Mesh, vtk_filename: str = "mesh.vtk", quality: float = 34.2) -> pg.core.Mesh:
        """
        Generate a mesh from the provided geometry, paint layers, and create rhomap.
        
        Args:
            geom (pg.core.Mesh): The geometry to mesh.
            vtk_filename (str, optional): The filename for exporting the mesh to VTK format. Defaults to 'mesh.vtk'.
            quality (float, optional): The mesh quality parameter (higher means finer/more regular elements). Defaults to 34.2.
        """
        # Generate the mesh (omit global area constraint to respect region areas)
        mesh = pg.meshtools.createMesh(geom, quality=quality)

        # 4. Paint the layers onto the soil cells
        for cell in mesh.cells():
            z_center = cell.center()[2]
            if cell.marker() in [2, 100]:  # Update both inner core and outer world cells to form true 1D layers
                for i, row in enumerate(self.layer_1d_geometry):
                    depth_top = float(row[0])
                    depth_bottom = float(row[1])
                    if i == len(self.layer_1d_geometry) - 1:
                        depth_bottom = 99999.0  # Extend the last layer downwards to the bottom of the world domain
                    
                    z_top = -depth_top
                    z_bottom = -depth_bottom
                    
                    if z_bottom - 0.1 <= z_center <= z_top + 0.1:
                        cell.setMarker(i + 2)
                        break

        # 5. Create rhomap according to PyGIMLi tutorial standard
        # A list of [marker, resistivity] pairs. Marker 1 (Borehole) defaults to background_resistivity.
        rhomap = [[1, self.background_resistivity]]
        
        for i, row in enumerate(self.layer_1d_geometry):
            res_value = float(row[2]) if len(row) > 2 else np.nan
            rhomap.append([i + 2, res_value])

        # Store the rhomap on the instance for future ERT simulation
        self.rhomap = rhomap

        # Map the rhomap to the mesh cells to create the Resistivity array for VTK export
        rhomap_dict = dict(rhomap)
        mesh["Resistivity"] = np.array([rhomap_dict.get(cell.marker(), np.nan) for cell in mesh.cells()])

        if vtk_filename:
            mesh.exportVTK(vtk_filename)
            mesh.save("mesh.bms")  # Save the mesh in binary format for future use
        
        return mesh

    def plot_boundary_conditions(self, mesh: pg.core.Mesh, show_internal: bool = True):
        """
        Plot the boundary conditions (markers) of the mesh using PyVista.
        
        Args:
            mesh (pg.core.Mesh): The mesh to extract boundaries from.
            show_internal (bool): If False, hides internal boundaries (marker == 0).
        """
        points = []
        faces = []
        markers = []

        node_map = {}
        node_idx = 0

        for bound in mesh.boundaries():
            marker = bound.marker()
            if not show_internal and marker == 0:
                continue
                
            b_node_ids = []
            for i in range(bound.nodeCount()):
                n = bound.node(i)
                if n.id() not in node_map:
                    node_map[n.id()] = node_idx
                    pos = n.pos()
                    points.append([pos[0], pos[1], pos[2]])
                    node_idx += 1
                b_node_ids.append(node_map[n.id()])
            faces.append([bound.nodeCount()] + b_node_ids)
            markers.append(marker)

        surf = pv.PolyData(np.array(points), np.hstack(faces))
        surf.cell_data["Boundary Markers"] = np.array(markers)

        pl = pv.Plotter()
        pl.add_mesh(surf, scalars="Boundary Markers", cmap="Set1", show_edges=True, opacity=0.5)
        pl.add_axes()
        pl.add_title("Mesh Boundary Conditions")
        pl.show()

    def run(self, mesh: pg.core.Mesh, data: pg.DataContainerERT, rhomap: list = None, output_filename: str = "simulated_data.dat", noise_level: float = 0.0, noise_abs: float = 0.0) -> pg.DataContainerERT:
        """
        Run the ERT forward model using the generated mesh and data protocol.
        
        Args:
            mesh (pg.core.Mesh): The generated mesh.
            data (pg.DataContainerERT): The measurement protocol.
            rhomap (list, optional): Custom rhomap. If provided, it updates the class instance's rhomap. Defaults to None.
            output_filename (str, optional): The filename for exporting the simulated data. Defaults to 'simulated_data.dat'.
            noise_level (float, optional): Relative noise level to add to the data (e.g. 0.05 for 5%). Defaults to 0.0.
            noise_abs (float, optional): Absolute noise level in Ohms. Defaults to 0.0.
            
        Returns:
            pg.DataContainerERT: The simulated data container.
        """
        if rhomap is not None:
            self.rhomap = rhomap

        if self.rhomap is None:
            raise ValueError("rhomap is not defined. Please run make_mesh first to generate the rhomap or provide it as an argument.")
            
        # Map the rhomap to the mesh cells
        rhomap_dict = dict(self.rhomap)
        res_list = [rhomap_dict.get(cell.marker(), self.background_resistivity) for cell in mesh.cells()]
        
        # Assign to mesh for VTK visualization
        #mesh["Resistivity"] = np.array(res_list)
        
        # Convert explicitly to pyGIMLi Vector to prevent internal region mapping crashes
        res_vec = pg.Vector(res_list)
        print(self.rhomap)
        #pg.show(mesh,data=self.rhomap,cMap="jet",label="Resistivity [Ohm.m]",logScale=True,sliceBy='x')
        #pl = pv.Plotter()
        # 2. Add your slice along the x-axis
        #drawSlice(
        #    pl, 
        #    mesh, 
        #    normal=[1, 0, 0],   # Normal vector pointing along x-axis creates a Y-Z cutting plane
        #    origin=[0, 0, 0], # The exact X coordinate where you want to slice
        #    data=np.array(res_list), 
        #    cMap="jet", 
        #    label="Resistivity [Ohm.m]", 
        #    logScale=True
        #)
        #pl.show() 
        
        print("Simulating ERT data...")
        sim_data = ert.simulate(
            mesh=mesh,
            scheme=data,
            res=res_vec,
            background=self.background_resistivity,
            noiseLevel=noise_level,
            noiseAbs=noise_abs,
            sr=True  # Use Singularity Removal for accurate primary potential calculation
        )
        
        if output_filename:
            sim_data.save(output_filename)
            
        return sim_data

    def plot_model_and_data(self, data: pg.DataContainerERT):
        """
        Plots a 2D slice of the 1D resistivity model and borehole alongside a 
        depth profile of the short and long Wenner measurements.
        
        Args:
            data (pg.DataContainerERT): The ERT data container to plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        
        # --- Plot 1: 2D Slice of 1D model & Borehole ---
        width = self.area_xy[0]
        
        last_layer_boundary = float(self.layer_1d_geometry[-1, 1]) if self.layer_1d_geometry is not None else self.borehole_length
        core_depth = last_layer_boundary + 10.0 * self.long_spacing
        
        # Determine color scaling bounds from layers and background
        res_values = [self.background_resistivity]
        if self.layer_1d_geometry is not None:
            res_values.extend(self.layer_1d_geometry[:, 2])
        
        min_res, max_res = min(res_values), max(res_values)
        if min_res == max_res:
            min_res, max_res = min_res * 0.5, max_res * 2.0
            
        norm = mcolors.LogNorm(vmin=min_res, vmax=max_res)
        cmap = cm.jet
        
        # Draw soil layers as horizontal rectangles
        if self.layer_1d_geometry is not None:
            for i, row in enumerate(self.layer_1d_geometry):
                top, bottom, res = float(row[0]), float(row[1]), float(row[2])
                if i == len(self.layer_1d_geometry) - 1:
                    #bottom = max(bottom, core_depth)
                    bottom = max(bottom, 1.1*self.borehole_length)
                # y_start is the bottom coordinate (negative depths)
                rect = patches.Rectangle((-width/2, -bottom), width, bottom - top, 
                                         facecolor=cmap(norm(res)), edgecolor='k')
                ax1.add_patch(rect)
        else:
            #rect = patches.Rectangle((-width/2, -core_depth), width, core_depth, 
            #                         facecolor=cmap(norm(self.background_resistivity)), edgecolor='k')
            rect = patches.Rectangle((-width/2, -1.1*self.borehole_length), width, 1.1*self.borehole_length, 
                                     facecolor=cmap(norm(self.background_resistivity)), edgecolor='k')           
            ax1.add_patch(rect)
            
        # Draw borehole rectangle in the center
        bh_res = self.background_resistivity
        if self.rhomap is not None:
            bh_res = next((r[1] for r in self.rhomap if r[0] == 1), bh_res)
            
        bh_width = self.borehole_diameter
        bh_rect = patches.Rectangle((-bh_width/2, -self.borehole_length), bh_width, self.borehole_length,
                                    facecolor=cmap(norm(bh_res)), edgecolor='white', hatch='///')
        ax1.add_patch(bh_rect)
        
        ax1.set_xlim(-width/2, width/2)
        #ax1.set_ylim(-core_depth, 0)
        ax1.set_ylim(-self.borehole_length * 1.05, 0)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("Depth (m)")
        ax1.set_title("1D Model & Borehole Slice")
        
        # Setup colorbar based on resistivity values
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.05, pad=0.05)
        cbar.set_label("Resistivity ($\Omega \cdot$m)")

        # --- Plot 2: Depth Profile of Wenner Measures ---
        # Calculate depth (z_center) and the 'a' spacing dynamically for each measurement quadrupole
        z_A = np.array([data.sensorPosition(int(i))[2] for i in data("a")])
        z_B = np.array([data.sensorPosition(int(i))[2] for i in data("b")])
        z_M = np.array([data.sensorPosition(int(i))[2] for i in data("m")])
        z_N = np.array([data.sensorPosition(int(i))[2] for i in data("n")])
        
        z_center = (z_A + z_B + z_M + z_N) / 4.0
        
        # In a generic collinear setup, sorting the coordinates easily identifies adjacent electrode spacing 'a'
        a_spacings = [abs(sorted([z_A[i], z_B[i], z_M[i], z_N[i]])[1] - sorted([z_A[i], z_B[i], z_M[i], z_N[i]])[0]) for i in range(data.size())]
        a_spacings = np.round(a_spacings, decimals=2)
        
        rho_a = np.array(data("rhoa")) if "rhoa" in data.tokenList() else np.array(data("r") * data("k"))
        unique_a = np.unique(a_spacings)
        
        if len(unique_a) > 0:
            mask_short = (a_spacings == unique_a[0])
            ax2.plot(rho_a[mask_short], z_center[mask_short], 'o-', label=f"Short (a={unique_a[0]}m)", color='blue')
            
        if len(unique_a) > 1:
            mask_long = (a_spacings == unique_a[-1])
            ax2.plot(rho_a[mask_long], z_center[mask_long], 's-', label=f"Long (a={unique_a[-1]}m)", color='red')
            
        ax2.set_xlabel("Apparent Resistivity ($\Omega \cdot$m)")
        ax2.set_title("Apparent Resistivity vs Depth")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Create a dummy layer configuration: [top, bottom, resistivity]
    layers = np.array([
        [0.0, 5.0, 100.0],
        [5.0, 10.0, 50.0],
        [10.0, 15.0, 200.0]
    ])
    geometry = Geometry(borehole_length=15.0, borehole_diameter=0.2, layer_1d_geometry=layers, background_resistivity=5.0, long_spacing=1.0)
    geom = geometry.make_basic_geometry()
    
    # Generate the electrode array and protocol prior to meshing
    data = geometry.make_array(
        geom=geom, 
        abmn_order=(1, 4, 2, 3), 
        short_spacing=0.25, 
        long_spacing=geometry.long_spacing, 
        measuring_spacing=0.2   
    )
    print(f"ERT Data generated with {data.size()} measurements.")
    
    mesh = geometry.make_mesh(geom)
    print(f"Mesh created: {mesh}")
    print(f"Rhomap generated: {geometry.rhomap}")
    
    # Visualize the boundary conditions (you can set show_internal=False to hide region boundaries)
    #geometry.plot_boundary_conditions(mesh, show_internal=True)

    # Run the forward simulation
    simulated_data = geometry.run(mesh, data)
    print(f"Forward simulation complete. Simulated data saved to 'simulated_data.dat' with {simulated_data.size()} measurements.")
    # pg.show(mesh) # Uncomment to visualize using PyGIMLi

    # Plot the 2D slice of the 1D model and the depth profiles of the simulated data
    geometry.plot_model_and_data(simulated_data)
