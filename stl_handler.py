import numpy as np
import trimesh
from PyQt5.QtWidgets import QFileDialog, QTreeWidgetItem, QMenu, QInputDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QSpinBox, QDialogButtonBox, QMessageBox, QDoubleSpinBox, QCheckBox, QGroupBox, QWidget, QLineEdit, QColorDialog
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QCursor, QDoubleValidator
import os
import qtawesome as qta
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time
import copy
from scipy.spatial.transform import Rotation as R_sps
import sys
import traceback
import ctypes

from pathlib import Path
from material_properties_dialog import MaterialPropertiesDialog

# --- PATCH: Add numpy-stl import for robust STL loading ---
try:
    from stl import mesh as numpy_stl_mesh
except ImportError:
    numpy_stl_mesh = None

# --- ADDED: Imports for DAE/Texture handling with bundled Assimp ---
try:
    # Add bundled Assimp to DLL search path
    if sys.platform == 'win32':
        assimp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'assimp')
        os.environ['PATH'] = assimp_path + os.pathsep + os.environ['PATH']
        # Load Assimp DLL
        try:
            ctypes.CDLL(os.path.join(assimp_path, 'assimp-vc143-mt.dll'))
        except Exception as e:
            print(f"Warning: Could not load bundled Assimp DLL: {e}")
    
    import pyassimp
    from PIL import Image
    assimp_loader = pyassimp
except ImportError:
    assimp_loader = None
    Image = None
    print("WARNING: PyAssimp or Pillow not installed. DAE/Texture loading will not be available.")
# --- END ADDED ---

class STLModelHandler:
    global_gizmo_visible = True
    """
    Handles STL model functionality including tree management, 
    transformations, and object manipulation.
    """
    
    @staticmethod
    def _rgb_to_color_name(rgb):
        """
        Convert RGB values (0-1 range) to human-readable color names.
        
        Args:
            rgb: List or tuple of [r, g, b] values in 0-1 range
            
        Returns:
            String color name
        """
        if not rgb or len(rgb) < 3:
            return "unknown"
        
        r, g, b = rgb[0], rgb[1], rgb[2]
        
        # Handle grayscale
        if abs(r - g) < 0.1 and abs(g - b) < 0.1 and abs(r - b) < 0.1:
            if r < 0.2:
                return "black"
            elif r < 0.4:
                return "dark_gray"
            elif r < 0.6:
                return "gray"
            elif r < 0.8:
                return "light_gray"
            else:
                return "white"
        
        # Find dominant color
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # Check for pure or dominant colors
        if max_val < 0.2:
            return "black"
        
        # Red dominant
        if r == max_val and r > 0.5:
            if g < 0.3 and b < 0.3:
                return "red"
            elif g > 0.4 and b < 0.3:
                return "orange"
            elif g > 0.4 and b > 0.4:
                return "pink"
        
        # Green dominant
        if g == max_val and g > 0.5:
            if r < 0.3 and b < 0.3:
                return "green"
            elif r > 0.4 and b < 0.3:
                return "yellow"
            elif r < 0.3 and b > 0.4:
                return "cyan"
        
        # Blue dominant
        if b == max_val and b > 0.5:
            if r < 0.3 and g < 0.3:
                return "blue"
            elif r > 0.4 and g < 0.3:
                return "purple"
            elif r < 0.3 and g > 0.4:
                return "cyan"
        
        # Mixed colors
        if r > 0.5 and g > 0.5 and b < 0.3:
            return "yellow"
        if r > 0.5 and b > 0.5 and g < 0.3:
            return "magenta"
        if g > 0.5 and b > 0.5 and r < 0.3:
            return "cyan"
        
        # Brown (low brightness, red-ish)
        if r > g and r > b and max_val < 0.6 and min_val < 0.3:
            return "brown"
        
        return "mixed"
    
    def __init__(self, robot_viewer, update_message_callback):
        """
        Initialize the STL model handler.
        
        Args:
            robot_viewer: The robot viewer object with STL handling capabilities
            update_message_callback: Callback function to update UI messages
        """
        self.robot_viewer = robot_viewer
        self.update_message = update_message_callback
        self.program_tree = None
        self.models_item = None
        
        # UI elements for transformation dialogs
        self.translation_popup = None
        self.rotation_popup = None
        self.scale_popup = None
        
        # Store original positions
        self.original_positions = {}
        self.original_orientations = {}
        self.original_scales = {}
        self.original_colors = {}
        
        self.meshes = []  # List to store imported meshes (mesh data like vertices, faces, edges)
        self.display_lists = []  # OpenGL display lists
        self.textures = {} # --- ADDED: Dictionary to store loaded textures {texture_path: gl_texture_id} ---
        
        # Performance optimization: Skip texture loading for faster startup
        self.skip_texture_loading = True  # Set to False if you need textures
        
        # OPTIMIZATION: Lazy edge extraction
        self.edges_extracted = {}  # Track which meshes have edges extracted
        
        # OPTIMIZATION: Cache last drawn state to avoid redundant GL calls
        self._last_drawn_state = None
        
        # Transform for the currently selected mesh
        self.selected_mesh_index = -1
        
        # Object transforms - position, orientation, scale for each mesh
        # This is the state that needs to be saved/loaded
        self.mesh_transforms = []
        
        # Gizmo state
        self.gizmo_mode = "translate"  # translate or rotate
        self.gizmo_state = 0  # 0: translate, 1: rotate, 2: hidden
        self.use_local_space = True
        self.is_manipulating_gizmo = False
        self.selected_axis = None
        self.hover_axis = None
        
        # Global scale factor (1.0 for MM world)
        self.global_scale = 0.001 
        
        # Selection for mesh picking
        self.mesh_selection_colors = []  # Colors used for mesh selection

        # Edge Selection
        self.hover_edge_info = None     # Tuple (mesh_index, edge_index) or None
        self.selected_edges = []        # Initialize the list for multiple selected edges
        self.edge_selection_colors = {} # Dictionary mapping unique edge ID to (mesh_idx, edge_idx)

        # Store file paths for reloading
        self.mesh_file_paths = []
        
        # Default color for new meshes
        self.default_color = [0.7, 0.7, 0.7]  # Light gray
        self.edge_color = [0.1, 0.1, 0.1] # Default edge color (dark gray)
        self.hover_edge_color = [0.0, 1.0, 1.0] # Cyan for hover
        self.selected_edge_color = [1.0, 0.0, 0.0] # Red for selected
    
        # --- ADDED: Bounding box visibility ---
        self.show_bounding_boxes = False
        # --- END ADDED ---
        self.highlight_bbox_indices = set() # Set to store indices of objects on conveyors

    def _position_to_mm(self, position):
        """Convert position to mm using heuristic: values < 50 are multiplied by 1000."""
        if hasattr(position, 'tolist'):
            pos_list = position.tolist()
        else:
            pos_list = list(position) if position else [0, 0, 0]
        return [p * 1000 if abs(p) < 50 else p for p in pos_list]
    
    def set_program_tree(self, program_tree, models_item=None):
        """
        Set the program tree widget reference for this handler.
        
        Args:
            program_tree: The QTreeWidget used for displaying the program structure
            models_item: The parent tree item for STL models (optional)
        """
        print(f"Setting program_tree reference in STLModelHandler")
        self.program_tree = program_tree
        self.models_item = models_item
        
        # Print debug info about the current state
        if hasattr(self, 'meshes'):
            print(f"Current handler has {len(self.meshes)} meshes when setting program_tree")
            for i, mesh in enumerate(self.meshes):
                print(f"  Mesh {i}: {mesh.get('name', 'unnamed')}")
        else:
            print("Current handler has no meshes attribute when setting program_tree")
            
        # Check if we have a robot_viewer reference with stl_handler
        if hasattr(self, 'robot_viewer') and hasattr(self.robot_viewer, 'stl_handler'):
            rv_handler = self.robot_viewer.stl_handler
            if hasattr(rv_handler, 'meshes'):
                print(f"robot_viewer.stl_handler has {len(rv_handler.meshes)} meshes")
                
                # If we have no meshes but robot_viewer.stl_handler does, sync them
                if (not hasattr(self, 'meshes') or not self.meshes) and rv_handler.meshes:
                    print("Syncing meshes from robot_viewer.stl_handler to self during set_program_tree")
                    self.meshes = rv_handler.meshes.copy()
                    self.mesh_transforms = rv_handler.mesh_transforms.copy() if hasattr(rv_handler, 'mesh_transforms') else []
                    self.mesh_selection_colors = rv_handler.mesh_selection_colors.copy() if hasattr(rv_handler, 'mesh_selection_colors') else []
                    self.display_lists = rv_handler.display_lists.copy() if hasattr(rv_handler, 'display_lists') else []
                    self.selected_mesh_index = rv_handler.selected_mesh_index
                    print(f"After sync in set_program_tree, self has {len(self.meshes)} meshes")
            else:
                print("robot_viewer.stl_handler has no meshes attribute")
        
        # If we now have meshes and a tree reference, update the tree
        if program_tree and models_item and hasattr(self, 'meshes') and self.meshes:
            print("Updating tree with existing models after setting program_tree")
            self.update_stl_models_in_tree()
    
    def cleanup_mesh(self, mesh_index):
        """Clean up resources for a specific mesh"""
        if mesh_index >= len(self.display_lists):
            return
        
        # Delete display list if it exists
        display_list = self.display_lists[mesh_index]
        if display_list is not None and glIsList(display_list):
            glDeleteLists(display_list, 1)
            print(f"Deleted display list {display_list} for mesh {mesh_index}")
        
        # Remove from edges extracted tracking
        if mesh_index in self.edges_extracted:
            del self.edges_extracted[mesh_index]
    
    def cleanup_all_meshes(self):
        """Clean up all mesh resources"""
        for i in range(len(self.display_lists)):
            self.cleanup_mesh(i)
        
        # Clear all lists
        self.meshes.clear()
        self.display_lists.clear()
        self.mesh_transforms.clear()
        self.mesh_selection_colors.clear()
        self.mesh_file_paths.clear()
        self.edges_extracted.clear()
        
        print("All mesh resources cleaned up")
    
    def _extract_edges_for_mesh(self, mesh_index):
        """Extract edges for a specific mesh on-demand"""
        if mesh_index in self.edges_extracted:
            return  # Already extracted
        
        if mesh_index >= len(self.meshes):
            return
        
        mesh_data = self.meshes[mesh_index]
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        try:
            # Create temporary trimesh to extract edges
            temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            edges_unique_indices = temp_mesh.edges_unique
            edge_vertices = vertices[edges_unique_indices]
            
            # Update mesh data
            mesh_data['edges'] = edge_vertices
            self.edges_extracted[mesh_index] = True
            
            print(f"Extracted {len(edges_unique_indices)} edges for mesh {mesh_index}")
        except Exception as e:
            print(f"Error extracting edges for mesh {mesh_index}: {e}")
            mesh_data['edges'] = np.array([])
    
    def robust_import_stl(self, file_path):
        """Try to load STL robustly: first with trimesh, then with numpy-stl as fallback."""
        # Try trimesh first
        try:
            mesh = trimesh.load(file_path, force='mesh', process=True)
            if isinstance(mesh, trimesh.Trimesh) and not mesh.is_empty:
                return mesh
        except Exception as e:
            print(f"trimesh failed: {e}")
        # Try numpy-stl as fallback
        if numpy_stl_mesh is not None:
            try:
                stl_mesh = numpy_stl_mesh.Mesh.from_file(file_path)
                # numpy-stl gives you faces as v0, v1, v2 arrays
                vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))
                # Remove duplicate vertices and build faces
                unique_verts, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
                faces = inverse_indices.reshape(-1, 3)
                mesh = trimesh.Trimesh(vertices=unique_verts, faces=faces, process=True)
                if not mesh.is_empty:
                    return mesh
            except Exception as e:
                print(f"numpy-stl failed: {e}")
        return None

    def import_stl(self):
        """Import an STL, STEP, or OBJ file and prepare it for rendering"""
        print(f"[STLHandler] import_stl called on instance ID: {id(self)}")
        print("Opening file dialog to select model file...")
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        main_window = QApplication.activeWindow()
        file_name, _ = QFileDialog.getOpenFileName(
            main_window,  # Use the main window as parent
            "Open Model File",
            "",
            "Model Files (*.stl *.step *.stp *.obj *.dae);;STL Files (*.stl);;STEP Files (*.step *.stp);;OBJ Files (*.obj);;DAE Files (*.dae);;All Files (*.*)" # MODIFIED: Added DAE
        )
        print(f"File dialog result: {'Selected: ' + file_name if file_name else 'Cancelled'}")
        if file_name:
            # Process import asynchronously to keep UI responsive
            QTimer.singleShot(0, lambda: self._import_stl_async(file_name))
            return True
        return False
    
    def _import_stl_async(self, file_name):
        """Async helper to import STL without blocking UI"""
        # Show loading message
        self.update_message(f"Loading {os.path.basename(file_name)}...")
        
        # Process the import
        self._do_import_stl(file_name)
        
        # Update message
        self.update_message(f"Loaded {os.path.basename(file_name)}")
    
    def _do_import_stl(self, file_name):
        """Internal method to perform the actual import"""
        # --- MODIFIED: Handle DAE files separately ---
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext == '.dae':
            if assimp_loader is None or Image is None: # CORRECTED CHECK
                self.update_message("PyAssimp or Pillow not installed. Cannot load DAE files.")
                print("Error: PyAssimp or Pillow not installed. DAE loading skipped.")
                return
            self._load_dae_with_pyassimp(file_name)
            return
        # --- END MODIFIED ---

        mesh = self.robust_import_stl(file_name)
        if mesh is None:
            print(f"Error: Could not load STL file '{file_name}' with any method.")
            self.update_message(f"Failed to import STL file: {os.path.basename(file_name)}. Try re-exporting or converting to standard STL.")
            return
        try:
                # The rest of the code expects a trimesh.Trimesh object in 'mesh'
                # Calculate and store the dimensions before any scaling
                bounding_box = mesh.bounds
                dimensions = bounding_box[1] - bounding_box[0]  # Calculate dimensions from bounding box
                self.mesh_dimensions = dimensions * self.global_scale  # Apply global scale
                print(f"Original model dimensions (before scaling): {dimensions}") # Units depend on source file, often mm
                print(f"Scaled model dimensions: {self.mesh_dimensions} m")
                # Ensure vertices and faces are numpy arrays
                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int32)
                # +++ Add Logging: Original Vertices +++
                print(f"[DEBUG Import] Original first vertex: {vertices[0] if len(vertices) > 0 else 'N/A'}")
                # +++ End Logging +++
                # Center the mesh in all three axes (X, Y, Z)
                center = np.mean(vertices, axis=0)
                vertices[:, 0] -= center[0]  # Center X
                vertices[:, 1] -= center[1]  # Center Y
                vertices[:, 2] -= center[2]  # Center Z

                # --- PATCH: Standardize to MILLIMETERS ---
                bbox = np.ptp(vertices, axis=0)
                max_dim = np.max(bbox)
                
                # If object is very small (< 1.0), it's likely specified in Meters.
                # Since our world is now MILLIMETERS, we must scale it up by 1000.
                if max_dim < 1.0:
                    scale = 1000.0
                    print("[STL Import] Detected small object (likely Meters), scaling to MM (x1000)")
                else:
                    # Otherwise assume it's already in MM (standard for STL)
                    scale = 1.0
                    print("[STL Import] Detected normal object (likely MM), keeping scale 1.0")

                vertices = vertices * scale
                # +++ Add Logging: Scaled Vertices +++
                print(f"[DEBUG Import] Scaled first vertex (scale={self.global_scale}): {vertices[0] if len(vertices) > 0 else 'N/A'}")
                # +++ End Logging +++
                # --- Extract edges AFTER transforming vertices ---
                # OPTIMIZATION: Defer edge extraction to avoid blocking UI
                edges_unique_indices = []
                edge_vertices = np.array([])  # Empty for now
                print(f"Deferring edge extraction for performance")
                # +++ Add Logging: Edge Vertices +++
                print(f"[DEBUG Import] Edge extraction deferred")
                # +++ End Logging +++
                # Extract filename only (without path and extension)
                base_name = os.path.basename(file_name)
                model_name = os.path.splitext(base_name)[0]
                # Try to extract color from filename (e.g., "red_box.stl" -> red)
                color_name = None
                for color in ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'purple', 'orange', 'brown', 'pink']:
                    if color in model_name.lower():
                        color_name = color
                        break
                # If no color found in filename, use default gray
                if not color_name:
                    color_name = 'gray'
                # Convert color name to RGB (normalized to [0-1] range)
                color_map = {
                    'red': [1.0, 0.0, 0.0], 'green': [0.0, 1.0, 0.0], 'blue': [0.0, 0.0, 1.0],
                    'yellow': [1.0, 1.0, 0.0], 'white': [1.0, 1.0, 1.0], 'black': [0.0, 0.0, 0.0],
                    'gray': [0.7, 0.7, 0.7], 'purple': [0.5, 0.0, 0.5], 'orange': [1.0, 0.5, 0.0],
                    'brown': [0.6, 0.3, 0.0], 'pink': [1.0, 0.7, 0.7]
                }
                color = color_map[color_name]
                # If the mesh has vertex colors, use those instead
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
                    vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
                    color = np.mean(vertex_colors, axis=0).tolist()
                    print(f"Using mesh vertex colors: RGB({color})")
                print(f"Assigned color to {model_name}: {color_name} (RGB: {color})")
                # --- ADDED: Calculate and store local AABB ---
                local_min_bounds = np.min(vertices, axis=0)
                local_max_bounds = np.max(vertices, axis=0)
                print(f"  - Calculated Local AABB: Min={local_min_bounds}, Max={local_max_bounds}")
                # --- END ADDED ---
                mesh_data = {
                    'vertices': vertices,
                    'faces': faces,
                    'edges': edge_vertices, # Store edge vertex coordinates as list
                    'name': model_name,
                    'dimensions': self.mesh_dimensions, # Scaled dimensions
                    'color': color,
                    'color_name': color_name,
                    'original_bounding_box': mesh.bounds, # Store original bounds if needed later
                    # --- ADDED: Store local bounds ---
                    'local_min_bounds': local_min_bounds,
                    'local_max_bounds': local_max_bounds
                    # --- END ADDED ---
                }
                self.meshes.append(mesh_data)
                self.mesh_transforms.append({
                    'position': np.array([0.0, -300.0, 100.0], dtype=np.float32), # Start at Y=-300mm to avoid robot collision
                    'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # quaternion [x,y,z,w]
                    'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
                })
                mesh_id = len(self.meshes) - 1
                r = 0.3 + (mesh_id % 3) * 0.02
                g = 0.3 + ((mesh_id // 3) % 3) * 0.02
                b = 0.3 + ((mesh_id // 9) % 3) * 0.02
                self.mesh_selection_colors.append((r, g, b))
                self.mesh_file_paths.append(file_name)
                
                # OPTIMIZATION: Defer display list creation to avoid blocking
                self.display_lists.append(None)  # Placeholder
                
                self.selected_mesh_index = len(self.meshes) - 1
                print(f"Successfully loaded {file_name}")
                print(f"Model name: {model_name}")
                print(f"Model dimensions: {np.ptp(vertices, axis=0)}")
                print(f"Vertex count: {len(vertices)}")
                print(f"Face count: {len(faces)}")
                print(f"Total models loaded: {len(self.meshes)}")
                
                # Create display list asynchronously
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(100, lambda idx=len(self.meshes)-1: self._create_display_list_deferred(idx))
                
                # Update UI
                if self.robot_viewer:
                    self.robot_viewer.update()
                
        except Exception as e:
            print(f"Error loading STL: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_display_list_deferred(self, mesh_index):
        """Create display list for a mesh asynchronously"""
        if mesh_index >= len(self.meshes):
            return
        
        mesh_data = self.meshes[mesh_index]
        display_list = self.create_mesh_display_list(mesh_data)
        self.display_lists[mesh_index] = display_list
        
        print(f"Display list created for mesh {mesh_index}")
        
        # Update viewer
        if self.robot_viewer:
            self.robot_viewer.update()
    
    def _load_texture(self, texture_path):
        """Loads a texture and returns its OpenGL ID. Caches textures."""
        # Skip texture loading if disabled for performance
        if self.skip_texture_loading:
            return None
            
        if texture_path in self.textures:
            return self.textures[texture_path]

        if not Image:
            return None

        try:
            
            # Check if file exists
            if not os.path.exists(texture_path):
                print(f"Texture file not found: {texture_path}")
                return None
                
            # Try to open the image with PIL - attempt different modes if the first try fails
            try:
                img = Image.open(texture_path)
                print(f"Opened image: {img.width}x{img.height}, mode: {img.mode}, format: {img.format}")
            except Exception as e:
                print(f"Error opening image: {e}")
                return None
                
            # Convert image to a suitable mode for OpenGL
            try:
                if img.mode == 'RGBA':
                    pass  # Already in RGBA mode
                elif img.mode == 'RGB':
                    pass  # RGB mode is also acceptable
                else:
                    print(f"Converting image from {img.mode} to RGBA")
                    img = img.convert("RGBA")
            except Exception as e:
                print(f"Error converting image mode: {e}")
                try:
                    # Fallback to RGB
                    img = img.convert("RGB")
                except:
                    print(f"Failed to convert image even to RGB")
                    return None
            
            # Flip image vertically for OpenGL (OpenGL has 0,0 at bottom-left, PIL at top-left)
            try:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            except Exception as e:
                print(f"Error flipping image: {e} - continuing anyway")
            
            # Convert to byte array for OpenGL
            try:
                # Convert to NumPy array
                img_data = np.array(img)
                print(f"Image array shape: {img_data.shape}, dtype: {img_data.dtype}")
                
                # Check for errors in the array
                if img_data.size == 0:
                    print(f"Error: Image data is empty")
                    return None
                
                # Create byte array for OpenGL
                img_data_bytes = img_data.tobytes()
                print(f"Image data size: {len(img_data_bytes)} bytes")
            except Exception as e:
                print(f"Error preparing image data: {e}")
                return None

            # Generate and bind an OpenGL texture
            try:
                texture_id = glGenTextures(1)
                if texture_id == 0:
                    print("Failed to generate texture ID")
                    return None
                    
                glBindTexture(GL_TEXTURE_2D, texture_id)
                
                # Check for OpenGL errors
                error = glGetError()
                if error != GL_NO_ERROR:
                    print(f"OpenGL error after binding texture: {gluErrorString(error)}")
            except Exception as e:
                print(f"Error generating/binding texture: {e}")
                return None
            
            # Set texture parameters
            try:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)  # Changed to LINEAR (no mipmaps)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                
                # Check for OpenGL errors
                error = glGetError()
                if error != GL_NO_ERROR:
                    print(f"OpenGL error after setting texture parameters: {gluErrorString(error)}")
            except Exception as e:
                print(f"Error setting texture parameters: {e}")
                return None
            
            # Upload texture data to OpenGL
            try:
                # Determine the OpenGL format from the image mode
                if img.mode == "RGBA":
                    gl_format = GL_RGBA
                    internal_format = GL_RGBA
                elif img.mode == "RGB":
                    gl_format = GL_RGB
                    internal_format = GL_RGB
                else:
                    gl_format = GL_RGBA
                    internal_format = GL_RGBA
                
                # Use the simpler glTexImage2D approach - more reliable
                glTexImage2D(GL_TEXTURE_2D, 0, internal_format, img.width, img.height,
                             0, gl_format, GL_UNSIGNED_BYTE, img_data_bytes)
                
                # Check for OpenGL errors
                error = glGetError()
                if error != GL_NO_ERROR:
                    return None
                
                # Generate mipmaps (optional, but helps with rendering quality)
                try:
                    # Check if glGenerateMipmap is available in the current OpenGL context
                    has_generate_mipmap = False
                    try:
                        # Try to see if the function exists
                        if callable(glGenerateMipmap):
                            has_generate_mipmap = True
                    except (NameError, AttributeError):
                        has_generate_mipmap = False
                    
                    if has_generate_mipmap:
                        print("Using glGenerateMipmap for mipmaps")
                        glGenerateMipmap(GL_TEXTURE_2D)
                    else:
                        print("glGenerateMipmap not available, using gluBuild2DMipmaps")
                        try:
                            # Try to use gluBuild2DMipmaps as fallback
                            gluBuild2DMipmaps(GL_TEXTURE_2D, internal_format, img.width, img.height,
                                              gl_format, GL_UNSIGNED_BYTE, img_data_bytes)
                        except Exception as mip_error:
                            print(f"Warning: Could not generate mipmaps using gluBuild2DMipmaps: {mip_error}")
                            # Fall back to GL_LINEAR filtering without mipmaps
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                except Exception as e:
                    print(f"Warning: Could not generate mipmaps: {e} - texture will still work")
                    # Fall back to GL_LINEAR filtering without mipmaps
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            except Exception as e:
                return None
            
            # Cache the texture ID
            self.textures[texture_path] = texture_id
            return texture_id
            
        except Exception as e:
            # Texture loading failed - skip silently for faster startup
            return None

    def _load_dae_with_pyassimp(self, file_path):
        """Loads a DAE file using PyAssimp, extracting meshes, materials, and textures."""
        if assimp_loader is None: # CORRECTED CHECK
            print("PyAssimp not available, cannot load DAE.")
            return False
        
        print(f"Loading DAE file: {file_path} with PyAssimp")
        try:
            with assimp_loader.load(
                file_path,
                processing=assimp_loader.postprocess.aiProcess_Triangulate |
                          assimp_loader.postprocess.aiProcess_GenNormals |
                          assimp_loader.postprocess.aiProcess_FlipUVs
            ) as scene:
                if not scene or not scene.meshes:
                    print(f"DAE file '{file_path}' loaded no scene or no meshes.")
                    self.update_message(f"DAE file '{os.path.basename(file_path)}' contains no mesh data.")
                    return False

                base_dir = os.path.dirname(file_path)
                model_name_base = os.path.splitext(os.path.basename(file_path))[0]
                
                # Check if we need to combine meshes (multiple meshes with different materials)
                if len(scene.meshes) > 1:
                    print(f"DAE file has {len(scene.meshes)} meshes - combining into single model")
                    return self._load_dae_combined_mesh(scene, file_path, base_dir, model_name_base)
                else:
                    # Single mesh, process normally
                    return self._load_dae_single_mesh(scene, file_path, base_dir, model_name_base)
        except Exception as e:
            print(f"Error importing DAE file '{file_path}' with PyAssimp: {e}")
            self.update_message(f"Failed to import DAE: {os.path.basename(file_path)}. Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _load_dae_single_mesh(self, scene, file_path, base_dir, model_name_base):
        """Process a DAE file with a single mesh (legacy implementation)."""
        imported_any_mesh = False
        
        for mesh_idx, p_mesh in enumerate(scene.meshes):
            vertices = np.array(p_mesh.vertices, dtype=np.float32)
            faces = np.array(p_mesh.faces, dtype=np.int32)
            normals = np.array(p_mesh.normals, dtype=np.float32) if p_mesh.normals is not None and len(p_mesh.normals) > 0 else None
            
            texture_coords = None
            texture_id = None
            has_texture_flag = False
            mesh_color = self.default_color # Default color
            color_name = 'gray'

            if p_mesh.materialindex is not None and p_mesh.materialindex < len(scene.materials):
                material = scene.materials[p_mesh.materialindex]
                # Get diffuse color from material
                diffuse_color = material.properties.get("diffuse") # Returns [r,g,b] or [r,g,b,a]
                if diffuse_color and len(diffuse_color) >= 3:
                    mesh_color = diffuse_color[:3] # Take RGB
                # Get texture
                tex_file = material.properties.get("file")
                if tex_file:
                    texture_path_candidate = os.path.join(base_dir, tex_file)
                    if not os.path.exists(texture_path_candidate):
                        if os.path.isabs(tex_file) and os.path.exists(tex_file):
                            texture_path_candidate = tex_file
                        else:
                            for subfolder in ["textures", "texture", "images", "image", "tex"]:
                                potential_path = os.path.join(base_dir, subfolder, os.path.basename(tex_file))
                                if os.path.exists(potential_path):
                                    texture_path_candidate = potential_path
                                    break
                            else:
                                print(f"Texture file '{tex_file}' not found. Searched in '{base_dir}' and common subdirs.")
                                texture_path_candidate = None
                    if texture_path_candidate:
                        texture_id = self._load_texture(texture_path_candidate)
                        if texture_id is not None:
                            if p_mesh.texturecoords and len(p_mesh.texturecoords[0]) == len(vertices):
                                texture_coords = np.array(p_mesh.texturecoords[0], dtype=np.float32)[:, :2]
                                has_texture_flag = True
                            else:
                                print(f"Warning: Mesh '{model_name_base}_{mesh_idx}' has texture but mismatched or no texture coordinates.")
                                texture_id = None
                        else:
                            print(f"Failed to load texture: {texture_path_candidate}")
            if len(vertices) > 0:
                center = np.mean(vertices, axis=0)
                vertices[:, 0] -= center[0]  # Center X
                vertices[:, 1] -= center[1]  # Center Y
                vertices[:, 2] -= center[2]  # Center Z
                vertices *= self.global_scale
            edge_vertices = []
            local_min_bounds = np.min(vertices, axis=0) if len(vertices) > 0 else np.array([0,0,0])
            local_max_bounds = np.max(vertices, axis=0) if len(vertices) > 0 else np.array([0,0,0])
            current_model_name = f"{model_name_base}_{mesh_idx}" if len(scene.meshes) > 1 else model_name_base
            mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'edges': edge_vertices,
                'name': current_model_name,
                'dimensions': (local_max_bounds - local_min_bounds).tolist(),
                'color': mesh_color,
                'color_name': color_name,
                'original_bounding_box': None,
                'local_min_bounds': local_min_bounds.tolist(),
                'local_max_bounds': local_max_bounds.tolist(),
                'texture_coords': texture_coords,
                'texture_id': texture_id,
                'has_texture': has_texture_flag
            }
            self.meshes.append(mesh_data)
            self.mesh_transforms.append({
                'position': np.array([0.0, -300.0, 100.0], dtype=np.float32),  # mm - same as STL import
                'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
            })
            mesh_list_idx = len(self.meshes) - 1
            r = 0.3 + (mesh_list_idx % 3) * 0.02
            g = 0.3 + ((mesh_list_idx // 3) % 3) * 0.02
            b = 0.3 + ((mesh_list_idx // 9) % 3) * 0.02
            self.mesh_selection_colors.append((r, g, b))
            self.mesh_file_paths.append(file_path)
            display_list = self.create_mesh_display_list(mesh_data)
            self.display_lists.append(display_list)
            print(f"Successfully processed mesh '{current_model_name}' from DAE. Has texture: {has_texture_flag}")
            imported_any_mesh = True
            
        if imported_any_mesh:
            self.selected_mesh_index = len(self.meshes) - 1
            self.update_message(f"Successfully imported DAE: {os.path.basename(file_path)}")
            print(f"Total models loaded: {len(self.meshes)}")
            return True
        else:
            self.update_message(f"No meshes found or processed in DAE: {os.path.basename(file_path)}")
            return False
            
    def _load_dae_combined_mesh(self, scene, file_path, base_dir, model_name_base):
        """Process a DAE file by combining all meshes into a single model with per-face materials."""
        print(f"Combining {len(scene.meshes)} meshes from DAE file into single model...")
        
        # First, collect all the materials
        materials = []
        texture_info = {}
        
        # Process all materials first
        for material_idx, material in enumerate(scene.materials):
            # Material processing - verbose logging disabled for faster loading
            # print(f"Processing material {material_idx}: {material.name if hasattr(material, 'name') else 'unnamed'}")
            
            diffuse_color = material.properties.get("diffuse", self.default_color)
            if diffuse_color and len(diffuse_color) >= 3:
                color = diffuse_color[:3]  # Take RGB
            else:
                color = self.default_color
                
            # Check for texture
            texture_id = None
            texture_path = None
            
            # Skip texture loading if disabled for performance
            if not self.skip_texture_loading:
                # Extract texture file path - use the correct key/tuple format from the log
                tex_file = None
                for key, value in material.properties.items():
                    # Check for file key in various formats that might appear in the properties dictionary
                    if isinstance(key, tuple) and len(key) >= 2 and key[0] == 'file':
                        tex_file = value
                        break
                    elif key == 'file':
                        tex_file = value
                        break
            else:
                tex_file = None
            
            if tex_file:
                # DAE file directory
                dae_dir = os.path.dirname(file_path)
                
                # First, check in the same directory as the DAE file directly (most common case)
                texture_path_candidate = os.path.join(dae_dir, os.path.basename(tex_file))
                
                # If texture file doesn't exist at the expected path, try other locations
                if not os.path.exists(texture_path_candidate):
                    # Try in base_dir (which might be different from dae_dir)
                    texture_path_candidate = os.path.join(base_dir, tex_file)
                    
                    if not os.path.exists(texture_path_candidate):
                        
                        # Try using absolute path if provided
                        if os.path.isabs(tex_file) and os.path.exists(tex_file):
                            texture_path_candidate = tex_file
                        # Try with just the filename in the base directory
                        else:
                            texture_basename = os.path.basename(tex_file)
                            
                            # Try in DAE directory first
                            direct_path = os.path.join(dae_dir, texture_basename)
                            
                            if os.path.exists(direct_path):
                                texture_path_candidate = direct_path
                            else:
                                # Try in base_dir 
                                direct_path = os.path.join(base_dir, texture_basename)
                                
                                if os.path.exists(direct_path):
                                    texture_path_candidate = direct_path
                                else:
                                    # Try common subdirectories (fast check only)
                                    found = False
                                    
                                    # First try subdirectories of the DAE directory
                                    for subfolder in ["textures", "texture", "images", "image", "tex", ""]:
                                        potential_path = os.path.join(dae_dir, subfolder, texture_basename)
                                        if os.path.exists(potential_path):
                                            texture_path_candidate = potential_path
                                            found = True
                                            break
                                    
                                    # Then try subdirectories of the base_dir if different from dae_dir
                                    if not found and dae_dir != base_dir:
                                        for subfolder in ["textures", "texture", "images", "image", "tex", ""]:
                                            potential_path = os.path.join(base_dir, subfolder, texture_basename)
                                            if os.path.exists(potential_path):
                                                texture_path_candidate = potential_path
                                                found = True
                                                break
                                    
                                    # Skip expensive directory tree walk - just try different extensions
                                    if not found:
                                        basename_no_ext = os.path.splitext(texture_basename)[0]
                                        for ext in ['.png', '.jpg', '.jpeg', '.tga', '.bmp']:
                                            test_file = os.path.join(dae_dir, basename_no_ext + ext)
                                            if os.path.exists(test_file):
                                                texture_path_candidate = test_file
                                                break
                                        else:
                                            # Texture not found - skip silently for faster loading
                                            texture_path_candidate = None
                
                # If we found a texture path, try to load it
                if texture_path_candidate and os.path.exists(texture_path_candidate):
                    texture_id = self._load_texture(texture_path_candidate)
                    if texture_id is not None:
                        texture_path = texture_path_candidate
                        # Store texture info for future reference
                        texture_info[material_idx] = {
                            'path': texture_path,
                            'id': texture_id
                        }
            
            # Create the material data
            materials.append({
                'color': color,
                'texture_id': texture_id,
                'texture_path': texture_path,
                'name': material.name if hasattr(material, 'name') else f"Material_{material_idx}"
            })
            
        # Now process meshes and combine them
        all_vertices = []
        all_faces = []
        all_normals = []
        all_texcoords = []
        all_face_materials = []
        
        vertex_offset = 0
        texcoord_offset = 0
        
        # Process each mesh and combine the data
        for mesh_idx, p_mesh in enumerate(scene.meshes):
            vertices = np.array(p_mesh.vertices, dtype=np.float32)
            faces = np.array(p_mesh.faces, dtype=np.int32)
            
            # Get material for this mesh
            # Convert to scalar if it's an array
            if hasattr(p_mesh, 'materialindex'):
                # Fix: handle if materialindex is an array
                if isinstance(p_mesh.materialindex, np.ndarray):
                    material_index = p_mesh.materialindex[0] if len(p_mesh.materialindex) > 0 else -1
                else:
                    material_index = p_mesh.materialindex
            else:
                material_index = -1
            
            # Get normals
            mesh_normals = None
            if p_mesh.normals is not None and len(p_mesh.normals) > 0:
                mesh_normals = np.array(p_mesh.normals, dtype=np.float32)
                all_normals.extend(mesh_normals)
            
            # Get texture coordinates if available
            has_texcoords = False
            mesh_texcoords = []
            
            # Fix: check properly if p_mesh has texture coordinates
            has_texture_coords = (hasattr(p_mesh, 'texturecoords') and 
                                  p_mesh.texturecoords is not None and 
                                  len(p_mesh.texturecoords) > 0 and 
                                  p_mesh.texturecoords[0] is not None and 
                                  len(p_mesh.texturecoords[0]) > 0)
            
            if isinstance(material_index, (int, np.integer)) and material_index >= 0 and material_index < len(materials) and has_texture_coords:
                try:
                    mesh_texcoords = np.array(p_mesh.texturecoords[0], dtype=np.float32)[:, :2]
                    has_texcoords = True
                    print(f"  Mesh has {len(mesh_texcoords)} texture coordinates")
                except (IndexError, ValueError) as e:
                    print(f"  Error processing texture coordinates: {e}")
            
            # Store per-face texture coordinates for this mesh
            face_texcoords = []
            if has_texcoords:
                for face in faces:
                    try:
                        face_texcoords.append([mesh_texcoords[idx] for idx in face])
                    except IndexError:
                        print(f"  Warning: Invalid texture coordinate index in face")
                        continue
                    
                # Add all texture coordinates to the combined list
                for tc in mesh_texcoords:
                    all_texcoords.append(tc)
            
            # Add faces with adjusted indices and material info
            for face_idx, face in enumerate(faces):
                try:
                    # Adjust vertex indices
                    adjusted_face = face + vertex_offset
                    all_faces.append(adjusted_face)
                    
                    # Store material index for this face
                    all_face_materials.append(material_index)
                    
                    # If this mesh doesn't have texture coords but we need them for consistent rendering
                    if not has_texcoords and len(all_texcoords) > 0:
                        # Add dummy texture coordinates for this face
                        all_texcoords.extend([[0, 0]] * 3)  # Assuming triangulated faces
                except Exception as e:
                    print(f"  Error processing face {face_idx}: {e}")
                    continue
            
            # Add vertices
            all_vertices.extend(vertices)
            
            # Update vertex offset for next mesh
            vertex_offset += len(vertices)
            
            print(f"  Processed mesh {mesh_idx}: {len(vertices)} vertices, {len(faces)} faces")
        
        # Convert lists to numpy arrays
        all_vertices = np.array(all_vertices, dtype=np.float32) if all_vertices else np.zeros((0, 3), dtype=np.float32)
        all_faces = np.array(all_faces, dtype=np.int32) if all_faces else np.zeros((0, 3), dtype=np.int32)
        
        # Perform centering on the combined mesh
        if len(all_vertices) > 0:
            center = np.mean(all_vertices, axis=0)
            all_vertices[:, 0] -= center[0]  # Center X
            all_vertices[:, 1] -= center[1]  # Center Y
            all_vertices[:, 2] -= center[2]  # Center Z
            all_vertices *= self.global_scale  # Apply global scale
        
        # Calculate bounds
        local_min_bounds = np.min(all_vertices, axis=0) if len(all_vertices) > 0 else np.array([0,0,0])
        local_max_bounds = np.max(all_vertices, axis=0) if len(all_vertices) > 0 else np.array([0,0,0])
        
        # Extract edges from the combined mesh
        edge_vertices = []
        if len(all_vertices) > 0 and len(all_faces) > 0:
            try:
                # Create a temporary trimesh to extract edges
                temp_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
                edges_unique_indices = temp_mesh.edges_unique
                edge_vertices = all_vertices[edges_unique_indices]
            except Exception as e:
                print(f"Warning: Error extracting edges: {e}")
                # Create empty edge data
                edge_vertices = np.zeros((0, 2, 3), dtype=np.float32)
        
        # Create the combined mesh data
        mesh_data = {
            'vertices': all_vertices,
            'faces': all_faces,
            'normals': np.array(all_normals, dtype=np.float32) if all_normals else None,
            'edges': edge_vertices,
            'name': model_name_base,
            'dimensions': (local_max_bounds - local_min_bounds).tolist(),
            'color': self.default_color,  # Default color for the overall mesh
            'color_name': 'multi',  # Indicate this has multiple materials
            'original_bounding_box': None,
            'local_min_bounds': local_min_bounds.tolist(),
            'local_max_bounds': local_max_bounds.tolist(),
            'texture_coords': np.array(all_texcoords, dtype=np.float32) if all_texcoords else None,
            'has_texture': any(m['texture_id'] is not None for m in materials),
            'face_materials': all_face_materials,  # Store material index per face
            'materials': materials,  # Store all materials
            'texture_info': texture_info  # Store texture info for debugging
        }
        
        # Add to our lists
        self.meshes.append(mesh_data)
        self.mesh_transforms.append({
            'position': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
        })
        
        # Add selection color
        mesh_list_idx = len(self.meshes) - 1
        r = 0.3 + (mesh_list_idx % 3) * 0.02
        g = 0.3 + ((mesh_list_idx // 3) % 3) * 0.02
        b = 0.3 + ((mesh_list_idx // 9) % 3) * 0.02
        self.mesh_selection_colors.append((r, g, b))
        
        # Store file path
        self.mesh_file_paths.append(file_path)
        
        # Create display list
        display_list = self.create_multi_material_display_list(mesh_data)
        self.display_lists.append(display_list)
        
        print(f"Successfully created combined mesh '{model_name_base}' from DAE with {len(all_vertices)} vertices, {len(all_faces)} faces, and {len(materials)} materials.")
        
        # Select the newly imported mesh
        self.selected_mesh_index = len(self.meshes) - 1
        self.update_message(f"Successfully imported DAE: {os.path.basename(file_path)} as a single model with {len(materials)} materials")
        
        return True

    def create_multi_material_display_list(self, mesh_data):
        """Create an OpenGL display list for a mesh with multiple materials."""
        try:
            mesh_name = mesh_data.get('name', 'Unnamed')
            num_vertices = len(mesh_data['vertices']) if 'vertices' in mesh_data else 0
            num_faces = len(mesh_data['faces']) if 'faces' in mesh_data else 0
            num_materials = len(mesh_data.get('materials', [])) 
            
            print(f"[DEBUG] Creating multi-material display list for '{mesh_name}' ({num_vertices} verts, {num_faces} faces, {num_materials} materials)")
            
            # Print texture information for debugging
            if 'texture_info' in mesh_data:
                for mat_idx, tex_info in mesh_data['texture_info'].items():
                    print(f"[DEBUG] Material {mat_idx} has texture: {tex_info['path']} (ID: {tex_info['id']})")
            
            # Print materials information
            materials = mesh_data.get('materials', [])
            for mat_idx, material in enumerate(materials):
                texture_id = material.get('texture_id')
                texture_path = material.get('texture_path')
                if texture_id is not None:
                    print(f"[DEBUG] Material {mat_idx} ({material.get('name', 'unnamed')}) has texture ID: {texture_id}, path: {texture_path}")
                else:
                    print(f"[DEBUG] Material {mat_idx} ({material.get('name', 'unnamed')}) has no texture")
            
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"[DEBUG] OpenGL error before glGenLists: {gluErrorString(error)}")
                
            display_list = glGenLists(1)
            if display_list == 0:
                print("Error: Could not create display list")
                return None
                
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            normals = mesh_data.get('normals')
            texture_coords = mesh_data.get('texture_coords')
            materials = mesh_data.get('materials', [])
            face_materials = mesh_data.get('face_materials', [])
            
            # Log texture coords info
            if texture_coords is not None:
                print(f"[DEBUG] Texture coordinates array shape: {texture_coords.shape}")
            else:
                print("[DEBUG] No texture coordinates available")
            
            # Ensure face_materials has an entry for each face
            if len(face_materials) < len(faces):
                print(f"[DEBUG] Extending face_materials from {len(face_materials)} to {len(faces)}")
                face_materials.extend([-1] * (len(faces) - len(face_materials)))
                
            # Group faces by material for efficient rendering
            material_to_faces = {}
            for face_idx, material_idx in enumerate(face_materials):
                if material_idx not in material_to_faces:
                    material_to_faces[material_idx] = []
                material_to_faces[material_idx].append(face_idx)
            
            # Log grouping result
            for mat_idx, faces_list in material_to_faces.items():
                mat_name = materials[mat_idx]['name'] if 0 <= mat_idx < len(materials) else "default"
                print(f"[DEBUG] Material {mat_idx} ({mat_name}) has {len(faces_list)} faces")
            
            glNewList(display_list, GL_COMPILE)
            
            # Enable lighting and normalize normals for proper lighting
            glEnable(GL_LIGHTING)
            glEnable(GL_NORMALIZE)  # Important for scaled objects
            
            # Pre-calculate face normals for efficiency (if not provided)
            face_normals = {}
            if normals is None:
                print("[DEBUG] Calculating face normals (vertex normals not provided)")
                for face_idx, face in enumerate(faces):
                    try:
                        v1, v2, v3 = [vertices[i] for i in face]
                        normal_vec = np.cross(v2 - v1, v3 - v1)
                        norm = np.linalg.norm(normal_vec)
                        if norm > 1e-10:  # Avoid division by zero
                            face_normals[face_idx] = normal_vec / norm
                        else:
                            face_normals[face_idx] = np.array([0.0, 1.0, 0.0])  # Default up vector for degenerate faces
                    except Exception as e:
                        print(f"Error calculating normal for face {face_idx}: {str(e)}")
                        face_normals[face_idx] = np.array([0.0, 1.0, 0.0])
            
            # First pass - disable depth writes and render backfaces for better edge rendering
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)  # Don't write to color buffer
            glDepthMask(GL_FALSE)  # Don't write to depth buffer
            glEnable(GL_CULL_FACE)
            glCullFace(GL_FRONT)  # Cull front faces, only render back faces
            
            for material_idx, face_indices in material_to_faces.items():
                if not face_indices:
                    continue
                    
                glBegin(GL_TRIANGLES)
                for face_idx in face_indices:
                    if face_idx >= len(faces):
                        continue
                    face = faces[face_idx]
                    for vertex_idx in face:
                        if vertex_idx >= len(vertices):
                            continue
                        glVertex3fv(vertices[vertex_idx].astype(np.float32))
                glEnd()
            
            # Restore state for main rendering pass
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glDepthMask(GL_TRUE)
            glCullFace(GL_BACK)  # Cull back faces, render front faces
            
            # Main rendering pass - draw the mesh by material groups
            for material_idx, face_indices in material_to_faces.items():
                # Skip if no faces use this material
                if not face_indices:
                    continue
                
                # Get material properties
                if material_idx < 0 or material_idx >= len(materials):
                    # Use default material
                    color = self.default_color
                    texture_id = None
                    has_texture = False
                    material_name = "default"
                else:
                    # Get material properties
                    material = materials[material_idx]
                    material_name = material.get('name', f"Material_{material_idx}")
                    color = material.get('color', self.default_color)
                    texture_id = material.get('texture_id')
                    texture_path = material.get('texture_path')
                    has_texture = texture_id is not None and texture_id > 0
                
                print(f"[DEBUG] Rendering material {material_idx} ({material_name}), has texture: {has_texture}, texture_id: {texture_id}")
                
                # Set material properties based on whether we have a texture
                if has_texture:
                    # Verify texture ID is valid
                    if texture_id <= 0:
                        print(f"[WARNING] Invalid texture ID {texture_id} for material {material_idx}")
                        has_texture = False
                    else:
                        # Try to reload texture if needed
                        if texture_path and texture_path not in self.textures and os.path.exists(texture_path):
                            print(f"[DEBUG] Attempting to reload texture {texture_path}")
                            new_texture_id = self._load_texture(texture_path)
                            if new_texture_id is not None:
                                texture_id = new_texture_id
                                material['texture_id'] = new_texture_id
                                print(f"[DEBUG] Reloaded texture with ID {new_texture_id}")
                            else:
                                has_texture = False
                
                if has_texture:
                    # Enable texturing and bind texture
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                    
                    # Check for OpenGL errors after binding texture
                    error = glGetError()
                    if error != GL_NO_ERROR:
                        print(f"[DEBUG] OpenGL error after binding texture {texture_id}: {gluErrorString(error)}")
                        glDisable(GL_TEXTURE_2D)
                        has_texture = False
                    else:
                        # Set material color to white to show texture properly, but use diffuse color as a tint
                        ambient = [c * 0.8 for c in color] + [1.0]  # Use material color as ambient tint
                        diffuse = [1.0, 1.0, 1.0, 1.0]  # White diffuse to show texture colors
                        specular = [0.3, 0.3, 0.3, 1.0]
                        shininess = 16.0
                        
                        # Set texture environment mode to modulate with material color
                        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                else:
                    # Disable texturing for non-textured materials
                    glDisable(GL_TEXTURE_2D)
                    ambient = [c * 0.3 for c in color] + [1.0]
                    diffuse = [c * 0.7 for c in color] + [1.0]
                    specular = [0.2, 0.2, 0.2, 1.0]
                    shininess = 32.0
                
                # Apply material properties
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
                
                # Draw the faces for this material
                glBegin(GL_TRIANGLES)
                
                face_count = 0
                for face_idx in face_indices:
                    if face_idx >= len(faces):
                        continue  # Skip invalid indices
                        
                    face = faces[face_idx]
                    
                    # Set normal for this face if we're not using per-vertex normals
                    if normals is None and face_idx in face_normals:
                        glNormal3fv(face_normals[face_idx].astype(np.float32))
                    
                    # Draw each vertex of the face
                    for i, vertex_idx in enumerate(face):
                        if vertex_idx >= len(vertices):
                            continue  # Skip invalid indices
                            
                        # Set per-vertex normal if available
                        if normals is not None and vertex_idx < len(normals):
                            glNormal3fv(normals[vertex_idx].astype(np.float32))
                        
                        # Set texture coordinates if available
                        if has_texture and texture_coords is not None:
                            try:
                                # Approach 1: Direct indexing if we have per-vertex texture coords
                                if vertex_idx < len(texture_coords):
                                    glTexCoord2fv(texture_coords[vertex_idx].astype(np.float32))
                                # Approach 2: Use face index * 3 + vertex within face
                                elif face_idx * 3 + i < len(texture_coords):
                                    tex_idx = face_idx * 3 + i
                                    glTexCoord2fv(texture_coords[tex_idx].astype(np.float32))
                                # Fallback: Use default texture coordinate
                                else:
                                    glTexCoord2f(0.0, 0.0)
                            except Exception as e:
                                print(f"[DEBUG] Error setting texture coordinate: {e}")
                                glTexCoord2f(0.0, 0.0)
                        
                        # Set vertex position
                        glVertex3fv(vertices[vertex_idx].astype(np.float32))
                    
                    face_count += 1
                
                glEnd()
                
                print(f"[DEBUG] Rendered {face_count} faces for material {material_idx}")
                
                # Disable texturing after use
                if has_texture:
                    glDisable(GL_TEXTURE_2D)
            
            # No edge rendering - edges are completely disabled as requested
            
            # Disable normalize after drawing
            glDisable(GL_NORMALIZE)
            
            glEndList()
            
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"[DEBUG] OpenGL error after glEndList for '{mesh_name}': {gluErrorString(error)}")
            else:
                print(f"[DEBUG] Successfully created multi-material display list {display_list} for '{mesh_name}'")
                
            return display_list
            
        except Exception as e:
            print(f"Error creating multi-material display list: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def create_mesh_display_list(self, mesh_data):
        """Create an OpenGL display list for the mesh with optimized rendering and lighting."""
        try:
            # DEBUG Logging
            mesh_name = mesh_data.get('name', 'Unnamed')
            num_vertices = len(mesh_data['vertices']) if 'vertices' in mesh_data else 0
            num_faces = len(mesh_data['faces']) if 'faces' in mesh_data else 0
            # Reduced logging for performance
            # print(f"[DEBUG create_mesh_display_list] Creating list for '{mesh_name}' ({num_vertices} verts, {num_faces} faces)")
            
            # Check for existing OpenGL errors before starting
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"[DEBUG create_mesh_display_list] OpenGL error before glGenLists: {gluErrorString(error)}")
                
            # Generate new display list
            display_list = glGenLists(1)
            if display_list == 0:
                print("Error: Could not create display list")
                return None
                
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            color = mesh_data.get('color', self.default_color)  # Get mesh color or use default
            normals = mesh_data.get('normals') # Get pre-calculated normals if available
            texture_coords = mesh_data.get('texture_coords')
            texture_id = mesh_data.get('texture_id')
            has_texture = mesh_data.get('has_texture', False)

            glNewList(display_list, GL_COMPILE)
            
            # Enable lighting and normalize normals for proper lighting
            glEnable(GL_LIGHTING)
            glEnable(GL_NORMALIZE)  # Important for scaled objects

            # --- ADDED: Texture Handling ---
            if has_texture and texture_id is not None:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                # Set material to white if texture is enabled, so texture colors are dominant
                ambient = [0.2, 0.2, 0.2, 1.0] # Low ambient for textured objects
                diffuse = [0.8, 0.8, 0.8, 1.0] # Higher diffuse for texture brightness
                specular = [0.1, 0.1, 0.1, 1.0] # Low specular unless material specifies otherwise
                shininess = 16.0
            else:
                glDisable(GL_TEXTURE_2D) # Ensure texturing is off if no texture
                # Set material properties with physically based values
                # Ambient is 20% of diffuse for realistic lighting
                ambient = [c * 0.2 for c in color] + [1.0]
                # Diffuse is 70% of the color for main surface illumination
                diffuse = [c * 0.7 for c in color] + [1.0]
                # Specular is neutral (white/gray) for realistic highlights
                specular = [0.5, 0.5, 0.5, 1.0]
                shininess = 32.0
            # --- END ADDED ---
            
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)  # Medium glossiness
            
            # OPTIMIZATION: Pre-calculate face normals only if needed
            face_normals = None
            if normals is None:
                face_normals = self._calculate_face_normals(vertices, faces)
            
            # Draw the mesh with efficient vertex/normal handling
            glBegin(GL_TRIANGLES)
            for face_idx, face in enumerate(faces):
                try:
                    # Set normal
                    if normals is not None:
                        # Use per-vertex normals for smooth shading
                        pass  # Will be set per vertex below
                    elif face_normals is not None and face_idx in face_normals:
                        # Use pre-calculated face normal
                        glNormal3fv(face_normals[face_idx].astype(np.float32))
                    
                    # Draw vertices
                    for i, vertex_idx in enumerate(face):
                        if normals is not None and vertex_idx < len(normals):
                            glNormal3fv(normals[vertex_idx].astype(np.float32))
                        
                        if has_texture and texture_coords is not None and vertex_idx < len(texture_coords):
                            glTexCoord2fv(texture_coords[vertex_idx].astype(np.float32))

                        glVertex3fv(vertices[vertex_idx].astype(np.float32))
                        
                except Exception as e:
                    # Silently skip problematic faces for performance
                    continue
            glEnd()
            
            # Disable normalize after drawing
            glDisable(GL_NORMALIZE)
            if has_texture and texture_id is not None:
                glDisable(GL_TEXTURE_2D)
            
            glEndList()
            
            # Check for OpenGL errors after finishing
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"[DEBUG create_mesh_display_list] OpenGL error after glEndList for '{mesh_name}': {gluErrorString(error)}")
            
            return display_list
            
        except Exception as e:
            print(f"Error creating display list: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_face_normals(self, vertices, faces):
        """Pre-calculate face normals for efficiency"""
        face_normals = {}
        for face_idx, face in enumerate(faces):
            try:
                v1, v2, v3 = [vertices[i] for i in face]
                normal_vec = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal_vec)
                if norm > 1e-10:
                    face_normals[face_idx] = normal_vec / norm
                else:
                    face_normals[face_idx] = np.array([0.0, 1.0, 0.0])
            except:
                face_normals[face_idx] = np.array([0.0, 1.0, 0.0])
        return face_normals

    def draw_meshes(self, selection_mode=False):
        """Draw all imported STL meshes
        
        Args:
            selection_mode: If True, render with selection colors instead of normal material
                            If 'edge', render edges with unique colors for picking.
        """
        # OPTIMIZATION: Skip drawing if no display lists ready
        if not self.display_lists:
            return
        
        # print(f"[STLHandler draw_meshes] Called on ID: {id(self)}, Display Lists: {len(self.display_lists)}") # Existing log
        for i, display_list in enumerate(self.display_lists):
            if i >= len(self.mesh_transforms) or i >= len(self.meshes):
                 # print(f"[DEBUG draw_meshes] Skipping index {i}, data mismatch.")
                 continue # Skip if data is inconsistent
            
            # OPTIMIZATION: Skip if display list not ready yet
            if display_list is None:
                continue

            mesh_data = self.meshes[i]
            transform = self.mesh_transforms[i]
            mesh_name = mesh_data.get('name', f'Mesh_{i}')
            edges = mesh_data.get('edges')

            # DEBUG Logging before drawing
            # print(f"[DEBUG draw_meshes] Drawing '{mesh_name}' (Index: {i}, List ID: {display_list})")
            # print(f"  Transform: Pos={transform['position']}, Orient={transform['orientation']}, Scale={transform['scale']}")

            # Check for errors before pushing matrix
            error = glGetError()
            if error != GL_NO_ERROR:
                # print(f"[DEBUG draw_meshes] OpenGL error before glPushMatrix for '{mesh_name}': {gluErrorString(error)}")
                pass
            glPushMatrix()

            # Apply transform: Position -> Orientation -> Scale
            glTranslatef(transform['position'][0], transform['position'][1], transform['position'][2])

            # Apply orientation (quaternion)
            qx, qy, qz, qw = transform['orientation']
            angle = 2 * math.acos(qw)
            x = qx / math.sqrt(1 - qw*qw) if (1 - qw*qw) > 1e-6 else 1
            y = qy / math.sqrt(1 - qw*qw) if (1 - qw*qw) > 1e-6 else 0
            z = qz / math.sqrt(1 - qw*qw) if (1 - qw*qw) > 1e-6 else 0
            glRotatef(math.degrees(angle), x, y, z)

            glScalef(transform['scale'][0], transform['scale'][1], transform['scale'][2])

            # --- Drawing Logic ---
            if selection_mode == True: # Mesh Selection Mode
                # Draw mesh filled with unique mesh selection color
                glDisable(GL_LIGHTING)
                color = self.mesh_selection_colors[i]
                glColor3f(color[0], color[1], color[2])
                if display_list:
                    glCallList(display_list)
                glEnable(GL_LIGHTING)

            elif selection_mode == 'edge': # Edge Selection Mode
                # Draw only edges with unique colors
                if edges is not None:
                    vertices = mesh_data['vertices'] # <<< ADD: Ensure vertices are available here too
                    glDisable(GL_LIGHTING)
                    glLineWidth(2.0) # Make edges easier to pick
                    glBegin(GL_LINES)
                    num_edges = len(edges)
                    # >>> START MODIFICATION: Encode edge info into color <<<
                    for edge_idx, edge in enumerate(edges):
                        # Encode mesh index 'i' and edge index 'edge_idx' into an ID
                        # Ensure ID fits within 24 bits (RGB)
                        edge_id = (i << 16) | (edge_idx & 0xFFFF) # Mesh index in high bits, edge index in low bits
                        
                        # Convert ID to RGB color (0-255 range)
                        r = (edge_id >> 16) & 0xFF
                        g = (edge_id >> 8) & 0xFF
                        b = edge_id & 0xFF
                        
                        # Store mapping for later decoding
                        # Key: integer ID, Value: tuple (mesh_index, edge_index)
                        self.edge_selection_colors[edge_id] = (i, edge_idx)
                        
                        color_float = (r / 255.0, g / 255.0, b / 255.0)
                        glColor3fv(color_float)
                        # >>> END MODIFICATION <<<

                        # --- FIX: Use coordinates directly from 'edge' --- 
                        # 'edge' now holds [[x1,y1,z1], [x2,y2,z2]]
                        if edge.shape == (2, 3):
                            vertex1 = edge[0] # Already the coordinates
                            vertex2 = edge[1] # Already the coordinates

                            # Color is set above based on ID
                            glVertex3fv(vertex1) # Pass coordinates
                            glVertex3fv(vertex2) # Pass coordinates
                        else:
                             print(f"Warning: Edge index {edge_idx} has unexpected shape {edge.shape} for mesh {i} (Edge Select Mode). Expected (2, 3).")
                        # --- END FIX ---
                    glEnd()
                    glEnable(GL_LIGHTING)
                # Do not draw filled mesh in edge selection mode

            else: # Normal Rendering Mode
                # Set material based on selection state and custom color
                # --- Check if edge interaction is enabled BEFORE drawing filled mesh --- 
                edge_interaction_on = hasattr(self.robot_viewer, 'edge_interaction_enabled') and self.robot_viewer.edge_interaction_enabled
                # --- End Check ---
                # print(f"[DEBUG draw_meshes] Edge Interaction Mode: {edge_interaction_on}") # <<< Add Print

                if not edge_interaction_on: # <<< Only draw filled mesh if edge interaction is OFF
                    if i == self.selected_mesh_index:
                        color = mesh_data.get('color', self.default_color)
                        darker_color = [c * 0.5 for c in color]
                        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [darker_color[0] * 0.2, darker_color[1] * 0.2, darker_color[2] * 0.2, 1.0])
                        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [darker_color[0] * 0.6, darker_color[1] * 0.6, darker_color[2] * 0.6, 1.0])
                        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
                    else:
                        color = mesh_data.get('color', self.default_color)
                        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [color[0] * 0.2, color[1] * 0.2, color[2] * 0.2, 1.0])
                        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, 1.0])
                        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])

                    # Draw the filled mesh <<< Moved inside conditional block
                    if display_list:
                        glCallList(display_list)

                    # --- ADDED: Face highlight for plane-pick mode ---
                    hovered_face = getattr(self.robot_viewer, 'hovered_face_info', None)
                    if getattr(self.robot_viewer, 'plane_pick_mode_enabled', False) and hovered_face and hovered_face[0] == i:
                        f_idx = hovered_face[1]
                        coplanar_faces = hovered_face[2] if len(hovered_face) > 2 and hovered_face[2] else [f_idx]
                        if 'faces' in mesh_data and 'vertices' in mesh_data:
                            faces = mesh_data['faces']
                            verts = mesh_data['vertices']
                            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT)
                            glDisable(GL_LIGHTING)
                            glDisable(GL_DEPTH_TEST) # Ensure it renders reliably over the face
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            
                            glColor4f(0.0, 1.0, 1.0, 0.6) # Cyan translucent highlight
                            glBegin(GL_TRIANGLES)
                            for c_idx in coplanar_faces:
                                if 0 <= c_idx < len(faces) and len(faces[c_idx]) >= 3:
                                    for vi in faces[c_idx]:
                                        if vi < len(verts):
                                            glVertex3fv(verts[vi].astype(np.float32))
                            glEnd()
                            
                            glColor4f(0.0, 1.0, 1.0, 1.0)
                            glLineWidth(3.0)
                            for c_idx in coplanar_faces:
                                if 0 <= c_idx < len(faces) and len(faces[c_idx]) >= 3:
                                    glBegin(GL_LINE_LOOP)
                                    for vi in faces[c_idx]:
                                        if vi < len(verts):
                                            glVertex3fv(verts[vi].astype(np.float32))
                                    glEnd()
                            glPopAttrib()
                                
                    # --- ADDED: Multi-face highlight support ---
                    selected_infos = getattr(self.robot_viewer, 'selected_face_infos', [])
                    # Support legacy single selection if list is empty but info exists
                    legacy_info = getattr(self.robot_viewer, 'selected_face_info', None)
                    if not selected_infos and legacy_info:
                        selected_infos = [legacy_info]

                    for current_sel in selected_infos:
                        if current_sel and current_sel[0] == i:
                            f_idx = current_sel[1]
                            coplanar_faces = current_sel[2] if len(current_sel) > 2 and current_sel[2] else [f_idx]
                            if 'faces' in mesh_data and 'vertices' in mesh_data:
                                faces = mesh_data['faces']
                                verts = mesh_data['vertices']
                                glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT)
                                glDisable(GL_LIGHTING)
                                glDisable(GL_DEPTH_TEST) 
                                glEnable(GL_BLEND)
                                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                                
                                glColor4f(0.0, 0.8, 1.0, 0.8) # Prominent cyan
                                glBegin(GL_TRIANGLES)
                                for c_idx in coplanar_faces:
                                    if 0 <= c_idx < len(faces) and len(faces[c_idx]) >= 3:
                                        for vi in faces[c_idx]:
                                            if vi < len(verts):
                                                glVertex3fv(verts[vi].astype(np.float32))
                                glEnd()
                                
                                glColor4f(0.0, 0.8, 1.0, 1.0)
                                glLineWidth(4.0)
                                for c_idx in coplanar_faces:
                                    if 0 <= c_idx < len(faces) and len(faces[c_idx]) >= 3:
                                        glBegin(GL_LINE_LOOP)
                                        for vi in faces[c_idx]:
                                            if vi < len(verts):
                                                glVertex3fv(verts[vi].astype(np.float32))
                                        glEnd()
                                glPopAttrib()
                    # --- END ADDED ---
                    # --- END ADDED ---

                # --- ADD DEBUG LOGGING ---
                # print(f"[DEBUG Edges] Checking mesh: '{mesh_name}', comparison result: {mesh_name != 'Conveyor_belt.stl'}")
                # --- END DEBUG LOGGING ---

                # Draw edges on top of the filled mesh (or alone if edge interaction is on)
                # --- FIX: Check for 'Conveyor_belt' without .stl ---
                if edges is not None and mesh_name != "Conveyor_belt":
                # --- END FIX ---
                    # OPTIMIZATION: Extract edges on-demand if not already done
                    if i not in self.edges_extracted and (edges is None or len(edges) == 0):
                        self._extract_edges_for_mesh(i)
                        edges = mesh_data.get('edges')
                    
                    if edges is not None and len(edges) > 0:
                        vertices = mesh_data['vertices'] # Ensure vertices are available
                    glDisable(GL_LIGHTING) # Disable lighting for consistent edge color
                    
                    # --- MODIFIED LOGIC for drawing edges ---
                    edge_interaction_on = hasattr(self.robot_viewer, 'edge_interaction_enabled') and self.robot_viewer.edge_interaction_enabled
                    show_visual_edges = hasattr(self.robot_viewer, 'show_edges_visually') and self.robot_viewer.show_edges_visually

                    # OPTIMIZATION: Batch edge drawing by color/width to reduce state changes
                    edges_by_state = {}  # Group edges by (color, line_width)
                    
                    for edge_idx, edge in enumerate(edges):
                        current_edge_info = (i, edge_idx)
                        is_selected = current_edge_info in self.selected_edges
                        is_hovered = current_edge_info == self.hover_edge_info

                        should_draw_edge = False
                        color_to_use = self.edge_color
                        line_width_to_use = 1.0 # Default minimum

                        if is_selected:
                            should_draw_edge = True
                            color_to_use = self.selected_edge_color
                            line_width_to_use = 8.0
                        elif is_hovered and show_visual_edges:
                            should_draw_edge = True
                            color_to_use = self.hover_edge_color
                            line_width_to_use = 7.0
                        elif show_visual_edges: # Default edge, draw only if show_visual_edges is true
                            should_draw_edge = True
                            color_to_use = self.edge_color
                            line_width_to_use = 4.0 if edge_interaction_on else 2.5
                        
                        if should_draw_edge and edge.shape == (2, 3):
                            # Group edges by rendering state
                            state_key = (tuple(color_to_use), line_width_to_use)
                            if state_key not in edges_by_state:
                                edges_by_state[state_key] = []
                            edges_by_state[state_key].append(edge)
                    
                    # Draw all edges grouped by state (reduces GL state changes)
                    for (color, line_width), edge_list in edges_by_state.items():
                        glColor3fv(color)
                        glLineWidth(line_width)
                        glBegin(GL_LINES)
                        for edge in edge_list:
                            glVertex3fv(edge[0])
                            glVertex3fv(edge[1])
                        glEnd()

                    # Reset line width to default after drawing all edges
                    glLineWidth(1.0)
                    glEnable(GL_LIGHTING) # Re-enable lighting
                    # --- End MODIFIED LOGIC ---

            # --- End Drawing Logic ---

            glPopMatrix()
            # Check for errors after popping matrix
            error = glGetError()
            if error != GL_NO_ERROR:
                # print(f"[DEBUG draw_meshes] OpenGL error after glPopMatrix for '{mesh_name}': {gluErrorString(error)}")
                pass
    

    def set_global_gizmo_visibility(self, visible):
        """Sets the global visibility for all STL gizmos."""
        STLModelHandler.global_gizmo_visible = visible
        if self.robot_viewer:
            self.robot_viewer.update()

    def draw_stl_gizmo(self, eye_x, eye_y, eye_z, look_at_x, look_at_y, look_at_z):
        """Draw gizmo for the currently selected STL mesh with different colors"""
        if not STLModelHandler.global_gizmo_visible:
            return
        # ADDED: Check if edge interaction is enabled in the viewer
        if hasattr(self.robot_viewer, 'edge_interaction_enabled') and self.robot_viewer.edge_interaction_enabled:
            return # Don't draw STL gizmo if edge mode is active

        # Check if we have a valid mesh selected
        if self.selected_mesh_index < 0 or self.selected_mesh_index >= len(self.meshes):
            return
        
        # Check gizmo state - return immediately if hidden
        if self.gizmo_state == 2:  # 2 = hidden
            return
            
        # Get the transform of the selected mesh
        transform = self.mesh_transforms[self.selected_mesh_index]
        position = transform['position']
        
        # Calculate view direction
        view_dir = np.array([look_at_x - eye_x, look_at_y - eye_y, look_at_z - eye_z])
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # Calculate gizmo scale based on distance
        distance = np.sqrt((eye_x - position[0])**2 + (eye_y - position[1])**2 + (eye_z - position[2])**2)
        scale_factor = distance * 0.08  # Scale gizmo based on distance
        
        # Get orientation
        qx, qy, qz, qw = transform['orientation']
        
        # Create rotation matrix from quaternion
        rot_matrix = np.array([
            [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)]
        ], dtype=np.float32)
        
        # Get local axes
        x_axis = rot_matrix @ np.array([1, 0, 0], dtype=np.float32)
        y_axis = rot_matrix @ np.array([0, 1, 0], dtype=np.float32)
        z_axis = rot_matrix @ np.array([0, 0, 1], dtype=np.float32)
        
        # Draw the appropriate gizmo with custom colors
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Draw gizmo based on mode
        if self.gizmo_state == 0:  # Translation mode
            self.draw_translation_gizmo(position, x_axis, y_axis, z_axis, scale_factor)
        elif self.gizmo_state == 1:  # Rotation mode
            self.draw_rotation_gizmo(position, x_axis, y_axis, z_axis, scale_factor)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def draw_translation_gizmo(self, position, x_axis, y_axis, z_axis, scale_factor):
        """Draw translation gizmo with custom colors"""
        # Position vector as numpy array
        pos = np.array([position[0], position[1], position[2]], dtype=np.float32)
        
        # Draw axes
        glLineWidth(2.0)
        
        # X axis - Cyan
        self.draw_gizmo_axis(pos, x_axis * scale_factor, (0, 1, 1), 
                           selected=self.selected_axis=='x',
                           hovered=self.hover_axis=='x')
        
        # Y axis - Magenta
        self.draw_gizmo_axis(pos, y_axis * scale_factor, (1, 0, 1), 
                           selected=self.selected_axis=='y',
                           hovered=self.hover_axis=='y')
        
        # Z axis - Yellow
        self.draw_gizmo_axis(pos, z_axis * scale_factor, (1, 1, 0), 
                           selected=self.selected_axis=='z',
                           hovered=self.hover_axis=='z')
    
    def draw_gizmo_axis(self, origin, direction, color, selected=False, hovered=False):
        """Draw a single gizmo axis with arrow"""
        if selected:
            glColor3f(1.0, 1.0, 0.0)  # Yellow for selected
        elif hovered:
            glColor3f(1.0, 1.0, 1.0)  # White for hover
        else:
            glColor3fv(color)
            
        # Calculate direction unit vector and length
        direction_len = np.linalg.norm(direction)
        if direction_len < 1e-6:
            return  # Skip if direction has no length
            
        direction_unit = direction / direction_len
        
        # Draw line
        glBegin(GL_LINES)
        glVertex3fv(origin)
        end_point = origin + direction * 0.7  # Shorten line for arrow (previously 0.8)
        glVertex3fv(end_point)
        glEnd()
        
        # Draw arrow head
        self.draw_cone(end_point, direction_unit, direction_len * 0.3, color, selected, hovered)  # Bigger arrow (previously 0.2)
    
    def draw_cone(self, pos, direction, length, color, selected=False, hovered=False):
        """Draw cone for arrow head"""
        if selected:
            glColor3f(1.0, 1.0, 0.0)  # Yellow for selected
        elif hovered:
            glColor3f(1.0, 1.0, 1.0)  # White for hover
        else:
            glColor3fv(color)
            
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        
        # Orient cone along direction
        # For that we need to create a rotation from [0,0,1] to direction
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        axis_len = np.linalg.norm(axis)
        
        if axis_len > 1e-6:
            # We have a valid rotation axis
            angle = np.arccos(np.dot(z_axis, direction)) * 180.0 / np.pi
            glRotatef(angle, axis[0], axis[1], axis[2])
        else:
            # Special case: direction is parallel to z-axis
            if direction[2] < 0:
                glRotatef(180, 1, 0, 0)  # Flip if pointing down
        
        # Draw cone
        quad = gluNewQuadric()
        radius = length * 0.35  # Increased cone radius (previously 0.25)
        gluCylinder(quad, radius, 0, length, 8, 1)
        gluDeleteQuadric(quad)
        
        glPopMatrix()
    
    def draw_rotation_gizmo(self, position, x_axis, y_axis, z_axis, scale_factor):
        """Draw rotation gizmo with rings for each axis"""
        # Make rotation gizmo smaller
        rotation_scale_factor = scale_factor * 0.7
        
        glDisable(GL_LIGHTING)
        glPushMatrix()
        
        # Position at target
        glTranslatef(position[0], position[1], position[2])
        
        # Set up colors for each axis based on hover/selected state
        # X axis (Cyan)
        if self.selected_axis == 'x_rot':
            x_color = (0.5, 1.0, 1.0)  # Lighter cyan for selected
        elif self.hover_axis == 'x_rot':
            x_color = (0.7, 1.0, 1.0)  # Even lighter cyan for hover
        else:
            x_color = (0.0, 1.0, 1.0)  # Cyan for X axis
        
        # Y axis (Magenta)
        if self.selected_axis == 'y_rot':
            y_color = (1.0, 0.5, 1.0)  # Lighter magenta for selected
        elif self.hover_axis == 'y_rot':
            y_color = (1.0, 0.7, 1.0)  # Even lighter magenta for hover
        else:
            y_color = (1.0, 0.0, 1.0)  # Magenta for Y axis
        
        # Z axis (Yellow)
        if self.selected_axis == 'z_rot':
            z_color = (1.0, 1.0, 0.5)  # Lighter yellow for selected
        elif self.hover_axis == 'z_rot':
            z_color = (1.0, 1.0, 0.7)  # Even lighter yellow for hover
        else:
            z_color = (1.0, 1.0, 0.0)  # Yellow for Z axis
        
        # Draw rotation rings
        self.draw_rotation_ring(x_axis, rotation_scale_factor, x_color, 'x_rot')
        self.draw_rotation_ring(y_axis, rotation_scale_factor, y_color, 'y_rot')
        self.draw_rotation_ring(z_axis, rotation_scale_factor, z_color, 'z_rot')
        
        glPopMatrix()
        glEnable(GL_LIGHTING)
    
    def draw_rotation_ring(self, axis, scale_factor, color, axis_name):
        """Draw a rotation ring around the given axis"""
        r, g, b = color
        glColor3f(r, g, b)
        
        # Number of segments for the ring
        segments = 64
        radius = scale_factor * 1.3
        
        # Adjust line width based on state
        if self.selected_axis == axis_name:
            glLineWidth(6.0)  # Thicker for selected
        elif self.hover_axis == axis_name:
            glLineWidth(5.0)  # Slightly thicker for hover
        else:
            glLineWidth(3.0)  # Normal width
        
        # Calculate rotation to align ring with axis
        glPushMatrix()
        
        # Align the ring with the axis
        z_axis = np.array([0, 0, 1], dtype=np.float32)
        axis_normalized = axis / np.linalg.norm(axis)
        
        # Calculate rotation axis and angle
        rotation_axis = np.cross(z_axis, axis_normalized)
        rotation_axis_len = np.linalg.norm(rotation_axis)
        
        if rotation_axis_len > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, axis_normalized), -1.0, 1.0)) * 180.0 / np.pi
            glRotatef(angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        elif axis_normalized[2] < 0:
            glRotatef(180, 1, 0, 0)
        
        # Draw the ring
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()
        
        glPopMatrix()
        glLineWidth(1.0)  # Reset line width
    
    def get_hovered_axis(self, ray_origin, ray_dir):
        """Detect which gizmo axis is under the ray"""
        # Check our own gizmo state and mesh selection
        if self.gizmo_state == 2 or self.selected_mesh_index < 0:  # Hidden or no mesh selected
            return None
            
        transform = self.mesh_transforms[self.selected_mesh_index]
        position = transform['position']
        
        # --- ADDED: Calculate scale factor based on camera distance ---
        # We need camera position to calculate distance-based scale
        # Assuming robot_viewer has camera attributes
        if hasattr(self.robot_viewer, 'camera_distance'):
             eye_x = self.robot_viewer.camera_distance * np.cos(np.radians(self.robot_viewer.camera_elevation)) * np.cos(np.radians(self.robot_viewer.camera_azimuth)) + self.robot_viewer.camera_offset_x
             eye_y = self.robot_viewer.camera_distance * np.cos(np.radians(self.robot_viewer.camera_elevation)) * np.sin(np.radians(self.robot_viewer.camera_azimuth)) + self.robot_viewer.camera_offset_y
             eye_z = self.robot_viewer.camera_distance * np.sin(np.radians(self.robot_viewer.camera_elevation)) + self.robot_viewer.camera_offset_z
             distance = np.sqrt((eye_x - position[0])**2 + (eye_y - position[1])**2 + (eye_z - position[2])**2)
             scale_factor = distance * 0.08 # Use same logic as draw_stl_gizmo
        else:
             scale_factor = 0.1 # Fallback if camera info isn't available
        # --- END ADDED ---
        
        # Get orientation
        qx, qy, qz, qw = transform['orientation']
        
        # Create rotation matrix from quaternion
        rot_matrix = np.array([
            [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)]
        ], dtype=np.float32)
        
        # Get local axes
        x_axis = rot_matrix @ np.array([1, 0, 0], dtype=np.float32)
        y_axis = rot_matrix @ np.array([0, 1, 0], dtype=np.float32)
        z_axis = rot_matrix @ np.array([0, 0, 1], dtype=np.float32)
        
        pos = np.array(position)
        
        # Check based on gizmo mode
        if self.gizmo_state == 0:  # Translation mode
            # Calculate distances from ray to each axis
            distance_threshold = 0.03 * scale_factor # Threshold relative to gizmo size
            
            # Check distance to each axis
            dist_x = self.distance_from_ray_to_line(ray_origin, ray_dir, pos, x_axis, scale_factor)
            dist_y = self.distance_from_ray_to_line(ray_origin, ray_dir, pos, y_axis, scale_factor)
            dist_z = self.distance_from_ray_to_line(ray_origin, ray_dir, pos, z_axis, scale_factor)
            
            # Find the closest axis that's within threshold
            min_dist = float('inf')
            closest_axis = None
            
            if dist_x < distance_threshold and dist_x < min_dist:
                min_dist = dist_x
                closest_axis = 'x'
                
            if dist_y < distance_threshold and dist_y < min_dist:
                min_dist = dist_y
                closest_axis = 'y'
                
            if dist_z < distance_threshold and dist_z < min_dist:
                min_dist = dist_z
                closest_axis = 'z'
                
            return closest_axis
            
        elif self.gizmo_state == 1:  # Rotation mode
            # Check distance to rotation rings
            rotation_scale_factor = scale_factor * 0.7
            radius = rotation_scale_factor * 1.3
            distance_threshold = 0.05 * scale_factor  # Slightly larger threshold for rings
            
            # Check each rotation ring
            dist_x_rot = self.distance_from_ray_to_ring(ray_origin, ray_dir, pos, x_axis, radius)
            dist_y_rot = self.distance_from_ray_to_ring(ray_origin, ray_dir, pos, y_axis, radius)
            dist_z_rot = self.distance_from_ray_to_ring(ray_origin, ray_dir, pos, z_axis, radius)
            
            # Find the closest ring that's within threshold
            min_dist = float('inf')
            closest_axis = None
            
            if dist_x_rot < distance_threshold and dist_x_rot < min_dist:
                min_dist = dist_x_rot
                closest_axis = 'x_rot'
                
            if dist_y_rot < distance_threshold and dist_y_rot < min_dist:
                min_dist = dist_y_rot
                closest_axis = 'y_rot'
                
            if dist_z_rot < distance_threshold and dist_z_rot < min_dist:
                min_dist = dist_z_rot
                closest_axis = 'z_rot'
                
            return closest_axis
            
        return None
    
    def distance_from_ray_to_line(self, ray_origin, ray_dir, line_start, line_dir, line_length=1.0):
        """Calculate the distance from a ray to a line segment
        
        Args:
            ray_origin: Origin of the ray
            ray_dir: Direction of the ray
            line_start: Start point of the line segment
            line_dir: Direction of the line segment
            line_length: Length of the line segment
            
        Returns:
            float: Distance from the ray to the line segment, or infinity if no valid intersection
        """
        # Calculate the shortest distance between the ray and the line
        # This is the distance between the ray origin and the closest point on the line
        
        # First normalize the direction vectors
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        line_dir = line_dir / np.linalg.norm(line_dir)
        
        # Cross product of the two directions
        cross = np.cross(ray_dir, line_dir)
        
        # If parallel, use point-to-line distance
        if np.linalg.norm(cross) < 1e-10:
            # Project the vector from ray origin to line start onto the line direction
            v = line_start - ray_origin
            proj = np.dot(v, line_dir) * line_dir
            closest = line_start - proj
            distance = np.linalg.norm(closest - ray_origin)
            return distance
        
        # Otherwise, calculate the distance
        v = line_start - ray_origin
        distance = abs(np.dot(v, cross)) / np.linalg.norm(cross)
        
        # Now check if the closest point is within the line segment
        # For this we need to solve the parametric equations of both lines
        # and check if the solution is within bounds
        
        # Matrix to solve the system
        M = np.array([
            [np.dot(ray_dir, ray_dir), -np.dot(ray_dir, line_dir)],
            [np.dot(ray_dir, line_dir), -np.dot(line_dir, line_dir)]
        ])
        
        # Right-hand side
        b = np.array([
            np.dot(ray_dir, line_start - ray_origin),
            np.dot(line_dir, line_start - ray_origin)
        ])
        
        try:
            # Solve for the parameters
            t_ray, t_line = np.linalg.solve(M, b)
            
            # Check if the intersection is within the line segment
            if 0 <= t_line <= line_length:
                return distance
        except:
            pass
        
        return float('inf')  # Default: no valid intersection
    
    def distance_from_ray_to_ring(self, ray_origin, ray_dir, ring_center, ring_normal, radius):
        """Calculate the distance from a ray to a circular ring
        
        Args:
            ray_origin: Origin of the ray
            ray_dir: Direction of the ray (normalized)
            ring_center: Center point of the ring
            ring_normal: Normal vector of the ring plane (axis of rotation)
            radius: Radius of the ring
            
        Returns:
            float: Distance from the ray to the ring, or infinity if no valid intersection
        """
        # Normalize vectors
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        ring_normal = ring_normal / np.linalg.norm(ring_normal)
        
        # Find intersection of ray with the plane of the ring
        denom = np.dot(ray_dir, ring_normal)
        
        # If ray is parallel to the plane, no intersection
        if abs(denom) < 1e-6:
            return float('inf')
        
        # Calculate intersection point
        t = np.dot(ring_center - ray_origin, ring_normal) / denom
        
        # If intersection is behind the ray origin, no valid intersection
        if t < 0:
            return float('inf')
        
        # Calculate the intersection point
        intersection_point = ray_origin + t * ray_dir
        
        # Calculate distance from intersection point to ring center
        to_intersection = intersection_point - ring_center
        dist_to_center = np.linalg.norm(to_intersection)
        
        # Calculate distance from the ring (difference from radius)
        distance_from_ring = abs(dist_to_center - radius)
        
        return distance_from_ring
   
    def select_mesh(self, index):
        """Set the selected mesh
        
        Args:
            index: Index of the mesh to select
            
        Returns:
            bool: True if selection was successful
        """
        if 0 <= index < len(self.meshes):
            print(f"Selecting mesh at index {index}")
            self.selected_mesh_index = index
            # Make gizmo visible when selecting a mesh
            self.gizmo_state = 0  # 0 = visible (translate mode)
            print(f"Set selected_mesh_index={index}, gizmo_state=0")
            
            # Sync selection to robot_viewer's stl_handler if it exists
            if hasattr(self.robot_viewer, 'stl_handler') and self.robot_viewer.stl_handler != self:
                self.robot_viewer.stl_handler.selected_mesh_index = index
                self.robot_viewer.stl_handler.gizmo_state = 0  # Also make gizmo visible there
                print(f"Synced selection to robot_viewer.stl_handler: selected_mesh_index={index}, gizmo_state=0")
            
            # Update position panel with current mesh position
            if hasattr(self.robot_viewer, 'parent') and self.robot_viewer.parent():
                parent_ui = self.robot_viewer.parent()
                if hasattr(parent_ui, 'position_panel'):
                    mesh_name = self.meshes[index].get('name', f'mesh_{index}')
                    mesh_obj = self.meshes[index]
                    print(f"[STL Handler] Notifying position panel of selection: {mesh_name}")
                    parent_ui.position_panel.set_object(mesh_obj, 'stl', mesh_name)
            
            # Force a redraw to show the gizmo
            if self.robot_viewer:
                self.robot_viewer.update()
                
            return True
        return False
    
    def toggle_gizmo_mode(self):
        """Toggle gizmo mode: translate -> rotate -> hidden -> translate"""
        # Only toggle if this is the selected robot's STL handler
        if self.robot_viewer.stl_handler == self:
            # Cycle through modes: translate (0) -> rotate (1) -> hidden (2) -> translate (0)
            if self.gizmo_state == 0:  # translate -> rotate
                self.gizmo_state = 1
                self.gizmo_mode = "rotate"
            elif self.gizmo_state == 1:  # rotate -> hidden
                self.gizmo_state = 2
            else:  # hidden -> translate
                self.gizmo_state = 0
                self.gizmo_mode = "translate"
                
            # Force a redraw to update the view
            if self.robot_viewer:
                self.robot_viewer.update()
    
    def handle_gizmo_manipulation(self, dx, dy, screen_z_dir):
        """Handle manipulation of the gizmo based on mouse movement"""
        if self.selected_mesh_index < 0 or not self.is_manipulating_gizmo or not self.selected_axis:
            return False
        
        # Get the transform of the selected mesh
        transform = self.mesh_transforms[self.selected_mesh_index]
        
        # Get camera information for proper screen-to-world projection
        if not hasattr(self.robot_viewer, 'camera_distance'):
            return False
            
        # Calculate camera position
        eye_x = self.robot_viewer.camera_distance * np.cos(np.radians(self.robot_viewer.camera_elevation)) * np.cos(np.radians(self.robot_viewer.camera_azimuth)) + self.robot_viewer.camera_offset_x
        eye_y = self.robot_viewer.camera_distance * np.cos(np.radians(self.robot_viewer.camera_elevation)) * np.sin(np.radians(self.robot_viewer.camera_azimuth)) + self.robot_viewer.camera_offset_y
        eye_z = self.robot_viewer.camera_distance * np.sin(np.radians(self.robot_viewer.camera_elevation)) + self.robot_viewer.camera_offset_z
        
        # Get target position (look at point) with offset
        look_at_x = self.robot_viewer.camera_target.x() + self.robot_viewer.camera_offset_x
        look_at_y = self.robot_viewer.camera_target.y() + self.robot_viewer.camera_offset_y
        look_at_z = self.robot_viewer.camera_target.z() + self.robot_viewer.camera_offset_z
        
        # Calculate camera vectors
        camera_pos = np.array([eye_x, eye_y, eye_z])
        target_pos = np.array([look_at_x, look_at_y, look_at_z])
        
        # Camera forward vector (from camera to target)
        camera_forward = target_pos - camera_pos
        camera_forward = camera_forward / np.linalg.norm(camera_forward)
        
        # Camera right vector
        world_up = np.array([0, 0, 1])
        camera_right = np.cross(camera_forward, world_up)
        camera_right = camera_right / np.linalg.norm(camera_right)
        
        # Camera up vector
        camera_up = np.cross(camera_right, camera_forward)
        camera_up = camera_up / np.linalg.norm(camera_up)
        
        # Get orientation for the mesh's local axes
        qx, qy, qz, qw = transform['orientation']
        rot_matrix = np.array([
            [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)]
        ], dtype=np.float32)
        
        # Get local axes
        x_axis = rot_matrix @ np.array([1, 0, 0], dtype=np.float32)
        y_axis = rot_matrix @ np.array([0, 1, 0], dtype=np.float32)
        z_axis = rot_matrix @ np.array([0, 0, 1], dtype=np.float32)
        
        # Scale factor for movement sensitivity
        base_scale = 0.003
        
        # Calculate distance-based scaling for consistent movement speed
        mesh_pos = transform['position']
        distance_to_camera = np.linalg.norm(camera_pos - mesh_pos)
        distance_scale = distance_to_camera * 0.001  # Adjust this multiplier as needed
        scale = base_scale + distance_scale
        
        # Apply movement based on selected axis with proper camera-relative projection
        if self.selected_axis == 'x':
            # Project screen movement onto the gizmo's X axis
            # Calculate how much the axis aligns with camera right and up vectors
            right_component = np.dot(x_axis, camera_right)
            up_component = np.dot(x_axis, camera_up)
            
            # Calculate movement along the axis based on screen movement
            movement_amount = (dx * right_component - dy * up_component) * scale
            transform['position'] += x_axis * movement_amount
            
        elif self.selected_axis == 'y':
            # Project screen movement onto the gizmo's Y axis
            right_component = np.dot(y_axis, camera_right)
            up_component = np.dot(y_axis, camera_up)
            
            movement_amount = (dx * right_component - dy * up_component) * scale
            transform['position'] += y_axis * movement_amount
            
        elif self.selected_axis == 'z':
            # Project screen movement onto the gizmo's Z axis
            right_component = np.dot(z_axis, camera_right)
            up_component = np.dot(z_axis, camera_up)
            
            movement_amount = (dx * right_component - dy * up_component) * scale
            transform['position'] += z_axis * movement_amount
            
        elif self.selected_axis in ['x_rot', 'y_rot', 'z_rot']:
            # Handle rotation
            rotation_scale = 0.5  # Degrees per pixel
            angle = dx * rotation_scale
            
            # Determine which axis to rotate around
            if self.selected_axis == 'x_rot':
                rotation_axis = x_axis
            elif self.selected_axis == 'y_rot':
                rotation_axis = y_axis
            else:  # z_rot
                rotation_axis = z_axis
            
            # Create rotation quaternion
            angle_rad = np.radians(angle)
            half_angle = angle_rad / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            
            # Rotation quaternion [x, y, z, w]
            rot_quat = np.array([
                rotation_axis[0] * sin_half,
                rotation_axis[1] * sin_half,
                rotation_axis[2] * sin_half,
                cos_half
            ], dtype=np.float32)
            
            # Multiply quaternions: new_orientation = rot_quat * current_orientation
            # Quaternion multiplication formula
            q1 = rot_quat  # [x, y, z, w]
            q2 = transform['orientation']  # [x, y, z, w]
            
            new_quat = np.array([
                q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1],  # x
                q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0],  # y
                q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3],  # z
                q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]   # w
            ], dtype=np.float32)
            
            # Normalize the quaternion
            norm = np.linalg.norm(new_quat)
            if norm > 1e-6:
                new_quat = new_quat / norm
            
            transform['orientation'] = new_quat
        
        # Update the popup values if it exists
        if hasattr(self, 'translation_popup') and self.translation_popup is not None:
            self.translation_popup.findChild(QLineEdit, 'x_input').setText(f"{transform['position'][0]:.2f}")
            self.translation_popup.findChild(QLineEdit, 'y_input').setText(f"{transform['position'][1]:.2f}")
            self.translation_popup.findChild(QLineEdit, 'z_input').setText(f"{transform['position'][2]:.2f}")
        
        # Update position panel if it exists and this mesh is selected
        if hasattr(self.robot_viewer, 'parent') and self.robot_viewer.parent():
            parent_ui = self.robot_viewer.parent()
            if hasattr(parent_ui, 'position_panel'):
                mesh_name = self.meshes[self.selected_mesh_index].get('name', f'mesh_{self.selected_mesh_index}')
                print(f"[STL Handler] Checking position panel update: mesh_name={mesh_name}, panel_type={parent_ui.position_panel.current_object_type}, panel_id={parent_ui.position_panel.current_object_id}")
                if (parent_ui.position_panel.current_object_type == 'stl' and 
                    parent_ui.position_panel.current_object_id == mesh_name):
                    # Convert quaternion to euler angles for rotation
                    from scipy.spatial.transform import Rotation as R_scipy
                    qx, qy, qz, qw = transform['orientation']
                    r = R_scipy.from_quat([qx, qy, qz, qw])
                    euler = r.as_euler('xyz', degrees=True)
                    
                    # Convert numpy arrays to Python lists
                    pos_list = [float(p) for p in transform['position']]
                    euler_list = [float(e) for e in euler]
                    
                    print(f"[STL Handler] Calling position panel update: pos={pos_list}, euler={euler_list}")
                    parent_ui.position_panel.update_from_viewer(pos_list, euler_list)
                else:
                    print(f"[STL Handler] Position panel not updated - object mismatch")
        
        return True
    
    def get_mesh_rotation(self):
        """
        Get the rotation of the selected mesh as Euler angles (in degrees)
        Returns:
            tuple: (x_rotation, y_rotation, z_rotation) in degrees
                   or None if no mesh is selected
        """
        if self.selected_mesh_index < 0 or self.selected_mesh_index >= len(self.mesh_transforms):
            print(f"Invalid mesh index in get_mesh_rotation: {self.selected_mesh_index}")
            return None
            
        # Get the quaternion from the transform
        try:
            qx, qy, qz, qw = self.mesh_transforms[self.selected_mesh_index]['orientation']
            
            # Convert quaternion to Euler angles (in radians)
            # Formula from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
            
            # Check for singularities (gimbal lock)
            test = qx*qy + qz*qw
            if test > 0.499:  # North pole singularity
                yaw = 2 * math.atan2(qx, qw)
                pitch = math.pi/2
                roll = 0
            elif test < -0.499:  # South pole singularity
                yaw = -2 * math.atan2(qx, qw)
                pitch = -math.pi/2
                roll = 0
            else:
                sqx, sqy, sqz, sqw = qx*qx, qy*qy, qz*qz, qw*qw
                yaw = math.atan2(2.0 * (qy*qw - qx*qz), sqw + sqx - sqy - sqz)
                pitch = math.asin(2.0 * test)
                roll = math.atan2(2.0 * (qx*qw - qy*qz), sqw - sqx - sqy + sqz)
                
            # Convert to degrees
            x_rot = math.degrees(roll)
            y_rot = math.degrees(pitch)
            z_rot = math.degrees(yaw)
            
            print(f"Retrieved rotation: {x_rot:.1f}, {y_rot:.1f}, {z_rot:.1f} degrees from quaternion [{qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f}]")
            
            return (x_rot, y_rot, z_rot)
            
        except Exception as e:
            print(f"Error in get_mesh_rotation: {str(e)}")
            import traceback
            traceback.print_exc()
            return (0.0, 0.0, 0.0)
    
    def set_mesh_rotation(self, x_rot, y_rot, z_rot):
        """
        Set the rotation for the selected mesh
        Args:
            x_rot (float): X-axis rotation in degrees
            y_rot (float): Y-axis rotation in degrees
            z_rot (float): Z-axis rotation in degrees
        """
        if self.selected_mesh_index < 0 or self.selected_mesh_index >= len(self.meshes):
            return False
        
        # Convert Euler angles to quaternion
        x_rad = math.radians(x_rot)
        y_rad = math.radians(y_rot)
        z_rad = math.radians(z_rot)
        
        # Calculate quaternion components from Euler angles
        # This is a simplified conversion - in practice, might need more robust conversion
        cx = math.cos(x_rad / 2)
        sx = math.sin(x_rad / 2)
        cy = math.cos(y_rad / 2)
        sy = math.sin(y_rad / 2)
        cz = math.cos(z_rad / 2)
        sz = math.sin(z_rad / 2)
        
        qw = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
        
        # Normalize quaternion
        magnitude = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qx /= magnitude
        qy /= magnitude
        qz /= magnitude
        qw /= magnitude
        
        # Update the mesh's transform
        self.mesh_transforms[self.selected_mesh_index]['orientation'] = np.array([qx, qy, qz, qw], dtype=np.float32)
        
        # Force a redraw
        if self.robot_viewer:
            self.robot_viewer.update()
            
        return True
        
    def apply_rotation_transformation(self, mesh_index, x_rot, y_rot, z_rot):
        """
        Apply rotation transformation to a mesh by index
        Args:
            mesh_index: Index of the mesh to transform
            x_rot (float): X-axis rotation in degrees
            y_rot (float): Y-axis rotation in degrees
            z_rot (float): Z-axis rotation in degrees
        """
        if mesh_index < 0 or mesh_index >= len(self.meshes):
            return False
            
        try:
            # Set the rotation using our existing method
            return self.set_mesh_rotation(x_rot, y_rot, z_rot)
        except Exception as e:
            print(f"Error applying rotation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        for display_list in self.display_lists:
            if display_list is not None:
                try:
                    glDeleteLists(int(display_list), 1)
                except:
                    pass 
        # --- ADDED: Clean up textures ---
        if self.textures:
            texture_ids = list(self.textures.values())
            if texture_ids:
                glDeleteTextures(texture_ids)
            self.textures.clear()
            print("Cleaned up OpenGL textures.")
        # --- END ADDED ---

    def get_mesh_at_cursor(self, x, y, width, height):
        """Use color picking to detect which mesh is under the cursor
        
        Args:
            x, y: Cursor position in window coordinates
            width, height: Window dimensions
            
        Returns:
            int: Index of the selected mesh, or -1 if none
        """
        if not self.meshes:
            return -1
            
        # Save the current rendering parameters
        glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT)
        
        # Set up a small viewport for picking
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw meshes with selection colors
        self.draw_meshes(selection_mode=True)
        
        # Read the pixel color at cursor position
        y = height - y - 1  # OpenGL has inverted Y
        pixel = glReadPixels(x, y, 1, 1, GL_RGB, GL_FLOAT)
        r, g, b = pixel[0][0]
        
        # Restore the rendering state
        glPopAttrib()
        
        # Find which mesh matches this color
        for i, color in enumerate(self.mesh_selection_colors):
            # Use a tolerance to account for potential rounding errors
            tolerance = 0.02
            if (abs(r - color[0]) < tolerance and 
                abs(g - color[1]) < tolerance and 
                abs(b - color[2]) < tolerance):
                return i
                
        return -1 

    def delete_model(self, item):
        """Delete a model from both the scene and tree"""
        # Get the main UI for undo tracking
        main_ui = None
        if hasattr(self.robot_viewer, 'parent'):
            main_ui = self.robot_viewer.parent()
            while main_ui and not hasattr(main_ui, 'push_scene_undo'):
                if hasattr(main_ui, 'parent'):
                    main_ui = main_ui.parent()
                else: break
                
        # Capture snapshot before
        before_snapshot = None
        if main_ui and hasattr(main_ui, 'scene_manager'):
            before_snapshot = main_ui.scene_manager.capture_scene_snapshot_json()

        # Get the model index from the item data
        model_index = item.data(0, Qt.UserRole)
        model_name = item.text(0)

        # Validate index
        if model_index is None or not (0 <= model_index < len(self.meshes)):
            self.update_message(f"Error: Invalid index {model_index} for model '{model_name}'")
            return

        # Show a confirmation dialog
        from PyQt5.QtWidgets import QApplication
        main_window = QApplication.activeWindow()
        # Delete without confirmation
        confirm = QMessageBox.Yes

        if confirm == QMessageBox.Yes:
            try:
                # --- Start of deletion logic ---
                # 1. Delete the OpenGL display list
                if model_index < len(self.display_lists) and self.display_lists[model_index] is not None:
                    glDeleteLists(int(self.display_lists[model_index]), 1)
                    del self.display_lists[model_index]
                elif model_index < len(self.display_lists):
                    # Still remove the placeholder if it was None
                    del self.display_lists[model_index]

                # 2. Remove mesh data, transform, selection color, and file path
                # Use pop to remove by index safely
                if model_index < len(self.meshes):
                    del self.meshes[model_index]
                if model_index < len(self.mesh_transforms):
                    del self.mesh_transforms[model_index]
                if model_index < len(self.mesh_selection_colors):
                    del self.mesh_selection_colors[model_index]
                if model_index < len(self.mesh_file_paths):
                    del self.mesh_file_paths[model_index]

                # 3. Adjust selected index if necessary
                if self.selected_mesh_index == model_index:
                    # If the deleted model was selected, select the previous one or none
                    self.selected_mesh_index = max(-1, model_index - 1)
                elif self.selected_mesh_index > model_index:
                    # If a model after the deleted one was selected, decrement its index
                    self.selected_mesh_index -= 1

                # --- End of deletion logic ---
                success = True # Assume success if no exceptions

            except Exception as e:
                print(f"Error deleting model data at index {model_index}: {str(e)}")
                import traceback
                traceback.print_exc()
                success = False

            if success:
                # Remove the item from the tree
                if self.models_item:
                    # Find the item again in case the tree structure changed
                    child_to_remove = None
                    for i in range(self.models_item.childCount()):
                         child = self.models_item.child(i)
                         if child and child.data(0, Qt.UserRole) == model_index:
                              child_to_remove = child
                              break
                    if child_to_remove:
                         self.models_item.takeChild(self.models_item.indexOfChild(child_to_remove))
                    else:
                         # Fallback: Remove by item reference if index lookup fails
                         index_in_parent = self.models_item.indexOfChild(item)
                         if index_in_parent >= 0:
                              self.models_item.takeChild(index_in_parent)

                # Update message
                self.update_message(f"Deleted model: {model_name}")

                # IMPORTANT: Update data for remaining tree items because indices shifted
                if self.models_item:
                     for i in range(self.models_item.childCount()):
                          child_item = self.models_item.child(i)
                          # The new index is simply its position in the list now
                          child_item.setData(0, Qt.UserRole, i)

                # Reselect the current mesh to update gizmo/highlighting
                current_selection = self.selected_mesh_index
                self.select_mesh(current_selection) # This also updates the tree highlighting

                # Force a redraw of the viewer
                if self.robot_viewer:
                    self.robot_viewer.update()
                
                # Capture snapshot after and push to undo
                if before_snapshot and main_ui and hasattr(main_ui, 'push_scene_undo'):
                    after_snapshot = main_ui.scene_manager.capture_scene_snapshot_json()
                    main_ui.push_scene_undo(before_snapshot, after_snapshot, f"Delete STL: {model_name}")
            else:
                self.update_message(f"Failed to delete model: {model_name}")

    def update_stl_models_in_tree(self):
        """Update the Models section in the program tree with current STL models"""
        print("============ update_stl_models_in_tree called ============")
        
        # Check if program_tree is available
        if not hasattr(self, 'program_tree') or self.program_tree is None:
            print("ERROR: No program_tree available in STLModelHandler")
            
            # Try to get program_tree from parent if available
            if hasattr(self.robot_viewer, 'parent') and self.robot_viewer.parent():
                parent = self.robot_viewer.parent()
                if hasattr(parent, 'program_tree') and hasattr(parent, 'models_item'):
                    print("Getting program_tree reference from parent")
                    self.set_program_tree(parent.program_tree, parent.models_item)
                else:
                    print("ERROR: Cannot get program_tree from parent - not found")
                    return False
            else:
                print("ERROR: Cannot get program_tree - no parent found")
                return False
            
        # Double-check after potential recovery
        if not hasattr(self, 'program_tree') or self.program_tree is None:
            print("FATAL ERROR: Still no program_tree available after recovery attempt")
            return False
        
        try:
            # Check if robot viewer is available
            if not self.robot_viewer:
                print("ERROR: No robot_viewer found")
                return False
                
            # Get stl_handler - use self if we have meshes, otherwise try robot_viewer.stl_handler
            stl_handler = None
            
            # Debug print our own meshes
            print(f"DEBUG: Self meshes count: {len(self.meshes) if hasattr(self, 'meshes') else 'no meshes attribute'}")
            
            # If we have meshes, use our own handler
            if hasattr(self, 'meshes') and self.meshes:
                stl_handler = self
                print(f"Using self as stl_handler with {len(self.meshes)} meshes")
                # Make sure our meshes list is up to date by syncing with the robot_viewer
                if hasattr(self.robot_viewer, 'stl_handler') and hasattr(self.robot_viewer.stl_handler, 'meshes'):
                    if len(self.robot_viewer.stl_handler.meshes) > len(self.meshes):
                        print(f"Syncing additional meshes from robot_viewer.stl_handler ({len(self.robot_viewer.stl_handler.meshes)} meshes) to self ({len(self.meshes)} meshes)")
                        self.meshes = self.robot_viewer.stl_handler.meshes.copy()
                        self.mesh_transforms = self.robot_viewer.stl_handler.mesh_transforms.copy() if hasattr(self.robot_viewer.stl_handler, 'mesh_transforms') else []
                        self.mesh_selection_colors = self.robot_viewer.stl_handler.mesh_selection_colors.copy() if hasattr(self.robot_viewer.stl_handler, 'mesh_selection_colors') else []
                        self.display_lists = self.robot_viewer.stl_handler.display_lists.copy() if hasattr(self.robot_viewer.stl_handler, 'display_lists') else []
                        self.selected_mesh_index = rv_handler.selected_mesh_index
                        print(f"After sync in set_program_tree, self has {len(self.meshes)} meshes")
            # Otherwise, try to use the robot_viewer's stl_handler
            elif hasattr(self.robot_viewer, 'stl_handler'):
                if hasattr(self.robot_viewer.stl_handler, 'meshes') and self.robot_viewer.stl_handler.meshes:
                    stl_handler = self.robot_viewer.stl_handler
                    print(f"Using robot_viewer.stl_handler with {len(stl_handler.meshes)} meshes")
                    
                    # Sync meshes from robot_viewer.stl_handler to self
                    self.meshes = stl_handler.meshes.copy() if stl_handler.meshes else []
                    self.mesh_transforms = stl_handler.mesh_transforms.copy() if hasattr(stl_handler, 'mesh_transforms') else []
                    self.mesh_selection_colors = stl_handler.mesh_selection_colors.copy() if hasattr(stl_handler, 'mesh_selection_colors') else []
                    self.display_lists = stl_handler.display_lists.copy() if hasattr(stl_handler, 'display_lists') else []
                    self.selected_mesh_index = stl_handler.selected_mesh_index
                    print(f"Synced {len(self.meshes)} meshes from robot_viewer.stl_handler to self")
            
            # If we still don't have a valid stl_handler with meshes, return failure
            if not stl_handler or not hasattr(stl_handler, 'meshes') or not stl_handler.meshes:
                print("ERROR: No meshes found in stl_handler")
                return False
            
            # Debug print mesh info
            print(f"Found {len(stl_handler.meshes)} meshes in stl_handler")
            for i, mesh in enumerate(stl_handler.meshes):
                print(f"  Mesh {i}: {mesh.get('name', 'unnamed')}")
                
            # Check if models_item exists
            if not self.models_item:
                print("ERROR: No models_item found in STLModelHandler - creating it")
                # Create it if needed
                self.models_item = QTreeWidgetItem(self.program_tree, ["STL Models"])
                self.models_item.setIcon(0, qta.icon('fa5s.cube', color='#444444'))
            else:
                print(f"Found existing models_item: {self.models_item}")
                
            # Make sure selected_mesh_index is valid
            if stl_handler.selected_mesh_index < 0 and len(stl_handler.meshes) > 0:
                stl_handler.selected_mesh_index = 0  # Auto-select first mesh if none selected
                print(f"Auto-selected first mesh (index 0)")
                
                # Sync selection to other handler if needed
                if stl_handler != self:
                    self.selected_mesh_index = stl_handler.selected_mesh_index
                if hasattr(self.robot_viewer, 'stl_handler') and self.robot_viewer.stl_handler != stl_handler:
                    self.robot_viewer.stl_handler.selected_mesh_index = stl_handler.selected_mesh_index
            
            # Save mesh count before clearing
            mesh_count = len(stl_handler.meshes)
            
            # Clear existing items to prevent duplicates
            prev_count = self.models_item.childCount()
            self.models_item.takeChildren()
            print(f"Cleared {prev_count} existing items from models_item")
            
            # Add each STL model directly to the tree under models_item
            for i, mesh in enumerate(stl_handler.meshes):
                model_name = mesh.get('name', f"Model {i+1}")
                print(f"Adding model to tree: {model_name}")
                
                # Create item with icon as direct child of models_item
                model_item = QTreeWidgetItem(self.models_item, [model_name])
                model_item.setIcon(0, qta.icon('fa5s.cube', color='#444444'))
                
                # Store the model index in the item data
                model_item.setData(0, Qt.UserRole, i)
                
                # Highlight the selected mesh
                if i == stl_handler.selected_mesh_index:
                    font = model_item.font(0)
                    font.setBold(True)
                    model_item.setFont(0, font)
                    print(f"Highlighted model {model_name} (index {i}) as selected")
            
            # If no mesh is selected and we have at least one, select the first
            if stl_handler.selected_mesh_index < 0 and len(stl_handler.meshes) > 0:
                stl_handler.selected_mesh_index = 0
                # Ensure we also update our own selected_mesh_index
                self.selected_mesh_index = 0
                # Update the highlighting for the newly selected model
                if self.models_item.childCount() > 0:
                    model_item = self.models_item.child(0)
                    if model_item:
                        font = model_item.font(0)
                        font.setBold(True)
                        model_item.setFont(0, font)
                print(f"Auto-selected first mesh (index 0)")
            
            # Verify that we added the correct number of items to the tree
            if self.models_item.childCount() != mesh_count:
                print(f"WARNING: Mismatch between mesh count ({mesh_count}) and tree items ({self.models_item.childCount()})")
            
            # Make sure the models item is visible
            self.models_item.setExpanded(True)
            self.program_tree.expandItem(self.models_item)
            print(f"Expanded models_item in tree")
            
            # Update message to show that models were added
            message = f"Updated Models list: {len(stl_handler.meshes)} model(s) available"
            print(message)
            self.update_message(message)
                
            # Force a redraw of the tree to ensure items are visible
            self.program_tree.update()
            print(f"Updated tree. Items in tree: {self.models_item.childCount()}")
            
            # Make sure models item is visible
            self.program_tree.scrollToItem(self.models_item)
            print(f"Scrolled to models_item")
            
            # Process pending events to ensure UI is updated
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            print(f"Processed UI events")
            
            # Set gizmo to visible by default - ensures gizmo is shown immediately
            self.gizmo_state = 0  # 0 = visible (translate mode)
            if hasattr(self.robot_viewer, 'stl_handler'):
                self.robot_viewer.stl_handler.gizmo_state = 0  # 0 = visible (translate mode)
                print(f"Set gizmo_state to 0 (visible) in update_stl_models_in_tree")
            
            # Force redraw of the viewer to show any changes
            if self.robot_viewer:
                self.robot_viewer.update()
                print(f"Updated robot viewer")
                
            print("============ update_stl_models_in_tree completed successfully ============")
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: Error updating STL models in tree: {str(e)}")
            return False

    def update_stl_selection_formatting(self):
        """Update bold formatting for all STL model items in the tree based on current selection"""
        if not self.program_tree or not self.models_item:
            return
        
        # Get all currently selected items in the tree
        selected_items = self.program_tree.selectedItems()
        print(f"[DEBUG] update_stl_selection_formatting called with {len(selected_items)} selected items")
        
        # Filter for STL model items only (children of models_item)
        stl_selected_items = [
            item for item in selected_items 
            if (item.parent() == self.models_item and 
                item.data(0, Qt.UserRole) is not None)
        ]
        
        # Get the model indices of selected STL items
        selected_stl_indices = [item.data(0, Qt.UserRole) for item in stl_selected_items]
        print(f"[DEBUG] Selected STL model indices: {selected_stl_indices}")
        
        # Update formatting for all STL model items
        for i in range(self.models_item.childCount()):
            child_item = self.models_item.child(i)
            if child_item:
                model_index = child_item.data(0, Qt.UserRole)
                font = child_item.font(0)
                
                # Make bold if this item is selected, normal if not
                is_selected = model_index in selected_stl_indices
                font.setBold(is_selected)
                child_item.setFont(0, font)
                
                if is_selected:
                    print(f"[DEBUG] Made model '{child_item.text(0)}' (index {model_index}) bold")
                else:
                    print(f"[DEBUG] Made model '{child_item.text(0)}' (index {model_index}) normal weight")

    def show_model_context_menu(self, position):
        """Show context menu for model items"""
        # Get the item at the clicked position
        item = self.program_tree.itemAt(position)
        if not item:
            return
            
        # Check if it's a model item (directly under models_item)
        if self.models_item and item.parent() is not None and item.parent() == self.models_item:
            # Get all selected items
            selected_items = self.program_tree.selectedItems()
            
            # Filter for STL model items only (children of models_item)
            stl_selected_items = [
                selected_item for selected_item in selected_items 
                if (selected_item.parent() == self.models_item and 
                    selected_item.data(0, Qt.UserRole) is not None)
            ]
            
            if len(stl_selected_items) > 1:
                # Multiple items selected - show batch operations menu
                self.show_multiple_model_menu(position, stl_selected_items)
            else:
                # Single item - show normal menu
                model_index = item.data(0, Qt.UserRole)
                self.show_model_item_menu(position, item, model_index)
    
    def show_model_item_menu(self, position, item, model_index):
        """Show the context menu for a model item"""
        # Create the context menu
        menu = QMenu()
        
        # Check if this is a sensor (from library/Sensors/)
        is_sensor = False
        if model_index is not None and 0 <= model_index < len(self.meshes):
            mesh_data = self.meshes[model_index]
            mesh_name = mesh_data.get('name', '')
            # Check if it's from the Sensors library
            if model_index < len(self.mesh_file_paths):
                file_path = self.mesh_file_paths[model_index]
                # Handle None file_path (e.g., for duplicated meshes)
                if file_path is not None:
                    normalized_path = file_path.replace('\\', '/')
                    is_sensor = 'library/Sensors' in normalized_path or 'library/sensors' in normalized_path.lower()
                else:
                    is_sensor = False
        
        # Add sensor-specific actions if this is a sensor
        if is_sensor:
            configure_sensor_action = menu.addAction("Configure Sensor")
            configure_sensor_action.triggered.connect(lambda: self.configure_sensor(item, model_index))
            menu.addSeparator()
        
        # Add Material Properties action (NEW)
        material_action = menu.addAction("Material Properties...")
        material_action.triggered.connect(lambda: self.show_material_properties_dialog(model_index))
        
        menu.addSeparator()
        
        # Add actions
        rename_action = menu.addAction("Rename Model")
        rename_action.triggered.connect(lambda: self.rename_model(item))
        
        # Add Properties action
        properties_action = menu.addAction("Properties")
        properties_action.triggered.connect(lambda: self.show_properties_dialog(item, model_index))
        
        # Add Duplicate action
        duplicate_action = menu.addAction("Duplicate")
        duplicate_action.triggered.connect(lambda: self.duplicate_model(item, model_index))
        
        delete_action = menu.addAction("Delete Model")
        delete_action.triggered.connect(lambda: self.delete_model(item))
        
        # Show the menu at the current position
        menu.exec_(self.program_tree.viewport().mapToGlobal(position))
    
    def show_multiple_model_menu(self, position, selected_items):
        """Show context menu for multiple selected model items"""
        menu = QMenu()
        
        # Add header showing number of selected items
        header_action = menu.addAction(f"Multiple STL Models Selected ({len(selected_items)} items)")
        header_action.setEnabled(False)
        menu.addSeparator()
        
        # Add batch operations
        delete_multiple_action = menu.addAction(f"Delete Selected Models ({len(selected_items)} items)")
        delete_multiple_action.triggered.connect(lambda: self.delete_multiple_models(selected_items))
        
        duplicate_multiple_action = menu.addAction(f"Duplicate Selected Models ({len(selected_items)} items)")
        duplicate_multiple_action.triggered.connect(lambda: self.duplicate_multiple_models(selected_items))
        
        # Show the menu at the current position
        menu.exec_(self.program_tree.viewport().mapToGlobal(position))
    
    def delete_multiple_models(self, selected_items):
        """Delete multiple selected STL models"""
        if not selected_items:
            return
        
        # Confirm deletion with proper styling
        # Delete without confirmation
        msg_box = QMessageBox(self.program_tree)
        msg_box.setWindowTitle("Delete Multiple Models")
        msg_box.setText(f"Deleting {len(selected_items)} selected STL models...")
        msg_box.setStandardButtons(QMessageBox.Yes)
        msg_box.setDefaultButton(QMessageBox.Yes)
        msg_box.setIcon(QMessageBox.Question)
        
        # Apply styling to prevent black dialog and ensure visible button text
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #f0f0f0;
                color: #333333;
            }
            QMessageBox QLabel {
                background-color: transparent;
                color: #333333;
            }
            QMessageBox QPushButton {
                background-color: #e0e0e0;
                color: #000000 !important;
                border: 2px solid #999999;
                border-radius: 4px;
                padding: 8px 20px;
                min-width: 80px;
                font-weight: bold;
                font-size: 12px;
            }
            QMessageBox QPushButton:hover {
                background-color: #d0d0d0;
                color: #000000 !important;
                border-color: #666666;
            }
            QMessageBox QPushButton:pressed {
                background-color: #c0c0c0;
                color: #000000 !important;
                border-color: #333333;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #000000 !important;
                border: 2px solid #999999;
                border-radius: 4px;
                padding: 8px 20px;
                min-width: 80px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                color: #000000 !important;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
                color: #000000 !important;
                border-color: #333333;
            }
        """)
        
        reply = msg_box.exec_()
        
        if reply != QMessageBox.Yes:
            return
        
        # Collect model indices and sort in descending order (delete from end to avoid index shifting)
        model_indices = []
        for item in selected_items:
            model_index = item.data(0, Qt.UserRole)
            if model_index is not None and 0 <= model_index < len(self.meshes):
                model_indices.append((model_index, item))
        
        # Sort by index in descending order
        model_indices.sort(key=lambda x: x[0], reverse=True)
        
        print(f"[DEBUG] Deleting {len(model_indices)} STL models: {[idx for idx, _ in model_indices]}")
        
        # Delete models one by one from highest index to lowest
        deleted_count = 0
        for model_index, item in model_indices:
            try:
                # Delete the model using existing delete_model method
                # But we need to create a temporary method that doesn't show confirmation
                self.delete_model_without_confirmation(item, model_index)
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete model at index {model_index}: {str(e)}")
        
        # Update message
        self.update_message(f"Deleted {deleted_count} STL models")
        
        # Force a redraw
        if self.robot_viewer:
            self.robot_viewer.update()
    
    def delete_model_without_confirmation(self, item, model_index):
        """Delete a model without showing confirmation dialog"""
        model_name = item.text(0)
        
        # Validate index
        if model_index is None or not (0 <= model_index < len(self.meshes)):
            print(f"Error: Invalid index {model_index} for model '{model_name}'")
            return False
        
        try:
            # 1. Delete the OpenGL display list
            if model_index < len(self.display_lists) and self.display_lists[model_index] is not None:
                glDeleteLists(int(self.display_lists[model_index]), 1)
                del self.display_lists[model_index]
            elif model_index < len(self.display_lists):
                del self.display_lists[model_index]

            # 2. Remove mesh data, transform, selection color, and file path
            if model_index < len(self.meshes):
                del self.meshes[model_index]
            if model_index < len(self.mesh_transforms):
                del self.mesh_transforms[model_index]
            if model_index < len(self.mesh_selection_colors):
                del self.mesh_selection_colors[model_index]
            if model_index < len(self.mesh_file_paths):
                del self.mesh_file_paths[model_index]

            # 3. Adjust selected index if necessary
            if self.selected_mesh_index == model_index:
                self.selected_mesh_index = max(-1, model_index - 1)
            elif self.selected_mesh_index > model_index:
                self.selected_mesh_index -= 1

            # 4. Remove from tree
            if self.models_item:
                index_in_parent = self.models_item.indexOfChild(item)
                if index_in_parent >= 0:
                    self.models_item.takeChild(index_in_parent)

            # 5. Update data for remaining tree items
            if self.models_item:
                for i in range(self.models_item.childCount()):
                    child_item = self.models_item.child(i)
                    child_item.setData(0, Qt.UserRole, i)

            return True
            
        except Exception as e:
            print(f"Error deleting model data at index {model_index}: {str(e)}")
            return False
    
    def duplicate_multiple_models(self, selected_items):
        """Duplicate multiple selected STL models"""
        if not selected_items:
            return
        
        print(f"[DEBUG] Duplicating {len(selected_items)} STL models")
        
        # Capture snapshot before
        main_ui = None
        if hasattr(self.robot_viewer, 'parent'):
            main_ui = self.robot_viewer.parent()
            while main_ui and not hasattr(main_ui, 'push_scene_undo'):
                main_ui = main_ui.parent() if hasattr(main_ui, 'parent') else None
                
        before_snapshot = None
        if main_ui and hasattr(main_ui, 'scene_manager'):
            before_snapshot = main_ui.scene_manager.capture_scene_snapshot_json()

        duplicated_count = 0
        for item in selected_items:
            model_index = item.data(0, Qt.UserRole)
            if model_index is not None and 0 <= model_index < len(self.meshes):
                try:
                    self.duplicate_model(item, model_index)
                    duplicated_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to duplicate model at index {model_index}: {str(e)}")
        
        # Update message
        self.update_message(f"Duplicated {duplicated_count} STL models")
        
        # Force a redraw
        if self.robot_viewer:
            self.robot_viewer.update()

        # Capture snapshot after and push to undo
        if before_snapshot and main_ui and hasattr(main_ui, 'push_scene_undo') and duplicated_count > 0:
            after_snapshot = main_ui.scene_manager.capture_scene_snapshot_json()
            main_ui.push_scene_undo(before_snapshot, after_snapshot, f"Duplicate {duplicated_count} STLs")

    
    def duplicate_model(self, item, model_index):
        """Create a duplicate of the selected model"""
        # Determine if we should push undo here (only if NOT called from multi-duplicate)
        # Check current stack depth or use a flag? 
        # Actually, if we just push undo in both, it might be okay, but multi-duplicate is better as one action.
        
        # Check if we are already in a multi-duplicate context (very simple check)
        is_multi = traceback.extract_stack()[-2][2] == 'duplicate_multiple_models'
        
        main_ui = None
        before_snapshot = None
        if not is_multi:
             if hasattr(self.robot_viewer, 'parent'):
                 main_ui = self.robot_viewer.parent()
                 while main_ui and not hasattr(main_ui, 'push_scene_undo'):
                     main_ui = main_ui.parent() if hasattr(main_ui, 'parent') else None
             if main_ui and hasattr(main_ui, 'scene_manager'):
                 before_snapshot = main_ui.scene_manager.capture_scene_snapshot_json()

        if model_index < 0 or model_index >= len(self.meshes):
            return

            
        # Get original mesh data
        original_mesh = self.meshes[model_index]
        
        # Create a copy of the mesh data
        mesh_copy = {
            'vertices': np.copy(original_mesh['vertices']),
            'faces': np.copy(original_mesh['faces']),
            'name': f"{original_mesh.get('name', 'model')}_copy",
            'color': original_mesh.get('color', [0.7, 0.7, 0.7]).copy()
        }
        
        # Add the mesh copy to the list
        self.meshes.append(mesh_copy)
        
        # Create initial transform with slight offset
        original_transform = self.mesh_transforms[model_index]
        new_transform = {
            'position': [
                original_transform['position'][0] + 0.1,  # Offset in X
                original_transform['position'][1] + 0.1,  # Offset in Y
                original_transform['position'][2]         # Same Z
            ],
            'orientation': original_transform['orientation'].copy(),
            'scale': original_transform['scale'].copy()
        }
        self.mesh_transforms.append(new_transform)
        
        # Create selection color
        selection_color = [1, 0, 0]  # Red for selection
        self.mesh_selection_colors.append(selection_color)

        # --- FIX: Track file path for duplicates ---
        original_path = self.mesh_file_paths[model_index] if model_index < len(self.mesh_file_paths) else None
        self.mesh_file_paths.append(original_path)
        # -----------------------------------------
        
        # Create display list for the new mesh
        display_list = self.create_mesh_display_list(mesh_copy)
        self.display_lists.append(display_list)
        
        # Update the tree if available
        if self.program_tree and self.models_item:
            self.update_stl_models_in_tree()

        # Capture snapshot after and push to undo (only for single duplicates)
        if not is_multi and before_snapshot and main_ui and hasattr(main_ui, 'push_scene_undo'):
            after_snapshot = main_ui.scene_manager.capture_scene_snapshot_json()
            main_ui.push_scene_undo(before_snapshot, after_snapshot, f"Duplicate STL: {mesh_copy['name']}")

        
        # Notify the robot viewer's AI controller about the new object
        if self.robot_viewer and hasattr(self.robot_viewer, 'ai_controller'):
            new_index = len(self.meshes) - 1
            self.robot_viewer.ai_controller.register_object(mesh_copy['name'], new_index)
        
        # Update message
        self.update_message(f"Duplicated model: {item.text(0)}")
        
        # Force a redraw
        if self.robot_viewer:
            self.robot_viewer.update()
    
    def show_material_properties_dialog(self, model_index):
        """Show material properties dialog for an STL object"""
        if model_index < 0 or model_index >= len(self.meshes):
            QMessageBox.warning(None, "Error", "Invalid model index")
            return
        
        mesh_data = self.meshes[model_index]
        
        # Show the material properties dialog
        dialog = MaterialPropertiesDialog(mesh_data, model_index, None)
        if dialog.exec_():
            # Get the properties from the dialog
            properties = dialog.get_properties()
            
            # Update the mesh data with material properties
            mesh_data['material'] = properties['material']
            mesh_data['density'] = properties['density']
            mesh_data['mass_override'] = properties['mass_override']
            mesh_data['mass'] = properties['mass']
            mesh_data['friction'] = properties['friction']
            mesh_data['restitution'] = properties['restitution']
            mesh_data['volume'] = properties['volume']
            
            # Log the update
            print(f"[Material Properties] Updated mesh {model_index} ({mesh_data.get('name', 'Unknown')})")
            print(f"  Material: {properties['material']}")
            print(f"  Density: {properties['density']:.1f} kg/m³")
            print(f"  Mass: {properties['mass']:.3f} kg")
            print(f"  Friction: {properties['friction']:.2f}")
            print(f"  Restitution: {properties['restitution']:.2f}")
            
            # Update status message
            self.update_message(
                f"Material set: {properties['material']} ({properties['mass']:.3f} kg)"
            )
            
            # If PyBullet physics is active, update the object's mass
            if self.robot_viewer and hasattr(self.robot_viewer, 'physics_manager'):
                physics_manager = self.robot_viewer.physics_manager
                if physics_manager and hasattr(physics_manager, 'physics_objects'):
                    # Check if this object has a physics body
                    if model_index in physics_manager.physics_objects:
                        phys_obj = physics_manager.physics_objects[model_index]
                        if phys_obj and hasattr(phys_obj, 'body_id'):
                            try:
                                import pybullet as p
                                # Update mass in PyBullet
                                p.changeDynamics(
                                    phys_obj.body_id,
                                    -1,  # Base link
                                    mass=properties['mass'],
                                    lateralFriction=properties['friction'],
                                    restitution=properties['restitution']
                                )
                                print(f"[Material Properties] Updated PyBullet physics for mesh {model_index}")
                            except Exception as e:
                                print(f"[Material Properties] Error updating PyBullet physics: {e}")
    
    def show_properties_dialog(self, item, model_index):
        """Show the properties dialog for a model"""
        if model_index < 0 or model_index >= len(self.meshes):
            return
            
        # Get current color
        current_color = self.meshes[model_index].get('color', self.default_color)
        
        # Convert normalized [0-1] color to [0-255] for QColor
        qcolor = QColor(
            int(current_color[0] * 255),
            int(current_color[1] * 255),
            int(current_color[2] * 255)
        )
        
        # Create color dialog
        color = QColorDialog.getColor(qcolor, self.program_tree, "Select Model Color")
        
        if color.isValid():
            # Convert QColor [0-255] back to normalized [0-1] RGB values
            new_color = [
                color.red() / 255.0,
                color.green() / 255.0,
                color.blue() / 255.0
            ]
            
            print(f"Updating color for {item.text(0)}")
            print(f"Old color: RGB({current_color})")
            print(f"New color: RGB({new_color})")
            
            # Update the mesh color
            self.meshes[model_index]['color'] = new_color
            
            # Update color name based on the new color
            if hasattr(self.robot_viewer, 'ai_controller'):
                color_name = self.robot_viewer.ai_controller.get_closest_color_name(new_color)
                self.meshes[model_index]['color_name'] = color_name
                print(f"Updated color name to: {color_name}")
            
            # Delete old display list
            if model_index < len(self.display_lists):
                old_display_list = self.display_lists[model_index]
                if old_display_list:
                    glDeleteLists(old_display_list, 1)
                
            # Create new display list with updated color
            new_display_list = self.create_mesh_display_list(self.meshes[model_index])
            self.display_lists[model_index] = new_display_list
            
            # Force a redraw
            if self.robot_viewer:
                self.robot_viewer.update()
                
            # Update message
            self.update_message(f"Updated color for {item.text(0)}")
            
            # Re-register objects with AI controller to update color mappings
            if hasattr(self.robot_viewer, 'ai_controller'):
                self.robot_viewer.ai_controller.register_imported_objects()
                print("Re-registered objects with AI controller")
    
    def rename_model(self, item):
        """Rename a model in the tree and update the mesh data"""
        if not hasattr(self.robot_viewer, 'stl_handler'):
            return
        
        # Get the model index from the item data
        model_index = item.data(0, Qt.UserRole)
        
        # Get the current model name
        stl_handler = self.robot_viewer.stl_handler
        current_name = stl_handler.meshes[model_index].get('name', f"Model {model_index+1}")
        
        # Show an input dialog to get the new name
        new_name, ok = QInputDialog.getText(
            None, "Rename Model", "Enter new name for model:",
            text=current_name
        )
        
        # Update the name if the user provided one
        if ok and new_name:
            # Update the mesh data
            stl_handler.meshes[model_index]['name'] = new_name
            
            # Update the tree item
            item.setText(0, new_name)
            
            # Update message
            self.update_message(f"Renamed model to: {new_name}")
            
            # Refresh the tree
            self.update_stl_models_in_tree()
    
    def delete_model_from_tree(self, item):
        """Delete a model from the tree (UI only)"""
        try:
            # Remove from tree
            model_index = item.data(0, Qt.UserRole)
            
            if not model_index:
                self.update_message("Cannot delete: invalid model index")
                return False
                
            # Check if parent exists
            parent = item.parent()
            if not parent:
                self.update_message("Cannot delete: item has no parent")
                return False
                
            # Get the model name for logging
            model_name = item.text(0)
            
            # Remove the item from its parent (whether it's models_item or base item)
            parent.removeChild(item)
            
            # Update the UI message
            self.update_message(f"Deleted model: {model_name}")
            
            # If we've deleted the last model under base and it's empty now, we can keep it
            # Base node should always remain even if empty
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_message(f"Error deleting model from tree: {str(e)}")
            return False

    def import_stl_with_tree_update(self):
        """Import an STL file and ensure it appears in the tree"""
        print("*"*50)
        print("IMPORT_STL_WITH_TREE_UPDATE CALLED - STL HANDLER")
        print("*"*50)
        
        # First, we'll debug print the current state to help identify issues
        if hasattr(self, 'meshes'):
            print(f"Current meshes count before import: {len(self.meshes)}")
            for i, mesh in enumerate(self.meshes):
                print(f"  Existing mesh {i}: {mesh.get('name', 'unnamed')}")
        
        # Check if we have a program_tree reference
        if not hasattr(self, 'program_tree') or self.program_tree is None:
            # Try to get program_tree from parent
            if hasattr(self.robot_viewer, 'parent') and self.robot_viewer.parent():
                parent = self.robot_viewer.parent()
                if hasattr(parent, 'program_tree') and hasattr(parent, 'models_item'):
                    print("Getting program_tree reference from parent")
                    self.set_program_tree(parent.program_tree, parent.models_item)
        
        # Call the robot viewer's import_stl method
        if hasattr(self.robot_viewer, 'import_stl'):
            success = self.robot_viewer.import_stl()
            if success:
                print("STL import successful via STL Handler")
                
                # Check if the meshes are in robot_viewer.stl_handler
                if hasattr(self.robot_viewer, 'stl_handler') and hasattr(self.robot_viewer.stl_handler, 'meshes'):
                    rv_handler = self.robot_viewer.stl_handler
                    rv_mesh_count = len(rv_handler.meshes)
                    print(f"robot_viewer.stl_handler has {rv_mesh_count} meshes after import")
                    
                    # Always sync with the robot_viewer.stl_handler meshes
                    if rv_handler != self:
                        print(f"Syncing meshes from robot_viewer.stl_handler to self")
                        self.meshes = rv_handler.meshes.copy()
                        self.mesh_transforms = rv_handler.mesh_transforms.copy() if hasattr(rv_handler, 'mesh_transforms') else []
                        self.mesh_selection_colors = rv_handler.mesh_selection_colors.copy() if hasattr(rv_handler, 'mesh_selection_colors') else []
                        self.display_lists = self.robot_viewer.stl_handler.display_lists.copy() if hasattr(self.robot_viewer.stl_handler, 'display_lists') else []
                        self.mesh_file_paths = rv_handler.mesh_file_paths.copy() if hasattr(rv_handler, 'mesh_file_paths') else [] # <-- ADDED
                        print(f"After sync, self has {len(self.meshes)} meshes, {len(self.mesh_file_paths)} paths")
                    
                    # Debug print the updated meshes
                    for i, mesh in enumerate(self.meshes):
                        print(f"  Mesh after import {i}: {mesh.get('name', 'unnamed')}")
                
                # Ensure we have program_tree reference before updating tree
                if not hasattr(self, 'program_tree') or self.program_tree is None:
                    print("WARNING: No program_tree available for STL handler after import")
                    # Try to get program_tree from parent again
                    if hasattr(self.robot_viewer, 'parent') and self.robot_viewer.parent():
                        parent = self.robot_viewer.parent()
                        if hasattr(parent, 'program_tree') and hasattr(parent, 'models_item'):
                            print("Getting program_tree reference from parent after import")
                            self.set_program_tree(parent.program_tree, parent.models_item)
                
                # Ensure models are properly initialized
                if len(self.meshes) > 0:
                    # Select the newly imported model (last in the list)
                    self.selected_mesh_index = len(self.meshes) - 1
                    print(f"Selected newly imported mesh at index {self.selected_mesh_index}")
                
                    # Make gizmo visible
                    self.gizmo_state = 0  # 0 = visible (translate mode)
                    print(f"Set selected_mesh_index={self.selected_mesh_index}, gizmo_state={self.gizmo_state}")
                
                # Force tree update immediately if we have program_tree reference
                if hasattr(self, 'program_tree') and self.program_tree is not None:
                    print("Updating tree with newly imported model")
                    self.update_stl_models_in_tree()
                    
                    # Make sure models_item is expanded and visible
                    if hasattr(self, 'models_item') and self.models_item:
                        self.models_item.setExpanded(True)
                        self.program_tree.expandItem(self.models_item)
                        self.program_tree.scrollToItem(self.models_item)
                else:
                    print("WARNING: Cannot update tree - no program_tree reference")
                
                # Make sure the robot_viewer.stl_handler is synced with this handler
                if self.robot_viewer and hasattr(self.robot_viewer, 'stl_handler'):
                    if self.robot_viewer.stl_handler != self:
                        rv_handler = self.robot_viewer.stl_handler
                        # Sync from us to robot_viewer.stl_handler if needed
                        if len(self.meshes) > len(rv_handler.meshes):
                            print("Syncing our meshes to robot_viewer.stl_handler")
                            rv_handler.meshes = self.meshes.copy()
                            rv_handler.mesh_transforms = self.mesh_transforms.copy() if hasattr(self, 'mesh_transforms') else []
                            rv_handler.mesh_selection_colors = self.mesh_selection_colors.copy() if hasattr(self, 'mesh_selection_colors') else []
                            rv_handler.display_lists = self.display_lists.copy() if hasattr(self, 'display_lists') else []
                            rv_handler.mesh_file_paths = self.mesh_file_paths.copy() if hasattr(self, 'mesh_file_paths') else [] # <-- ADDED
                        
                        # Sync the selected_mesh_index
                        rv_handler.selected_mesh_index = self.selected_mesh_index
                        # Make gizmo visible
                        rv_handler.gizmo_state = 0  # 0 = visible (translate mode)
                        print(f"Synced robot_viewer.stl_handler: selected_mesh_index={rv_handler.selected_mesh_index}, gizmo_state={rv_handler.gizmo_state}")
                
                # Force an immediate redraw to show the gizmo
                if self.robot_viewer:
                    self.robot_viewer.update()
                
                # Then force another update after a short delay to ensure the model is processed
                if hasattr(self, 'program_tree') and self.program_tree is not None:
                    QTimer.singleShot(200, self.update_stl_models_in_tree)
                
                return True
            else:
                print("STL import failed or was cancelled")
                return False
        else:
            print("Cannot import STL: robot_viewer not found or has no import_stl method")
            return False
    
    def toggle_stl_gizmo(self):
        """Toggle the visibility of the gizmo for the selected STL model"""
        if not hasattr(self.robot_viewer, 'stl_handler'):
            self.update_message("STL handler not available")
            return
            
        stl_handler = self.robot_viewer.stl_handler
        
        # Check if a model is selected
        if stl_handler.selected_mesh_index < 0:
            self.update_message("No STL model selected. Please select a model first.")
            return
            
        # Toggle the gizmo state
        current_state = stl_handler.gizmo_state  # 0: translate, 1: hidden
        new_state = 1 - current_state  # Toggle between 0 and 1
        stl_handler.gizmo_state = new_state
        
        # Update message
        if new_state == 0:
            self.update_message("STL gizmo activated")
        else:
            self.update_message("STL gizmo hidden")
            
        # Force a redraw
        self.robot_viewer.update()
    
    def reset_stl_positions(self):
        """Reset all STL models to their original positions, orientations, scales, and colors"""
        if not hasattr(self.robot_viewer, 'stl_handler'):
            self.update_message("STL handler not available")
            return
            
        stl_handler = self.robot_viewer.stl_handler
        
        # Check if we have any meshes
        if not stl_handler.meshes:
            self.update_message("No STL models to reset")
            return
            
        try:
            reset_count = 0
            for i in range(len(stl_handler.mesh_transforms)):
                # Reset position to original if stored, otherwise use default
                if hasattr(stl_handler, 'original_positions') and i in stl_handler.original_positions:
                    stl_handler.mesh_transforms[i]['position'] = stl_handler.original_positions[i].copy()
                else:
                    # Default position for new models
                    stl_handler.mesh_transforms[i]['position'] = np.array([0.0, -0.3, 0.1], dtype=np.float32)
                
                # Reset orientation to original if stored, otherwise use identity quaternion
                if hasattr(stl_handler, 'original_orientations') and i in stl_handler.original_orientations:
                    stl_handler.mesh_transforms[i]['orientation'] = stl_handler.original_orientations[i].copy()
                else:
                    stl_handler.mesh_transforms[i]['orientation'] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                
                # Reset scale to original if stored, otherwise use unit scale
                if hasattr(stl_handler, 'original_scales') and i in stl_handler.original_scales:
                    stl_handler.mesh_transforms[i]['scale'] = stl_handler.original_scales[i].copy()
                else:
                    stl_handler.mesh_transforms[i]['scale'] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                
                # Reset color to original if stored
                if hasattr(stl_handler, 'original_colors') and i in stl_handler.original_colors:
                    if i < len(stl_handler.meshes):
                        stl_handler.meshes[i]['color'] = stl_handler.original_colors[i].copy()
                
                reset_count += 1
                
            self.update_message(f"Reset {reset_count} STL model(s) to original state")
            
            # Force a redraw
            self.robot_viewer.update()
            
        except Exception as e:
            self.update_message(f"Error resetting STL positions: {str(e)}")
            import traceback
            traceback.print_exc() 

    def pick_object(self, robot_programmer=None):
        """Pick up the selected STL object with the robot"""
        # Find the currently selected mesh
        if not hasattr(self.robot_viewer, 'stl_handler'):
            self.update_message("STL handler not available")
            return
            
        stl_handler = self.robot_viewer.stl_handler
        if stl_handler.selected_mesh_index < 0:
            self.update_message("No STL model selected. Please select a model first.")
            return
            
        # Store the original position before modifying
        object_index = stl_handler.selected_mesh_index
        try:
            # Get current object transform
            transform = stl_handler.mesh_transforms[object_index]
            current_object_pos = transform['position'].copy() # Use current position
            current_object_orn = transform['orientation'].copy() # FIX ISSUE #1: Store current orientation

            # Get the robot's current end effector position
            if hasattr(self.robot_viewer, 'robot_chain') and self.robot_viewer.robot_chain:
                # Get transforms for all joints
                transforms = self.robot_viewer.robot_chain.forward_kinematics(
                    self.robot_viewer.joint_values, full_kinematics=True)

                # Get the selected robot index
                robot_index = getattr(self.robot_viewer, 'selected_robot_index', 0)
                gripper_index = 7 # Assuming visual gripper index is 7
                gripper_transform_world = transforms[-1] @ self.robot_viewer.link_info[gripper_index]['visual_transform']

                # --- NEW: If duplicate robot, transform object position into robot's base frame ---
                object_pos_world = np.array(current_object_pos)
                object_pos_world_homogeneous = np.append(object_pos_world, 1)
                if robot_index > 0 and hasattr(self.robot_viewer, 'duplicate_robot_bases'):
                    dup = next((d for d in self.robot_viewer.duplicate_robot_bases if d['index'] == robot_index), None)
                    if dup is not None:
                        angle_rad = np.radians(dup.get('rotation', 0.0))
                        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                        base_transform = np.eye(4)
                        base_transform[:3, :3] = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
                        base_transform[:3, 3] = [dup['offset'][0], dup['offset'][1], 0.0]
                        base_transform_inv = np.linalg.inv(base_transform)
                        object_pos_in_robot_base = base_transform_inv @ object_pos_world_homogeneous
                    else:
                        object_pos_in_robot_base = object_pos_world_homogeneous
                else:
                    object_pos_in_robot_base = object_pos_world_homogeneous

                # --- Calculate offset in Gripper's Local Frame (in robot base frame) ---
                try:
                    gripper_transform_inv = np.linalg.inv(gripper_transform_world)
                except np.linalg.LinAlgError:
                    print("Error: Gripper transform is not invertible.")
                    return
                offset_local_homogeneous = gripper_transform_inv @ object_pos_in_robot_base
                offset_local = offset_local_homogeneous[:3] # Extract x, y, z
                
                # FIX ISSUE #1: Calculate orientation offset to preserve object rotation
                from scipy.spatial.transform import Rotation as R
                gripper_rot_matrix = gripper_transform_world[:3, :3]
                object_rot_matrix = R.from_quat(current_object_orn).as_matrix()
                # Calculate relative rotation: offset_rot = gripper_inv * object
                offset_rot_matrix = gripper_rot_matrix.T @ object_rot_matrix
                offset_orn_quat = R.from_matrix(offset_rot_matrix).as_quat()

                print(f"[Pick Object] Current Object Pos (World): {object_pos_world}")
                print(f"[Pick Object] Current Object Orn (Quat): {current_object_orn}")
                print(f"[Pick Object] Gripper Transform (World): \\n{gripper_transform_world}")
                print(f"[Pick Object] Calculated Offset (Local): {offset_local}")
                print(f"[Pick Object] Calculated Orientation Offset (Quat): {offset_orn_quat}")
                # --- END FIX ---

                # Set the robot viewer's attached object
                self.robot_viewer.attached_object_index = object_index
                self.robot_viewer.attached_object_offset = {
                    'position': offset_local.tolist(), # Store the calculated LOCAL offset
                    'orientation': offset_orn_quat.tolist(), # FIX ISSUE #1: Store orientation offset
                    'gripper_index': gripper_index,    # Store the index of the gripper frame used
                    'robot_index': robot_index # Store which robot picked up the object
                }

                self.update_message(f"Picked up object {object_index+1}")

                # Update the visualization AFTER setting attachment and position
                self.robot_viewer.update()

                # Add the pick operation to the program if programmer exists
                if robot_programmer:
                    # Pass the calculated local offset to the programmer
                    robot_programmer.record_pick_operation(object_index, offset_local.tolist())

            else:
                self.update_message("Error: Robot chain not found.")
                
        except Exception as e:
            self.update_message(f"Error picking up object: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def place_object(self, robot_programmer=None):
        """Place the currently held STL object"""
        # Check if we have an attached object
        if not hasattr(self.robot_viewer, 'attached_object_index') or self.robot_viewer.attached_object_index < 0:
            self.update_message("No object currently picked up")
            return
            
        try:
            # Get the index of the attached object
            object_index = self.robot_viewer.attached_object_index
            
            # The object's transform should already be correctly updated by the viewer's loop
            # We just need to finalize its state before detaching.
            transform = self.mesh_transforms[object_index]
            # The current transform['position'] is the final desired position.
            # No need to recalculate using ee_position + offset here.
            
            # Add the place operation to the program if programmer exists
            if robot_programmer:
                # Pass the final position explicitly to the programmer if needed
                # Assuming record_place_operation doesn't need the position explicitly, 
                # or can get it from the handler/viewer if required.
                robot_programmer.record_place_operation(object_index)
            
            # Release the object (set to -1)
            self.robot_viewer.attached_object_index = -1
            self.robot_viewer.attached_object_offset = None
            
            self.update_message(f"Placed object {object_index+1} at current position")
            
            # Update the visualization
            self.robot_viewer.update()
            
        except Exception as e:
            self.update_message(f"Error placing object: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_scale_popup(self):
        """Create the popup dialog for scaling STL models"""
        print("Creating scale popup dialog")
        
        # Create a dialog for scaling
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QCheckBox
        from PyQt5.QtCore import Qt
        
        self.scale_popup = QWidget()
        self.scale_popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.scale_popup.setFixedWidth(200)  # Make it more compact
        self.scale_popup.setAttribute(Qt.WA_StyledBackground, True)
        self.scale_popup.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #cccccc;
            }
            QLabel {
                color: black;
                padding: 0px;
                margin: 0px;
            }
            QDoubleSpinBox {
                background-color: white;
                border: 1px solid #cccccc;
                padding: 2px;
                min-width: 50px;
            }
            QCheckBox {
                color: black;
                padding: 2px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #cccccc;
                background: white;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #cccccc;
                background: #4CAF50;
            }
        """)
        
        # Create main layout
        main_layout = QVBoxLayout(self.scale_popup)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Add "Scale" label at top
        scale_label = QLabel("Scale")
        scale_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(scale_label)
        
        # Create grid layout for inputs
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(4)
        
        # Create inputs with labels
        labels = ['X', 'Y', 'Z']
        self.scale_inputs = {}
        
        for label in labels:
            # Create vertical layout for each input group
            input_layout = QVBoxLayout()
            input_layout.setSpacing(2)
            
            # Add label
            label_widget = QLabel(label)
            label_widget.setAlignment(Qt.AlignCenter)
            input_layout.addWidget(label_widget)
            
            # Create input field
            input_field = QDoubleSpinBox()
            input_field.setRange(1, 1000)  # 1% to 1000%
            input_field.setValue(100)
            input_field.setSingleStep(1)
            input_field.setAlignment(Qt.AlignRight)
            input_field.setFixedWidth(50)
            input_field.valueChanged.connect(lambda v, axis=label.lower(): self.apply_scale(axis))
            self.scale_inputs[f'{label.lower()}_input'] = input_field
            
            # Create horizontal layout for input and unit
            field_layout = QHBoxLayout()
            field_layout.setSpacing(2)
            field_layout.addWidget(input_field)
            field_layout.addWidget(QLabel("%"))
            
            input_layout.addLayout(field_layout)
            grid_layout.addLayout(input_layout)
        
        main_layout.addLayout(grid_layout)
        
        # Add checkboxes
        uniform_check = QCheckBox("Uniform Scaling")
        uniform_check.setChecked(True)
        uniform_check.stateChanged.connect(self.handle_uniform_scale)
        main_layout.addWidget(uniform_check)
        
        show_controls = QCheckBox("Show controls")
        show_controls.stateChanged.connect(self.toggle_gizmo_controls)
        main_layout.addWidget(show_controls)
        
        # Store input references
        self.scale_inputs.update({
            'uniform_check': uniform_check,
            'show_controls': show_controls
        })

    def add_mesh(self, mesh, name="Generated Mesh"):
        """Adds a mesh object (like one from trimesh) to the scene.

        Args:
            mesh (trimesh.Trimesh): The mesh object to add.
            name (str): The name for the mesh in the scene tree.

        Returns:
            bool: True if the mesh was added successfully, False otherwise.
        """
        mesh_id = -1 # Initialize mesh_id to indicate failure initially
        try:
            if not mesh or not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
                print("Error: Invalid mesh object provided to add_mesh")
                return False # Moved return inside the if

            # --- Start of code that should be inside the try block ---
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)

            # Use name from mesh metadata if available
            if hasattr(mesh, 'metadata') and 'name' in mesh.metadata:
                model_name = mesh.metadata['name']
            else:
                model_name = name

            # Generate a unique name based on existing names
            base_name = model_name
            count = 1
            existing_names = {m['name'] for m in self.meshes}
            while model_name in existing_names:
                model_name = f"{base_name} {count}"
                count += 1

            # Center the mesh (optional, consider if placement should be elsewhere)
            # Assuming primitives are already centered, this might not be needed,
            # but it's consistent with import_stl.
            # Note: If primitives are NOT centered, this might shift them unexpectedly.
            # Let's comment this out for primitives for now, assuming they are centered.
            # center = np.mean(vertices, axis=0)
            # vertices = vertices - center

            # Apply global scale if it wasn't applied before (e.g., for primitives)
            # Check metadata hint from drawing.py (if it exists)
            # Primitives from drawing.py are likely in meters, so they might need scaling *up* if global_scale is 0.001
            # Or maybe they don't need scaling at all if global_scale is 1.
            # Let's assume drawing.py primitives are already in the correct scale (meters) for now.
            # if 'already_scaled' not in mesh.metadata or not mesh.metadata['already_scaled']:
            #      print(f"Applying global scale ({self.global_scale}) to '{model_name}' in add_mesh.")
            #      vertices = vertices * self.global_scale
            # else:
            #      print(f"Skipping global scale for '{model_name}' in add_mesh (likely already scaled).")
            # Reverted: Let's keep the original scaling logic for now
            # Apply global scale if it wasn't applied before
            # Check if metadata indicates scaling is needed (e.g., from primitives)
            if 'already_scaled' not in mesh.metadata or not mesh.metadata['already_scaled']:
                 print(f"Applying global scale ({self.global_scale}) to '{model_name}' in add_mesh.")
                 vertices = vertices * self.global_scale
            else:
                 print(f"Skipping global scale for '{model_name}' in add_mesh (already scaled).")


            # --- Edges ---
            # Use trimesh's edge calculation on the potentially scaled/centered vertices
            temp_mesh_for_edges = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            edges_unique_indices = temp_mesh_for_edges.edges_unique
            edge_vertices = vertices[edges_unique_indices] # Index the final vertex array

            # --- Mesh Data Dict ---
            # IMPORTANT: Copy relevant metadata HERE
            mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'edges': edge_vertices, # Store edge vertex coordinates
                'edge_indices': edges_unique_indices, # Store edge vertex indices
                'name': model_name, # Use the unique name
                'color': mesh.metadata.get('color', self.default_color), # Get color or use default
                # --- ADDED: Copy bounds from metadata if they exist ---
                # Use metadata bounds if available, otherwise calculate from final vertices
                'local_min_bounds': mesh.metadata.get('local_min_bounds', np.min(vertices, axis=0).tolist() if len(vertices)>0 else [0,0,0]),
                'local_max_bounds': mesh.metadata.get('local_max_bounds', np.max(vertices, axis=0).tolist() if len(vertices)>0 else [0,0,0]),
                # --- END ADDED ---
            }
            # --- ADDED: Print statement to confirm bounds ---
            print(f"  Added mesh '{mesh_data['name']}' with bounds: MIN={mesh_data['local_min_bounds']}, MAX={mesh_data['local_max_bounds']}")
            # --- END ADDED ---

            # Add to handler lists
            self.meshes.append(mesh_data)
            self.mesh_file_paths.append(None) # Add placeholder for file path

            # Create OpenGL display list
            display_list = self.create_mesh_display_list(mesh_data)
            if display_list is None:
                print(f"Error: Failed to create display list for {model_name}")
                self.meshes.pop()  # Remove mesh data if display list creation failed
                self.mesh_file_paths.pop() # Remove placeholder path
                return False
            self.display_lists.append(display_list) # This line needed dedenting too

            # Add default transform (position at origin, no rotation, unit scale)
            transform = {
                'position': np.array([0.0, -0.3, 0.1], dtype=np.float32), # Start at Y=-0.3 to avoid robot collision
                'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), # Identity quaternion
                'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
            }
            self.mesh_transforms.append(transform)

            # Store original transform for resetting
            # Make sure original_positions and original_orientations are initialized as dicts
            if not hasattr(self, 'original_positions'): self.original_positions = {}
            if not hasattr(self, 'original_orientations'): self.original_orientations = {}
            if not hasattr(self, 'original_scales'): self.original_scales = {}
            if not hasattr(self, 'original_colors'): self.original_colors = {}
            current_index = len(self.meshes) - 1
            self.original_positions[current_index] = transform['position'].copy()
            self.original_orientations[current_index] = transform['orientation'].copy()
            self.original_scales[current_index] = transform['scale'].copy()
            # Store original color from mesh data
            if current_index < len(self.meshes) and 'color' in self.meshes[current_index]:
                self.original_colors[current_index] = list(self.meshes[current_index]['color'])

            # Assign a unique selection color
            mesh_id = current_index # Assign mesh_id here, after successful appends
            r = 0.1 + (mesh_id % 5) * 0.02 # More distinct colors
            g = 0.1 + ((mesh_id // 5) % 5) * 0.02
            b = 0.1 + ((mesh_id // 25) % 5) * 0.02
            self.mesh_selection_colors.append((r, g, b))

            # Update the models tree
            self.update_stl_models_in_tree()

            print(f"Successfully added mesh: {model_name}")
            print(f"Total models: {len(self.meshes)}")

            # Select the newly added mesh
            self.select_mesh(mesh_id)
            self.ensure_edges_for_all_meshes()
            return True

        except Exception as e:
            print(f"Error adding mesh: {str(e)}")
            import traceback
            traceback.print_exc()
            # Clean up potentially added data on error
            # Check if mesh_id was assigned before trying to use it for cleanup
            if mesh_id != -1: # Only clean up if mesh_id got assigned
                if len(self.meshes) > mesh_id: self.meshes.pop(mesh_id)
                if len(self.display_lists) > mesh_id: self.display_lists.pop(mesh_id)
                if len(self.mesh_transforms) > mesh_id: self.mesh_transforms.pop(mesh_id)
                if len(self.mesh_file_paths) > mesh_id: self.mesh_file_paths.pop(mesh_id)
                if len(self.mesh_selection_colors) > mesh_id: self.mesh_selection_colors.pop(mesh_id)
                if hasattr(self, 'original_positions') and mesh_id in self.original_positions:
                     del self.original_positions[mesh_id]
                if hasattr(self, 'original_orientations') and mesh_id in self.original_orientations:
                     del self.original_orientations[mesh_id]
                if hasattr(self, 'original_scales') and mesh_id in self.original_scales:
                     del self.original_scales[mesh_id]
                if hasattr(self, 'original_colors') and mesh_id in self.original_colors:
                     del self.original_colors[mesh_id]
            # Even if mesh_id wasn't assigned, check lengths relative to each other
            elif len(self.meshes) > len(self.display_lists): self.meshes.pop()
            # ... (Add similar checks for other lists if needed)

            return False

    def clear_models(self):
        """Remove all STL models from the handler and viewer."""
        print("[STLHandler] Clearing all models...")
        try:
            # Delete OpenGL display lists
            self.cleanup() # cleanup already deletes display lists

            # Clear internal lists
            self.meshes = []
            self.mesh_transforms = []
            self.mesh_selection_colors = []
            self.mesh_file_paths = []
            self.display_lists = [] # cleanup should have handled this, but clear anyway
            self.original_positions = {}
            self.original_orientations = {}
            self.original_scales = {}
            self.original_colors = {}

            # Reset selection
            self.selected_mesh_index = -1

            # Update the tree (should show empty Models section)
            self.update_stl_models_in_tree()

            # Force viewer redraw
            if self.robot_viewer:
                self.robot_viewer.update()

            self.update_message("Cleared all STL models.")
            print("[STLHandler] Models cleared successfully.")
            return True
        except Exception as e:
            print(f"[STLHandler] Error clearing models: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_message(f"Error clearing models: {str(e)}")
            return False

    def load_models_from_data(self, models_data):
        """Load STL models from a list of dictionaries (from a scene file), including primitives/in-memory objects."""
        print(f"[STLHandler] Loading {len(models_data)} models from data...")
        loaded_count = 0
        failed_count = 0
        for model_data in models_data:
            file_path = model_data.get("file_path")
            name = model_data.get("name", "Unnamed Model")
            transform_data = model_data.get("transform", {})
            mesh_data = model_data.get("mesh_data")

            mesh_dict = None
            centering_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            if file_path and os.path.exists(file_path):
                try:
                    # Load the mesh using trimesh
                    mesh = trimesh.load(file_path)
                    vertices = np.array(mesh.vertices, dtype=np.float32)
                    faces = np.array(mesh.faces, dtype=np.int32)
                    
                    # CRITICAL FIX: We MUST center the mesh just like during import
                    # to keep the gizmo at the center. Center all three axes.
                    
                    # Calculate centering offset BEFORE scaling
                    center = np.mean(vertices, axis=0)
                    
                    # Apply centering (same as import_stl)
                    vertices[:, 0] -= center[0]  # Center X
                    vertices[:, 1] -= center[1]  # Center Y
                    vertices[:, 2] -= center[2]  # Center Z
                    
                    # Track the centering offset (in original units, before scaling)
                    centering_offset = center.copy()
                    
                    # Apply global scale (MM to Meters conversion)
                    vertices *= self.global_scale
                    centering_offset *= self.global_scale
                    
                    print(f"[LOAD DEBUG] Loaded '{name}' from file:")
                    print(f"  - Centering offset (M): {centering_offset}")
                    print(f"  - Vertices after centering and scaling: min={vertices.min():.6f}, max={vertices.max():.6f}")
                    
                    edges_unique_indices = mesh.edges_unique
                    edge_vertices = vertices[edges_unique_indices]
                    color = model_data.get('color', self.default_color)
                    color_name = model_data.get('color_name', 'gray')
                    local_min_bounds = np.min(vertices, axis=0)
                    local_max_bounds = np.max(vertices, axis=0)
                    
                    # Verify against saved bounds if available
                    if 'local_min_bounds' in model_data and 'local_max_bounds' in model_data:
                        saved_min = np.array(model_data['local_min_bounds'], dtype=np.float32) / 1000.0
                        saved_max = np.array(model_data['local_max_bounds'], dtype=np.float32) / 1000.0
                        print(f"  - Saved bounds (M): min={saved_min}, max={saved_max}")
                        print(f"  - Calculated bounds (M): min={local_min_bounds}, max={local_max_bounds}")
                        
                        # Check if bounds match (within tolerance)
                        if not (np.allclose(local_min_bounds, saved_min, atol=0.001) and 
                                np.allclose(local_max_bounds, saved_max, atol=0.001)):
                            print(f"  WARNING: Bounds mismatch! File may have changed since save.")
                    
                    mesh_dict = {
                        'vertices': vertices,
                        'faces': faces,
                        'edges': edge_vertices,
                        'name': name,
                        'color': color,
                        'color_name': color_name,
                        'file_path': file_path,
                        'local_min_bounds': local_min_bounds,
                        'local_max_bounds': local_max_bounds
                    }
                except Exception as e:
                    print(f"Error loading model '{name}' from '{file_path}': {str(e)}")
                    import traceback
                    traceback.print_exc()
                    failed_count += 1
                    continue
            elif mesh_data:
                try:
                    # CRITICAL FIX: For in-memory/primitives, vertices are saved in MM
                    # We need to convert them back to meters for internal use
                    vertices_mm = np.array(mesh_data["vertices"], dtype=np.float32)
                    faces = np.array(mesh_data["faces"], dtype=np.int32)
                    edges_mm = np.array(mesh_data.get("edges", []), dtype=np.float32) if mesh_data.get("edges") else []
                    
                    print(f"[LOAD DEBUG] Loading in-memory object '{name}':")
                    print(f"  - Vertices from file (MM): min={vertices_mm.min():.3f}, max={vertices_mm.max():.3f}")
                    if len(vertices_mm) > 0:
                        print(f"  - First vertex (MM): {vertices_mm[0]}")
                    
                    # Convert from MM to meters (divide by 1000)
                    vertices = vertices_mm / 1000.0
                    edges = edges_mm / 1000.0 if len(edges_mm) > 0 else edges_mm
                    
                    print(f"  - Vertices after conversion (M): min={vertices.min():.6f}, max={vertices.max():.6f}")
                    if len(vertices) > 0:
                        print(f"  - First vertex (M): {vertices[0]}")
                    
                    color = model_data.get('color', self.default_color)
                    color_name = model_data.get('color_name', 'gray')
                    
                    # Local bounds are also in MM, convert to meters
                    local_min_bounds_mm = np.array(mesh_data.get('local_min_bounds', [0,0,0]), dtype=np.float32)
                    local_max_bounds_mm = np.array(mesh_data.get('local_max_bounds', [0,0,0]), dtype=np.float32)
                    local_min_bounds = local_min_bounds_mm / 1000.0
                    local_max_bounds = local_max_bounds_mm / 1000.0
                    
                    print(f"  - Local bounds (M): min={local_min_bounds}, max={local_max_bounds}")
                    
                    mesh_dict = {
                        'vertices': vertices,
                        'faces': faces,
                        'edges': edges,
                        'name': name,
                        'color': color,
                        'color_name': color_name,
                        'file_path': None,
                        'local_min_bounds': local_min_bounds,
                        'local_max_bounds': local_max_bounds
                    }
                except Exception as e:
                    print(f"Error reconstructing mesh for '{name}': {str(e)}")
                    import traceback
                    traceback.print_exc()
                    failed_count += 1
                    continue
            else:
                print(f"Warning: Skipping model '{name}'. No file_path or mesh_data present.")
                failed_count += 1
                continue

            # Create transform from loaded data
            # CRITICAL FIX: Positions are saved in MM, need to convert back to meters
            position_mm = np.array(transform_data.get('position', [0, 0, 0]), dtype=np.float32)
            # Convert from MM to meters (divide by 1000)
            position = position_mm / 1000.0
            
            orientation = np.array(transform_data.get('orientation', [0, 0, 0, 1]), dtype=np.float32)
            scale = np.array(transform_data.get('scale', [1, 1, 1]), dtype=np.float32)
            
            print(f"  - Transform loaded (MM): position={position_mm}")
            print(f"  - Transform converted (M): position={position}, orientation={orientation}, scale={scale}")
            
            transform = {
                'position': position,
                'orientation': orientation,
                'scale': scale
            }

            # Create display list
            display_list = self.create_mesh_display_list(mesh_dict)
            if display_list is None:
                print(f"Warning: Failed to create display list for {name}. Skipping.")
                failed_count += 1
                continue

            # Append everything
            self.meshes.append(mesh_dict)
            self.mesh_transforms.append(transform)
            self.display_lists.append(display_list)
            self.mesh_file_paths.append(file_path if file_path and os.path.exists(file_path) else None)

            # Assign selection color (based on current count)
            mesh_id = len(self.meshes) - 1
            r = 0.3 + (mesh_id % 3) * 0.02
            g = 0.3 + ((mesh_id // 3) % 3) * 0.02
            b = 0.3 + ((mesh_id // 9) % 3) * 0.02
            self.mesh_selection_colors.append((r, g, b))

            # Store original transform (important for reset functionality)
            self.original_positions[mesh_id] = position.copy()
            self.original_orientations[mesh_id] = orientation.copy()
            self.original_scales[mesh_id] = scale.copy()
            # Store original color from mesh data
            if 'color' in mesh_dict:
                self.original_colors[mesh_id] = list(mesh_dict['color'])

            loaded_count += 1
            print(f"Successfully loaded model: {name} (file: {file_path if file_path else 'in-memory'})")

        self.update_message(f"Loaded {loaded_count} STL models ({failed_count} failed).")

        # Select the first loaded model if any
        if loaded_count > 0:
            self.selected_mesh_index = 0
        else:
            self.selected_mesh_index = -1

        # Update the tree and viewer
        self.update_stl_models_in_tree()
        if self.robot_viewer:
            self.robot_viewer.update()

        return loaded_count > 0

    def get_models_data(self):
        """Returns a list of dictionaries representing the current STL models for saving, including primitives/in-memory objects."""
        models_to_save = []
        for i, mesh in enumerate(self.meshes):
            # Check transform existence
            if i < len(self.mesh_transforms):
                transform = self.mesh_transforms[i]
                
                # Robustly get file path
                file_path = self.mesh_file_paths[i] if i < len(self.mesh_file_paths) else None

                # Calculate world bounds using existing method (returns mm)
                world_bounds_data = self.get_model_bounds_by_index_or_name(i)
                world_bounds = None
                if world_bounds_data:
                    world_bounds = {
                        "min": world_bounds_data['min_point'],
                        "max": world_bounds_data['max_point']
                    }

                model_data = {
                    "name": mesh.get("name", "Unnamed Model"),
                    "color": mesh.get("color", self.default_color),
                    "color_name": self._rgb_to_color_name(mesh.get("color", self.default_color)),
                    "transform": {
                        # Convert position to mm using heuristic: values < 50 are multiplied by 1000
                        "position": self._position_to_mm(transform['position']),
                        "orientation": transform['orientation'].tolist() if isinstance(transform.get('orientation'), np.ndarray) else transform.get('orientation', [0,0,0,1]),
                        "scale": transform['scale'].tolist() if isinstance(transform.get('scale'), np.ndarray) else transform.get('scale', [1,1,1])
                    },
                    "world_bounds": world_bounds
                }

                if file_path and os.path.exists(file_path):
                    model_data["file_path"] = file_path
                    # CRITICAL: Also save local bounds for file-based models
                    # This allows us to verify centering is correct when reloading
                    l_min = np.array(mesh.get("local_min_bounds", [0,0,0]))
                    l_max = np.array(mesh.get("local_max_bounds", [0,0,0]))
                    model_data["local_min_bounds"] = [round(float(v) * 1000, 3) for v in l_min]
                    model_data["local_max_bounds"] = [round(float(v) * 1000, 3) for v in l_max]
                else:
                    # Save FULL mesh data for primitives/in-memory objects
                    # We need vertices and faces to reconstruct the mesh
                    vertices = mesh.get("vertices", np.array([]))
                    faces = mesh.get("faces", np.array([]))
                    edges = mesh.get("edges", np.array([]))
                    
                    print(f"[SAVE DEBUG] Saving in-memory object '{mesh.get('name', 'unnamed')}':")
                    if isinstance(vertices, np.ndarray) and len(vertices) > 0:
                        print(f"  - Vertices before conversion (M): min={vertices.min():.6f}, max={vertices.max():.6f}")
                        print(f"  - First vertex (M): {vertices[0]}")
                    
                    # CRITICAL FIX: Convert vertices to MM for consistency
                    # Vertices are stored in meters internally, but we save in mm
                    vertices_mm = vertices * 1000 if isinstance(vertices, np.ndarray) and len(vertices) > 0 else vertices
                    edges_mm = edges * 1000 if isinstance(edges, np.ndarray) and len(edges) > 0 else edges
                    
                    if isinstance(vertices_mm, np.ndarray) and len(vertices_mm) > 0:
                        print(f"  - Vertices after conversion (MM): min={vertices_mm.min():.3f}, max={vertices_mm.max():.3f}")
                        print(f"  - First vertex (MM): {vertices_mm[0]}")
                    
                    # Convert to lists for JSON serialization
                    vertices_list = vertices_mm.tolist() if isinstance(vertices_mm, np.ndarray) else []
                    faces_list = faces.tolist() if isinstance(faces, np.ndarray) else []
                    edges_list = edges_mm.tolist() if isinstance(edges_mm, np.ndarray) else []
                    
                    # Local bounds are converted to MM here (original is in Meters)
                    l_min = np.array(mesh.get("local_min_bounds", [0,0,0]))
                    l_max = np.array(mesh.get("local_max_bounds", [0,0,0]))
                    
                    print(f"  - Local bounds before conversion (M): min={l_min}, max={l_max}")
                    print(f"  - Local bounds after conversion (MM): min={l_min * 1000}, max={l_max * 1000}")
                    
                    model_data["mesh_data"] = {
                        "vertices": vertices_list,
                        "faces": faces_list,
                        "edges": edges_list,
                        "local_min_bounds": [round(float(v) * 1000, 3) for v in l_min],
                        "local_max_bounds": [round(float(v) * 1000, 3) for v in l_max],
                    }
                models_to_save.append(model_data)
            else:
                print(f"Warning: Skipping model at index {i} during save. Data mismatch (transform or file path missing).")

        print(f"[STLHandler] Gathered data for {len(models_to_save)} STL models to save.")
        return models_to_save

    # --- Methods for AI Control ---

    def translate_model_directly(self, index, vector):
        """Translates the model at the specified index by the given vector.

        Args:
            index (int): The index of the model to translate.
            vector (list or np.ndarray): The translation vector [x, y, z].

        Returns:
            bool: True if translation was successful, False otherwise.
        """
        if not (0 <= index < len(self.mesh_transforms)):
            print(f"Error: Invalid index {index} for translate_model_directly.")
            self.update_message(f"Error: Cannot translate model at invalid index {index}.")
            return False

        try:
            translation_vector = np.array(vector, dtype=np.float32)
            if translation_vector.shape != (3,):
                raise ValueError("Translation vector must have 3 elements.")

            # Apply global scale to the translation vector (MM -> Meters)
            scaled_vector = translation_vector * self.global_scale

            # Add the vector to the current position
            current_position = self.mesh_transforms[index]['position']
            self.mesh_transforms[index]['position'] = current_position + scaled_vector

            print(f"Translated model {index} by {vector}. New position: {self.mesh_transforms[index]['position']}")
            self.update_message(f"Translated model {index} by {vector}.")

            # Force viewer redraw
            if self.robot_viewer:
                self.robot_viewer.update()

            return True

        except Exception as e:
            print(f"Error during translate_model_directly for index {index}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_message(f"Error translating model {index}: {str(e)}")
            return False

    def rotate_model_directly(self, index, euler_angles):
        """Sets the rotation of the model at the specified index using Euler angles.

        Args:
            index (int): The index of the model to rotate.
            euler_angles (list or np.ndarray): Rotation angles [rx, ry, rz] in degrees.

        Returns:
            bool: True if rotation was successful, False otherwise.
        """
        if not (0 <= index < len(self.mesh_transforms)):
            print(f"Error: Invalid index {index} for rotate_model_directly.")
            self.update_message(f"Error: Cannot rotate model at invalid index {index}.")
            return False

        try:
            if len(euler_angles) != 3:
                raise ValueError("Euler angles must have 3 elements [rx, ry, rz].")

            x_rot, y_rot, z_rot = euler_angles

            # Convert Euler angles (degrees) to quaternion
            # (Adapted from set_mesh_rotation logic)
            x_rad = math.radians(x_rot)
            y_rad = math.radians(y_rot)
            z_rad = math.radians(z_rot)

            cx = math.cos(x_rad / 2)
            sx = math.sin(x_rad / 2)
            cy = math.cos(y_rad / 2)
            sy = math.sin(y_rad / 2)
            cz = math.cos(z_rad / 2)
            sz = math.sin(z_rad / 2)

            qw = cx * cy * cz + sx * sy * sz
            qx = sx * cy * cz - cx * sy * sz
            qy = cx * sy * cz + sx * cy * sz
            qz = cx * cy * sz - sx * sy * cz

            # Normalize quaternion
            magnitude = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if magnitude < 1e-9:
                 # Avoid division by zero if angles are all zero
                 qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            else:
                qx /= magnitude
                qy /= magnitude
                qz /= magnitude
                qw /= magnitude

            # Update the mesh's transform orientation
            self.mesh_transforms[index]['orientation'] = np.array([qx, qy, qz, qw], dtype=np.float32)

            print(f"Rotated model {index} to Euler(deg): {euler_angles}. New quaternion: {self.mesh_transforms[index]['orientation']}")
            self.update_message(f"Rotated model {index} to {euler_angles} degrees.")

            # Force viewer redraw
            if self.robot_viewer:
                self.robot_viewer.update()

            return True

        except Exception as e:
            print(f"Error during rotate_model_directly for index {index}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_message(f"Error rotating model {index}: {str(e)}")
            return False    # --- End Methods for AI Control ---

    def import_stl_from_path(self, file_path):
        print(f"[STLHandler] import_stl_from_path called with path: {file_path}")
        if not file_path or not os.path.exists(file_path):
            self.update_message(f"Error: STL file not found at {file_path}")
            print(f"Error: STL file not found at {file_path}")
            return False
        mesh = self.robust_import_stl(file_path)
        if mesh is None:
            print(f"Error: Could not load STL file '{file_path}' with any method.")
            self.update_message(f"Failed to import STL file: {os.path.basename(file_path)}. Try re-exporting or converting to standard STL.")
            return False
        try:
            bounding_box = mesh.bounds
            dimensions = bounding_box[1] - bounding_box[0]
            self.mesh_dimensions = dimensions * self.global_scale
            print(f"Original STL dimensions: {dimensions} mm")
            print(f"Scaled STL dimensions: {self.mesh_dimensions} m")
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)
            center = np.mean(vertices, axis=0)
            vertices[:, 0] -= center[0]  # Center X
            vertices[:, 1] -= center[1]  # Center Y
            vertices[:, 2] -= center[2]  # Center Z
            vertices = vertices * self.global_scale
            base_name = os.path.basename(file_path)
            model_name = os.path.splitext(base_name)[0]
            color_name = None
            for color in ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'purple', 'orange', 'brown', 'pink']:
                if color in model_name.lower():
                    color_name = color
                    break
            if not color_name: color_name = 'gray'
            color_map = {
                'red': [1.0, 0.0, 0.0], 'green': [0.0, 1.0, 0.0], 'blue': [0.0, 0.0, 1.0],
                'yellow': [1.0, 1.0, 0.0], 'white': [1.0, 1.0, 1.0], 'black': [0.0, 0.0, 0.0],
                'gray': [0.7, 0.7, 0.7], 'purple': [0.5, 0.0, 0.5], 'orange': [1.0, 0.5, 0.0],
                'brown': [0.6, 0.3, 0.0], 'pink': [1.0, 0.7, 0.7]
            }
            color = color_map[color_name]
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
                color = np.mean(vertex_colors, axis=0).tolist()
            local_min_bounds = np.min(vertices, axis=0)
            local_max_bounds = np.max(vertices, axis=0)
            print(f"  - Calculated Local AABB (from path): Min={local_min_bounds}, Max={local_max_bounds}")
            mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'name': model_name,
                'dimensions': self.mesh_dimensions,
                'color': color,
                'color_name': color_name,
                 # --- ADDED: Store local bounds ---
                'local_min_bounds': local_min_bounds,
                'local_max_bounds': local_max_bounds
                # --- END ADDED ---
            }
            self.meshes.append(mesh_data)
            self.mesh_transforms.append({
                'position': np.array([0.0, -0.3, 0.0], dtype=np.float32), # Position at Y=-0.3 to avoid robot collision
                'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
            })
            mesh_id = len(self.meshes) - 1
            r = 0.3 + (mesh_id % 3) * 0.02
            g = 0.3 + ((mesh_id // 3) % 3) * 0.02
            b = 0.3 + ((mesh_id // 9) % 3) * 0.02
            self.mesh_selection_colors.append((r, g, b))
            self.mesh_file_paths.append(file_path)
            display_list = self.create_mesh_display_list(mesh_data)
            self.display_lists.append(display_list)
            self.selected_mesh_index = len(self.meshes) - 1
            self.update_message(f"Successfully loaded STL: {model_name}")
            print(f"Successfully loaded {file_path}")
            print(f"Model name: {model_name}")
            print(f"Model dimensions: {np.ptp(vertices, axis=0)}")
            print(f"Vertex count: {len(vertices)}")
            print(f"Face count: {len(faces)}")
            print(f"Total models loaded: {len(self.meshes)}")
            self.update_stl_models_in_tree()
            if hasattr(self.robot_viewer, 'update'):
                self.robot_viewer.update()
            return True
        except Exception as e:
            print(f"Error importing STL file from path {file_path}: {e}")
            import traceback
            traceback.print_exc()
            self.update_message(f"Error loading STL from path: {e}")
            return False

    def load_stl(self, file_path, custom_position=None, custom_rotation=None):
        """Load an STL file and add it to the scene
        
        Args:
            file_path: Path to the STL file
            custom_position: Optional [x, y, z] position in meters (default: [0.0, -0.3, 0.0])
            custom_rotation: Optional [rx, ry, rz] rotation in degrees (default: [0, 0, 0])
        """
        try:
            # Load the STL file
            mesh = trimesh.load(file_path)
            
            # Convert to numpy arrays
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)
            
            # Center the mesh
            center = np.mean(vertices, axis=0)
            vertices = vertices - center
            
            # Apply global scale
            vertices = vertices * self.global_scale
            
            # --- ADDED: Calculate and store local AABB ---
            local_min_bounds = np.min(vertices, axis=0)
            local_max_bounds = np.max(vertices, axis=0)
            print(f"  - Calculated Local AABB (load_stl): Min={local_min_bounds}, Max={local_max_bounds}")
            # --- END ADDED ---

            # Create mesh data
            mesh_data = {
                'name': os.path.basename(file_path),
                'vertices': vertices,
                'faces': faces,
                'path': file_path,
                'color': self.default_color,
                'color_name': 'gray',
                # --- ADDED: Store local bounds ---
                'local_min_bounds': local_min_bounds,
                'local_max_bounds': local_max_bounds
                # --- END ADDED ---
            }
            
            # Add to meshes list
            self.meshes.append(mesh_data)
            
            # Create display list for the mesh
            display_list = self.create_mesh_display_list(mesh_data)
            if display_list is None:
                print(f"Error: Failed to create display list for {file_path}")
                # Clean up potentially added mesh data
                self.meshes.pop()
                return False
                
            self.display_lists.append(display_list)
            
            # Determine position and orientation
            if custom_position is not None:
                position = custom_position
            else:
                position = [0.0, -0.3, 0.0]
            
            if custom_rotation is not None:
                # Convert rotation from degrees to quaternion
                from scipy.spatial.transform import Rotation
                r = Rotation.from_euler('xyz', custom_rotation, degrees=True)
                orientation = r.as_quat()  # Returns [x, y, z, w]
            else:
                orientation = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
            
            # Initialize transform for the new mesh
            self.mesh_transforms.append({
                'position': position,
                'orientation': orientation,
                'scale': [1.0, 1.0, 1.0]
            })
            
            # Store original values for reset functionality
            mesh_id = len(self.meshes) - 1
            if not hasattr(self, 'original_positions'): self.original_positions = {}
            if not hasattr(self, 'original_orientations'): self.original_orientations = {}
            if not hasattr(self, 'original_scales'): self.original_scales = {}
            if not hasattr(self, 'original_colors'): self.original_colors = {}
            self.original_positions[mesh_id] = np.array(position, dtype=np.float32).copy()
            self.original_orientations[mesh_id] = np.array(orientation, dtype=np.float32).copy()
            self.original_scales[mesh_id] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self.original_colors[mesh_id] = list(self.default_color)
            
            # Select the newly added mesh
            self.selected_mesh_index = len(self.meshes) - 1
            
            # Enable gizmo for the new mesh
            self.gizmo_state = 0  # 0 = translate mode
            
            # Update the tree view if available
            if self.program_tree and self.models_item:
                self.update_stl_models_in_tree()
            
            # Force redraw if we have a parent
            if self.robot_viewer and hasattr(self.robot_viewer, 'update'):
                self.robot_viewer.update()
            
            print(f"Successfully loaded STL: {os.path.basename(file_path)}")
            print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
            print(f"Size: {np.ptp(vertices, axis=0)}")
            
            # --- ADDED: Store the file path for saving --- #
            self.mesh_file_paths.append(file_path)
            # ------------------------------------------- #
            self.ensure_edges_for_all_meshes()

            return True
            
        except Exception as e:
            print(f"Error loading STL file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_stl_file(self, file_path):
        """Load an STL file and return its data."""
        try:
            # Load the STL file
            mesh = trimesh.load(file_path)
            
            # Create mesh data
            mesh_data = {
                'name': os.path.basename(file_path),
                'mesh': mesh,
                'path': file_path
            }
            
            return mesh_data
        except Exception as e:
            print(f"Error loading STL file: {e}")
            return None

    def import_stl_file(self, file_path):
        """Import an STL file and add it to the scene"""
        try:
            # Load the STL file
            mesh = trimesh.load(file_path)
            
            # Create a new object in the scene
            obj = self.scene.add_object()
            obj.mesh = mesh
            
            # Check if this is a conveyor belt
            if "Conveyor_belt.stl" in file_path:
                obj.is_conveyor_belt = True
                self.ui.conveyor_belt = obj
            else:
                obj.is_conveyor_belt = False
                obj.is_on_belt = False
            
            # Set initial position and orientation
            obj.position = [0, 0, 0]
            obj.rotation = [0, 0, 0]
            obj.update_transform()
            
            return obj
            
        except Exception as e:
            print(f"Error importing STL file: {e}")
            return None
    
    def place_object_on_belt(self, obj):
        """Place an object on the conveyor belt"""
        if self.ui.conveyor_belt is None:
            return False
            
        # Check if object is above the belt
        belt_pos = self.ui.conveyor_belt.position
        obj_pos = obj.position
        
        # Simple check: if object is above belt and close enough
        if (obj_pos[2] > belt_pos[2] and 
            abs(obj_pos[0] - belt_pos[0]) < 100 and 
            abs(obj_pos[1] - belt_pos[1]) < 100):
            obj.is_on_belt = True
            return True
            
        return False

    def select_edge(self, mesh_index, edge_index):
        """Sets the currently selected edge. Note: This is deprecated by handle_edge_click."""
        # This method might be unused now that handle_edge_click manages the list.
        # Consider removing or adapting if single-selection logic is still needed elsewhere.
        # For now, just log a warning if called.
        print("Warning: select_edge method called, but handle_edge_click should be used for multi-select.")

        if mesh_index is None or edge_index is None:
            # Clear selection if invalid indices provided
            if self.selected_edges: # Check if list is not empty
                self.selected_edges.clear()
                print("Cleared edge selection list (invalid index in select_edge).")
                if self.robot_viewer: self.robot_viewer.update()
                if hasattr(self.robot_viewer, 'edge_selection_changed'):
                     self.robot_viewer.edge_selection_changed.emit(False) # Emit False as selection is now empty
                     if hasattr(self.robot_viewer, '_last_edge_selection_state'):
                          self.robot_viewer._last_edge_selection_state = False
            return False

        if 0 <= mesh_index < len(self.meshes):
            mesh_data = self.meshes[mesh_index]
            if 'edges' in mesh_data and 0 <= edge_index < len(mesh_data['edges']):
                new_selection = (mesh_index, edge_index)
                # Replace the entire list with just this one selection
                self.selected_edges = [new_selection]
                print(f"Selected single edge via select_edge: {new_selection}")
                self.selected_mesh_index = mesh_index # Keep mesh selected
                self.update_message(f"Selected edge {edge_index} of model '{mesh_data.get('name', 'N/A')}'")
                if self.robot_viewer: self.robot_viewer.update()
                # Emit signal for successful selection
                if hasattr(self.robot_viewer, 'edge_selection_changed'):
                    self.robot_viewer.edge_selection_changed.emit(True) # Emit True as selection is not empty
                    if hasattr(self.robot_viewer, '_last_edge_selection_state'):
                         self.robot_viewer._last_edge_selection_state = True
                return True

        # If indices invalid, clear selection
        if self.selected_edges:
            self.selected_edges.clear()
            print("Cleared edge selection list (invalid index in select_edge - part 2).")
            if self.robot_viewer: self.robot_viewer.update()
            if hasattr(self.robot_viewer, 'edge_selection_changed'):
                 self.robot_viewer.edge_selection_changed.emit(False)
                 if hasattr(self.robot_viewer, '_last_edge_selection_state'):
                      self.robot_viewer._last_edge_selection_state = False
        return False

    def set_hover_edge(self, mesh_index, edge_index):
        """Set the currently hovered edge and trigger redraw."""
        if mesh_index is not None and edge_index is not None:
            new_hover_info = (mesh_index, edge_index)
        else:
            new_hover_info = None

        # Only update if the hover state actually changes
        if new_hover_info != self.hover_edge_info:
            old_hover_info = self.hover_edge_info
            self.hover_edge_info = new_hover_info
            
            # Only log significant hover changes (not clearing hover)
            if new_hover_info is not None:
                print(f"[Edge Hover] Hovering edge: mesh={mesh_index}, edge={edge_index}")
            elif old_hover_info is not None:
                print(f"[Edge Hover] Cleared hover from: mesh={old_hover_info[0]}, edge={old_hover_info[1]}")
            
            # Use request_immediate_update if available, otherwise fallback to update()
            if hasattr(self.robot_viewer, 'request_immediate_update'):
                self.robot_viewer.request_immediate_update() # Request redraw
            elif self.robot_viewer:
                self.robot_viewer.update()

    def handle_edge_click(self, mesh_idx, edge_idx, ctrl_pressed):
        """Handles logic for selecting/deselecting edges based on click and Ctrl key state."""
        edge_info = (mesh_idx, edge_idx)
        was_selected = edge_info in self.selected_edges
        selection_changed = False
        emit_signal = False # Flag to track if signal needs emitting

        if ctrl_pressed:
            if was_selected:
                self.selected_edges.remove(edge_info)
                selection_changed = True
                print(f"Removed edge {edge_info} from selection (Ctrl pressed).")
            else:
                self.selected_edges.append(edge_info)
                selection_changed = True
                print(f"Added edge {edge_info} to selection (Ctrl pressed).")
        else: # Ctrl not pressed
            if was_selected and len(self.selected_edges) == 1:
                # Clicked the only selected edge -> deselect it
                self.selected_edges.clear()
                selection_changed = True
                print(f"Deselected edge {edge_info} (was only selection).")
            elif not was_selected or len(self.selected_edges) > 1:
                # Clicked a new edge OR clicked one of many selected -> select only this one
                self.selected_edges = [edge_info]
                selection_changed = True
                print(f"Selected only edge {edge_info} (Ctrl not pressed).")

        print(f"Current selected edges: {self.selected_edges}")

        # Emit signal only if the overall selection state (empty vs non-empty) changed
        if selection_changed:
            has_selection = len(self.selected_edges) > 0
            # Check if signal emission is necessary
            last_state = None
            if hasattr(self.robot_viewer, '_last_edge_selection_state'):
                last_state = self.robot_viewer._last_edge_selection_state

            if last_state is None or last_state != has_selection:
                 emit_signal = True
                 # Store the new state immediately to prevent race conditions if emit is slow
                 if hasattr(self.robot_viewer, '_last_edge_selection_state'):
                     self.robot_viewer._last_edge_selection_state = has_selection


            if emit_signal:
                # <<< ADDED DEBUG PRINT >>>
                print(f"[STL Handler] Emitting edge_selection_changed signal with value: {has_selection}")
                if hasattr(self.robot_viewer, 'edge_selection_changed'):
                    self.robot_viewer.edge_selection_changed.emit(has_selection)
                else:
                     print("[STL Handler] Error: RobotViewer missing edge_selection_changed signal!")
            else:
                 print("[STL Handler] Selection changed, but state (empty/non-empty) is the same. No signal emitted.")


            # Update the viewer to reflect the change
            if self.robot_viewer:
                self.robot_viewer.update()

    def get_edge_at_cursor(self, x, y, width, height):
        """Use color picking to detect which edge is under the cursor.

        Args:
            x, y: Cursor position in window coordinates
            width, height: Window dimensions

        Returns:
            tuple: (mesh_index, edge_index) of the selected edge, or None if none
        """
        if not self.meshes:
            return None

        # print(f"Starting edge picking at ({x}, {y})")
        start_time = time.time()
        self.edge_selection_colors.clear() # Clear previous mapping

        # --- Render Edges with Unique Colors ---
        glPushAttrib(GL_ALL_ATTRIB_BITS) # Save more state
        glDisable(GL_DITHER)
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST) # Keep depth testing

        # Set up a small viewport for picking (use full viewport for accuracy?)
        # glViewport(x - 1, height - y - 1, 3, 3) # Small picking region
        glViewport(0, 0, width, height) # Full viewport
        # glClearColor(0.0, 0.0, 0.0, 0.0) # Clear to black (or a known non-ID color)
        glClearColor(1.0, 1.0, 1.0, 0.0) # MODIFIED: Clear to white
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Need to setup projection and modelview matrices same as paintGL
        # This ensures the picking render matches the visible scene
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        # --- Use parameters directly from RobotViewer --- 
        # Use getattr to safely access attributes that might not exist
        fov = getattr(self.robot_viewer, 'fov', 45.0) 
        aspect = width / height if height > 0 else 1.0
        near_clip = getattr(self.robot_viewer, 'near_clip', 0.1) 
        far_clip = getattr(self.robot_viewer, 'far_clip', 100.0)
        # Use the *actual* perspective setup from RobotViewer
        gluPerspective(fov, aspect, near_clip, far_clip)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        # --- Apply camera transform EXACTLY as in RobotViewer.paintGL --- 
        # Calculate eye position including offsets
        cam_dist = self.robot_viewer.camera_distance
        cam_elev = self.robot_viewer.camera_elevation
        cam_azim = self.robot_viewer.camera_azimuth
        base_eye_x = cam_dist * np.cos(np.radians(cam_elev)) * np.cos(np.radians(cam_azim))
        base_eye_y = cam_dist * np.cos(np.radians(cam_elev)) * np.sin(np.radians(cam_azim))
        base_eye_z = cam_dist * np.sin(np.radians(cam_elev))
        eye_x = base_eye_x + self.robot_viewer.camera_offset_x
        eye_y = base_eye_y + self.robot_viewer.camera_offset_y
        eye_z = base_eye_z + self.robot_viewer.camera_offset_z
        
        # Calculate target position including offsets
        target_x = self.robot_viewer.camera_target.x() + self.robot_viewer.camera_offset_x
        target_y = self.robot_viewer.camera_target.y() + self.robot_viewer.camera_offset_y
        target_z = self.robot_viewer.camera_target.z() + self.robot_viewer.camera_offset_z
        
        # Apply the exact same LookAt as paintGL
        gluLookAt(eye_x, eye_y, eye_z, 
                  target_x, target_y, target_z, 
                  0, 0, 1) # Assuming Up vector is always (0, 0, 1)

        # --- Draw ONLY edges with unique colors using the draw_meshes method --- 
        # print("Rendering edges for picking using draw_meshes(selection_mode='edge')...")
        self.draw_meshes(selection_mode='edge') # This handles iterating, transforms, and drawing edges with unique colors

        glFinish() # Ensure rendering is complete before reading pixels
        render_time = time.time() - start_time
        # print(f"Edge picking render done in {render_time:.4f}s")

        # --- Read the pixel color at cursor position ---
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1) # Ensure correct byte alignment
        # OpenGL Y coordinate is inverted from Qt Y
        pick_y = height - y - 1
        # Add boundary checks for x and pick_y
        if not (0 <= x < width and 0 <= pick_y < height):
             print(f"Warning: Pick coordinates ({x}, {pick_y}) outside viewport ({width}x{height}).")
             # Restore OpenGL state before returning
             glMatrixMode(GL_PROJECTION)
             glPopMatrix()
             glMatrixMode(GL_MODELVIEW)
             glPopMatrix()
             glPopAttrib()
             return None
        
        pixel_data = glReadPixels(x, pick_y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        read_time = time.time() - start_time - render_time
        # print(f"glReadPixels done in {read_time:.4f}s")

        r, g, b = bytes(pixel_data) # Get RGB values (0-255)

        # --- Restore OpenGL State ---
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib() # Restore all attributes

        # --- Decode the color ---
        picked_id = (r << 16) | (g << 8) | b
        # print(f"Read color: R={r}, G={g}, B={b} -> ID={picked_id}") # More detailed print

        decode_start_time = time.time()
        # Check if the picked ID exists in our mapping
        # Use self.edge_selection_colors which is populated by draw_meshes in 'edge' mode
        # MODIFIED: Check for background color (white) first
        if picked_id == 0xFFFFFF: # If white background (255,255,255) was picked
            # print("Picked background (white).")
            result = None
        else:
            result = self.edge_selection_colors.get(picked_id)

        if result:
            mesh_idx, edge_idx = result
            # print(f"Decoded ID {picked_id} to Mesh {mesh_idx}, Edge {edge_idx}")
            decode_time = time.time() - decode_start_time
            # print(f"Decoding done in {decode_time:.4f}s")
            return result
        else:
            # if picked_id != 0xFFFFFF: # Avoid redundant print if it was already handled by the background check
                # print(f"Picked color ID {picked_id} not in edge_selection_colors map or it was background.")
            decode_time = time.time() - decode_start_time
            # print(f"Decoding done (no match) in {decode_time:.4f}s")
            return None

    # --- End Methods for AI Control ---

    # --- ADDED: Method to draw bounding boxes ---
    def draw_bounding_boxes(self):
        """Draws the Oriented Bounding Boxes (OBBs) for all loaded meshes."""
        if not self.show_bounding_boxes or not self.meshes:
            return

        # Get the set of indices to highlight
        highlight_indices = self.highlight_bbox_indices

        glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT | GL_POLYGON_BIT)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST) # Draw on top
        glLineWidth(1.5)
        # Default color set inside the loop now

        for i, mesh_data in enumerate(self.meshes):
            if 'local_min_bounds' not in mesh_data or 'local_max_bounds' not in mesh_data:
                continue # Skip if bounds weren't stored

            # Set color based on highlight state
            if i in highlight_indices:
                glColor3f(1.0, 0.0, 0.0) # Red for objects ON conveyor
            else:
                glColor3f(0.3, 1.0, 0.3) # Bright green for others

            local_min = mesh_data['local_min_bounds']
            local_max = mesh_data['local_max_bounds']

            # Get the current world transform for this mesh
            if i >= len(self.mesh_transforms): continue
            transform_data = self.mesh_transforms[i]
            world_pos = np.array(transform_data['position'])
            world_orient_quat = transform_data['orientation']
            world_scale = np.array(transform_data['scale'])

            # Create full 4x4 transform matrix
            try:
                # from scipy.spatial.transform import Rotation as R_sps # REMOVED LOCAL IMPORT
                rotation = R_sps.from_quat(world_orient_quat)
                rot_matrix = rotation.as_matrix()

                transform_matrix = np.identity(4)
                transform_matrix[:3, :3] = rot_matrix @ np.diag(world_scale) # Apply scale before rotation for OBB
                transform_matrix[:3, 3] = world_pos
            except Exception as e:
                # print(f"Error creating transform matrix for bbox {i}: {e}")
                continue # Skip if transform fails

            # Define the 8 corners of the local AABB
            corners_local = [
                np.array([local_min[0], local_min[1], local_min[2], 1.0]), # 0: ---
                np.array([local_max[0], local_min[1], local_min[2], 1.0]), # 1: +--
                np.array([local_max[0], local_max[1], local_min[2], 1.0]), # 2: ++-
                np.array([local_min[0], local_max[1], local_min[2], 1.0]), # 3: -+-
                np.array([local_min[0], local_min[1], local_max[2], 1.0]), # 4: --+
                np.array([local_max[0], local_min[1], local_max[2], 1.0]), # 5: +-+
                np.array([local_max[0], local_max[1], local_max[2], 1.0]), # 6: +++
                np.array([local_min[0], local_max[1], local_max[2], 1.0])  # 7: -++
            ]

            # Transform corners to world space
            corners_world = [(transform_matrix @ c)[:3] for c in corners_local]

            # Define edges connecting the corners
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4), # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
            ]

            # Draw the edges
            glBegin(GL_LINES)
            for edge in edges:
                glVertex3fv(corners_world[edge[0]])
                glVertex3fv(corners_world[edge[1]])
            glEnd()

        glPopAttrib()
    # --- END ADDED ---

    # --- ADD NEW METHOD HERE ---
    def set_model_position_directly(self, index, absolute_position_vector):
        """Sets the absolute world position of the model at the specified index.

        Args:
            index (int): The index of the model to move.
            absolute_position_vector (list or np.ndarray): The target absolute position [x, y, z].

        Returns:
            bool: True if setting position was successful, False otherwise.
        """
        if not (0 <= index < len(self.mesh_transforms)):
            print(f"Error: Invalid index {index} for set_model_position_directly.")
            self.update_message(f"Error: Cannot set position for invalid index {index}.")
            return False

        try:
            target_position = np.array(absolute_position_vector, dtype=np.float32)
            if target_position.shape != (3,):
                raise ValueError("Absolute position vector must have 3 elements.")

            # Apply global scale to convert input (assumed MM) to internal units (Meters)
            # if global_scale is 0.001, 500mm -> 0.5m
            scaled_position = target_position * self.global_scale

            # Directly set the position
            self.mesh_transforms[index]['position'] = scaled_position

            # Don't print every time text changes, rely on viewer update
            # print(f"Set model {index} position to: {target_position}")
            # self.update_message(f"Set model {index} position to {target_position}.")

            # Force viewer redraw (might already be handled by caller)
            # if self.robot_viewer:
            #     self.robot_viewer.update()

            return True

        except Exception as e:
            print(f"Error during set_model_position_directly for index {index}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_message(f"Error setting model {index} position: {str(e)}")
            return False
    # --- END NEW METHOD ---

    def get_model_transform_by_index_or_name(self, index_or_name):
        """
        Retrieves the transformation data for a model specified by its index or name.

        Args:
            index_or_name (int or str): The index or name of the model.

        Returns:
            dict or None: A dictionary containing 'name', 'position', 'orientation' (quaternion), 
                          and 'scale' if found, otherwise None.
        """
        mesh_index = -1
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(self.meshes):
                mesh_index = index_or_name
        elif isinstance(index_or_name, str):
            for i, mesh_data_iter in enumerate(self.meshes):
                if mesh_data_iter.get('name') == index_or_name:
                    mesh_index = i
                    break
        else: # Neither int nor string
            self.update_message(f"Error: Invalid type for model identifier '{index_or_name}' ({type(index_or_name)}). Expected int or str.", "error")
            return None

        if mesh_index == -1:
            self.update_message(f"Error: Model '{index_or_name}' not found for transform retrieval.", "error")
            return None

        if mesh_index >= len(self.mesh_transforms) or mesh_index >= len(self.meshes):
            self.update_message(f"Error: Data inconsistency for model index {mesh_index} (transform).", "error")
            return None

        transform = self.mesh_transforms[mesh_index]
        mesh_name = self.meshes[mesh_index].get('name', f"Model_{mesh_index}")

        # Convert position back to MM for external usage
        # Position is stored in Meters (internally), so divide by scale (0.001) -> multiply by 1000
        position_m = transform['position'].tolist() if isinstance(transform.get('position'), np.ndarray) else transform.get('position', [0.0, 0.0, 0.0])
        position_mm = [p / self.global_scale for p in position_m]

        return {
            'name': mesh_name,
            'position': position_mm, # Return MM
            'orientation': transform['orientation'].tolist() if isinstance(transform.get('orientation'), np.ndarray) else transform.get('orientation', [0.0, 0.0, 0.0, 1.0]), # quat [x,y,z,w]
            'scale': transform['scale'].tolist() if isinstance(transform.get('scale'), np.ndarray) else transform.get('scale', [1.0, 1.0, 1.0])
        }

    def get_model_bounds_by_index_or_name(self, index_or_name):
        """
        Retrieves the world-axis-aligned bounding box for a model specified by index or name.

        Args:
            index_or_name (int or str): The index or name of the model.

        Returns:
            dict or None: A dictionary containing 'name', 'min_point', and 'max_point' 
                          (world coordinates) if found, otherwise None.
        """
        mesh_index = -1
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(self.meshes):
                mesh_index = index_or_name
        elif isinstance(index_or_name, str):
            for i, mesh_data_iter in enumerate(self.meshes):
                if mesh_data_iter.get('name') == index_or_name:
                    mesh_index = i
                    break
        else: # Neither int nor string
            self.update_message(f"Error: Invalid type for model identifier '{index_or_name}' ({type(index_or_name)}). Expected int or str.", "error")
            return None
            
        if mesh_index == -1:
            self.update_message(f"Error: Model '{index_or_name}' not found for bounds retrieval.", "error")
            return None

        if mesh_index >= len(self.meshes) or mesh_index >= len(self.mesh_transforms):
            self.update_message(f"Error: Data inconsistency for model index {mesh_index} (bounds).", "error")
            return None

        mesh_data = self.meshes[mesh_index]
        transform_data = self.mesh_transforms[mesh_index]
        model_name = mesh_data.get('name', f"Model_{mesh_index}")

        # Local bounds are from the mesh data (already affected by global_scale during import/add)
        local_min = np.array(mesh_data.get('local_min_bounds', [0.0, 0.0, 0.0]))
        local_max = np.array(mesh_data.get('local_max_bounds', [0.0, 0.0, 0.0]))

        # Define the 8 corners of the local AABB
        corners_local_aabb = [
            np.array([local_min[0], local_min[1], local_min[2], 1.0]),
            np.array([local_max[0], local_min[1], local_min[2], 1.0]),
            np.array([local_max[0], local_max[1], local_min[2], 1.0]),
            np.array([local_min[0], local_max[1], local_min[2], 1.0]),
            np.array([local_min[0], local_min[1], local_max[2], 1.0]),
            np.array([local_max[0], local_min[1], local_max[2], 1.0]),
            np.array([local_max[0], local_max[1], local_max[2], 1.0]),
            np.array([local_min[0], local_max[1], local_max[2], 1.0])
        ]

        # Get world transformation components
        world_pos = np.array(transform_data['position'])
        world_orient_quat = np.array(transform_data['orientation']) # Expected as [x,y,z,w]
        mesh_specific_scale = np.array(transform_data['scale'])

        # NOTE: get_model_bounds returns WORLD coordinates.
        # Since our internal world is Scaled (Meters), we must convert the result back to MM
        # if the caller expects MM (which they likely do if they are plotting in the UI or doing collision checks in MM space).
        
        # However, checking where bounds are used is important. If used for raycasting in the meter-scene, we shouldn't convert.
        # BUT this method is `get_model_bounds_by_index_or_name`, likely for external usage/UI.
        # Let's standardize on MM.

        # ... calculation logic ...
        # (This tool usage is tricky for replacing large logic blocks, I will just wrap the return)
        
        # ACTUALLY, I should iterate corners and convert them at the end.
        # But I can't easily replace the whole function body safely with this tool if I don't confirm the middle content.
        # I will leave get_model_bounds unchanged for now OR assume I can reconstruct it.
        # Let's look at the return logic specifically.
        
        # The code calculates `corners_world`. 
        # I can just add scaling at the end.
        
        # Original:
        # min_point = np.min(corners_world, axis=0).tolist()
        # max_point = np.max(corners_world, axis=0).tolist()
        
        # Helper: I will replace the calculation lines around 4720+ (Need to view to be sure, or guess).
        # Actually I viewed down to 4700 in Step 406. I need to view 4700+ to see the end of `get_model_bounds_by_index_or_name`.
        # I'll enable valid output for this block by reading lines 4700-4750 first.


        try:
            # scipy.spatial.transform.Rotation is already imported in this file (used in draw_bounding_boxes)
            # from scipy.spatial.transform import Rotation as R_sps # Ensure R_sps is defined or use full path
            # Assuming R_sps is available or direct call to Rotation is fine
            rotation = R_sps.from_quat(world_orient_quat) 
            rot_matrix_3x3 = rotation.as_matrix()

            # Build the full 4x4 world transform matrix: T * R * S
            # S: applies mesh_specific_scale to the local AABB corners
            # R: applies world_orient_quat
            # T: applies world_pos

            scale_matrix_4x4 = np.diag(np.append(mesh_specific_scale, 1.0))

            rot_matrix_4x4 = np.identity(4)
            rot_matrix_4x4[:3, :3] = rot_matrix_3x3

            trans_matrix_4x4 = np.identity(4)
            trans_matrix_4x4[:3, 3] = world_pos
            
            # Combined transform: First scale the local AABB points, then rotate, then translate
            world_transform_matrix = trans_matrix_4x4 @ rot_matrix_4x4 @ scale_matrix_4x4

        except Exception as e:
            self.update_message(f"Error creating transform matrix for bounds of '{model_name}': {e}", "error")
            import traceback
            traceback.print_exc()
            return None

        # Transform local AABB corners to world space
        corners_world_homogeneous = [world_transform_matrix @ c for c in corners_local_aabb]
        corners_world = [c_hom[:3]/c_hom[3] if c_hom[3] != 0 else c_hom[:3] for c_hom in corners_world_homogeneous] # Dehomogenize


        if not corners_world or len(corners_world) != 8:
            self.update_message(f"Error: Could not transform AABB corners for '{model_name}'.", "error")
            return None

        # Calculate world AABB from the 8 transformed corners
        world_min_bounds = np.min(corners_world, axis=0)
        world_max_bounds = np.max(corners_world, axis=0)

        # Convert back to MM if global_scale is used (Meters -> MM)
        # We divide by scale (0.001) which is same as multiplying by 1000
        min_point_mm = (world_min_bounds / self.global_scale).tolist()
        max_point_mm = (world_max_bounds / self.global_scale).tolist()

        return {
            'name': model_name,
            'min_point': min_point_mm,
            'max_point': max_point_mm
        }

    @property
    def mesh_names(self):
        """Return a list of mesh names for compatibility with draw_stl_meshes."""
        return [mesh.get('name', f'Model_{i}') for i, mesh in enumerate(self.meshes)]

    def ensure_edges_for_all_meshes(self):
        """Ensure every mesh in self.meshes has its 'edges' field populated."""
        import trimesh
        for mesh_data in self.meshes:
            if 'edges' not in mesh_data or mesh_data['edges'] is None:
                vertices = mesh_data['vertices']
                faces = mesh_data['faces']
                temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                edges_unique_indices = temp_mesh.edges_unique
                edge_vertices = vertices[edges_unique_indices]
                mesh_data['edges'] = edge_vertices

    def scale_model_directly(self, index, scale_factors):
        """
        Scale a model directly by index without showing a dialog.
        
        Args:
            index (int): Index of the model to scale
            scale_factors (list): [sx, sy, sz] scale factors
            
        Returns:
            bool: True if successful, False otherwise
        """
        if index < 0 or index >= len(self.meshes):
            self.update_message(f"Error: Model index {index} is out of range.")
            return False
            
        try:
            sx, sy, sz = scale_factors
            
            # Update the transform
            transform = self.mesh_transforms[index]
            transform['scale'] = np.array([sx, sy, sz], dtype=np.float32)
            
            # Update the message
            model_name = self.meshes[index].get('name', f'Model_{index}')
            self.update_message(f"Scaled model {model_name} by ({sx}, {sy}, {sz})")
            
            # Force a redraw
            if self.robot_viewer:
                self.robot_viewer.update()
                
            return True
            
        except Exception as e:
            self.update_message(f"Error scaling model {index}: {str(e)}")
            return False

    def color_model_directly(self, index, color_name):
        """
        Color a model directly by index and color name without showing a dialog.
        
        Args:
            index (int): Index of the model to color
            color_name (str): Name of the color (e.g., 'red', 'blue', 'green')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if index < 0 or index >= len(self.meshes):
            self.update_message(f"Error: Model index {index} is out of range.")
            return False
            
        # Color name to RGB mapping (normalized to [0-1] range)
        color_map = {
            'red': [1.0, 0.0, 0.0], 'green': [0.0, 1.0, 0.0], 'blue': [0.0, 0.0, 1.0],
            'yellow': [1.0, 1.0, 0.0], 'white': [1.0, 1.0, 1.0], 'black': [0.0, 0.0, 0.0],
            'gray': [0.7, 0.7, 0.7], 'purple': [0.5, 0.0, 0.5], 'orange': [1.0, 0.5, 0.0],
            'brown': [0.6, 0.3, 0.0], 'pink': [1.0, 0.7, 0.7], 'cyan': [0.0, 1.0, 1.0],
            'magenta': [1.0, 0.0, 1.0]
        }
        
        # Normalize color name to lowercase
        color_name = color_name.lower()
        
        if color_name not in color_map:
            self.update_message(f"Error: Unknown color '{color_name}'. Available colors: {', '.join(color_map.keys())}")
            return False
            
        try:
            # Get the RGB color values
            new_color = color_map[color_name]
            
            # Update the mesh color
            self.meshes[index]['color'] = new_color
            self.meshes[index]['color_name'] = color_name
            
            # Delete old display list
            if index < len(self.display_lists):
                old_display_list = self.display_lists[index]
                if old_display_list:
                    glDeleteLists(old_display_list, 1)
                
            # Create new display list with updated color
            new_display_list = self.create_mesh_display_list(self.meshes[index])
            self.display_lists[index] = new_display_list
            
            # Update the message
            model_name = self.meshes[index].get('name', f'Model_{index}')
            self.update_message(f"Colored model {model_name} to {color_name}")
            
            # Re-register objects with AI controller to update color mappings
            if hasattr(self.robot_viewer, 'ai_controller'):
                self.robot_viewer.ai_controller.register_imported_objects()
            
            # Force a redraw
            if self.robot_viewer:
                self.robot_viewer.update()
                
            return True
            
        except Exception as e:
            self.update_message(f"Error coloring model {index}: {str(e)}")
            return False

    def configure_sensor(self, item, model_index):
        """Open sensor configuration dialog"""
        if model_index is None or model_index < 0 or model_index >= len(self.meshes):
            return
        
        mesh_data = self.meshes[model_index]
        sensor_name = mesh_data.get('name', 'Sensor')
        
        # Import sensor config dialog
        from sensor_config_dialog import SensorConfigDialog
        
        # Get current configuration from parent if available
        current_config = {}
        if hasattr(self.robot_viewer, 'parent_widget') and self.robot_viewer.parent_widget:
            parent = self.robot_viewer.parent_widget
            if hasattr(parent, 'sensor_configs'):
                current_config = parent.sensor_configs.get(sensor_name, {})
        
        # If no config exists, create default
        if not current_config:
            current_config = {
                'trigger_distance': 50.0,
                'show_visualization': True
            }
        
        # Show dialog with correct signature: (sensor_name, sensor_config, parent)
        dialog = SensorConfigDialog(sensor_name, current_config, self.program_tree)
        
        if dialog.exec_() == QDialog.Accepted:
            # Get new configuration from dialog
            new_config = dialog.get_config()
            
            # Store configuration in parent
            if hasattr(self.robot_viewer, 'parent_widget') and self.robot_viewer.parent_widget:
                parent = self.robot_viewer.parent_widget
                if not hasattr(parent, 'sensor_configs'):
                    parent.sensor_configs = {}
                parent.sensor_configs[sensor_name] = new_config
                
                print(f"[SENSOR CONFIG] Updated {sensor_name}: distance={new_config.get('trigger_distance', 50)}mm, viz={new_config.get('show_visualization', True)}")
                
                # Update message
                if hasattr(parent, 'update_message'):
                    parent.update_message(f"Sensor '{sensor_name}' configured: {new_config.get('trigger_distance', 50)}mm trigger distance", "success")
