import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
                            QDoubleSpinBox, QPushButton, QProgressBar, QComboBox,
                            QCheckBox, QGroupBox, QSlider, QFileDialog, QMessageBox,
                            QTabWidget, QWidget, QFormLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from scipy.spatial.transform import Rotation

# Apply collections compatibility fixes for Python 3.11+
import collections_compat

import trimesh
import os
import time
import math
import traceback
from stl_slicer import STLSlicer

class SlicerDialog(QDialog):
    """Dialog for 3D printing settings and slicing control"""
    
    # Signals to communicate with main application
    slicing_complete = pyqtSignal(object)  # Emitted when slicing is complete, passes all toolpaths
    slicing_progress = pyqtSignal(int)     # Progress percentage
    slicing_status = pyqtSignal(str)       # Status message
    start_printing_signal = pyqtSignal(list)  # Signal to start printing with targets
    targets_created = pyqtSignal(list)       # Emitted after robot targets are created
    
    def __init__(self, stl_handler, robot_programmer, parent=None):
        super().__init__(parent)
        self.stl_handler = stl_handler
        self.robot_programmer = robot_programmer
        self.parent = parent
        self.selected_model_index = -1
        self.toolpaths = []
        self.slicer = None
        self.octoprint_handler = None
        
        # Connect signals
        if hasattr(self.parent, 'start_3d_print_execution'):
            self.start_printing_signal.connect(self.parent.start_3d_print_execution)
        
        # Verify robot_programmer is valid
        if self.robot_programmer:
            print(f"Robot programmer provided: {type(self.robot_programmer)}")
            # Check essential attributes
            if not hasattr(self.robot_programmer, 'program'):
                print("WARNING: robot_programmer does not have 'program' attribute")
                self.robot_programmer.program = {"name": "3D Print", "targets": []}
            if not hasattr(self.robot_programmer, 'clear_all_targets'):
                print("WARNING: robot_programmer does not have 'clear_all_targets' method")
            if not hasattr(self.robot_programmer, 'update_tree_from_program'):
                print("WARNING: robot_programmer does not have 'update_tree_from_program' method")
            if not hasattr(self.robot_programmer, 'run_program'):
                print("WARNING: robot_programmer does not have 'run_program' method")
        else:
            print("WARNING: No robot_programmer provided")
        
        # Default settings
        self.layer_height = 0.2  # mm
        self.infill_percentage = 20  # %
        self.wall_count = 2  # number of walls
        self.print_speed = 20  # mm/s
        self.travel_speed = 80  # mm/s
        self.nozzle_diameter = 0.4  # mm
        self.filament_diameter = 1.75  # mm
        self.bed_temperature = 60  # °C
        self.nozzle_temperature = 200  # °C
        self.retraction_distance = 5  # mm
        self.retraction_speed = 40  # mm/s
        self.fan_speed = 100  # %
        self.brim_width = 0  # mm
        self.raft_layers = 0  # number of layers
        
        # Add Z offset for first layer
        self.first_layer_height = 0.3  # mm
        self.first_layer_speed = 15  # mm/s
        self.z_offset = 0.0  # mm
        
        # Additional parameters
        self.use_octoprint = False
        
        # Initialize OctoPrint handler if needed
        try:
            from octoprint_handler import OctoPrintHandler
            self.octoprint_handler = OctoPrintHandler()
        except ImportError:
            self.octoprint_handler = None
            print("OctoPrint handler not available")
        
        # Create the slicing bridge for thread communication
        from PyQt5.QtCore import QObject, pyqtSignal
        class SlicingCompleteBridge(QObject):
            success_signal = pyqtSignal(bool, str)
        self.slicing_bridge = SlicingCompleteBridge()
        self.slicing_bridge.success_signal.connect(self._on_slicing_complete_from_signal)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI for the slicer dialog"""
        self.setWindowTitle("3D Printing Slicer")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.setMaximumHeight(700)  # Limit dialog height to prevent full screen
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tabs for different setting groups
        self.tabs = QTabWidget()
        
        # Create tabs
        self.main_tab = QWidget()
        self.advanced_tab = QWidget()
        self.preview_tab = QWidget()
        
        # Add tabs to tab widget
        self.tabs.addTab(self.main_tab, "Basic Settings")
        self.tabs.addTab(self.advanced_tab, "Advanced Settings")
        self.tabs.addTab(self.preview_tab, "Layer Preview")
        
        # Set up each tab
        self.setup_main_tab()
        self.setup_advanced_tab()
        self.setup_preview_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Progress bar (bottom of dialog)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to slice")
        main_layout.addWidget(self.status_label)
        
        # Buttons at bottom
        button_layout = QHBoxLayout()
        
        # Change the slice_button text and logic
        self.slice_button = QPushButton("Slice Model and Create Targets")
        self.slice_button.clicked.connect(self.slice_model)
        button_layout.addWidget(self.slice_button)
        
        self.print_button = QPushButton("Start 3D Print")
        self.print_button.setEnabled(False)
        self.print_button.clicked.connect(self.start_printing)
        button_layout.addWidget(self.print_button)
        
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        
        # Connect signals
        self.slicing_progress.connect(self.update_progress)
        self.slicing_status.connect(self.update_status)
        
    def setup_main_tab(self):
        """Set up the main settings tab"""
        layout = QVBoxLayout(self.main_tab)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.refresh_model_list()
        model_layout.addWidget(QLabel("Select STL model:"))
        model_layout.addWidget(self.model_combo)
        # Disconnect and reconnect to prevent signals during setup
        self.model_combo.currentIndexChanged.connect(self.on_model_selection_changed)
        model_group.setLayout(model_layout)
        
        # If we have models, select the first one by default
        if self.model_combo.count() > 0 and self.selected_model_index < 0:
            self.model_combo.setCurrentIndex(0)
            self.selected_model_index = 0
        
        layout.addWidget(model_group)
        
        # Print parameters
        params_group = QGroupBox("Print Parameters")
        params_layout = QFormLayout()
        
        # Layer height
        self.layer_height_input = QDoubleSpinBox()
        self.layer_height_input.setRange(0.1, 5.0)  # Allow layer heights from 0.1mm to 5mm
        self.layer_height_input.setSingleStep(0.1)
        self.layer_height_input.setDecimals(2)  # Show 2 decimal places
        self.layer_height_input.setValue(self.layer_height)
        params_layout.addRow("Layer Height (mm):", self.layer_height_input)
        
        # First layer height
        self.first_layer_height_input = QDoubleSpinBox()
        self.first_layer_height_input.setRange(0.1, 5.0)  # Allow first layer heights from 0.1mm to 5mm
        self.first_layer_height_input.setSingleStep(0.1)
        self.first_layer_height_input.setDecimals(2)  # Show 2 decimal places
        self.first_layer_height_input.setValue(self.first_layer_height)
        params_layout.addRow("First Layer Height (mm):", self.first_layer_height_input)
        
        # Z offset
        self.z_offset_input = QDoubleSpinBox()
        self.z_offset_input.setRange(-2.0, 2.0)
        self.z_offset_input.setSingleStep(0.05)
        self.z_offset_input.setValue(self.z_offset)
        params_layout.addRow("Z Offset (mm):", self.z_offset_input)
        
        # Infill percentage
        self.infill_input = QSpinBox()
        self.infill_input.setRange(0, 100)
        self.infill_input.setSingleStep(5)
        self.infill_input.setValue(self.infill_percentage)
        params_layout.addRow("Infill Percentage (%):", self.infill_input)
        
        # Wall count
        self.wall_input = QSpinBox()
        self.wall_input.setRange(1, 5)
        self.wall_input.setValue(self.wall_count)
        params_layout.addRow("Wall Count:", self.wall_input)
        
        # Nozzle diameter
        self.nozzle_input = QDoubleSpinBox()
        self.nozzle_input.setRange(0.1, 1.0)
        self.nozzle_input.setSingleStep(0.1)
        self.nozzle_input.setValue(self.nozzle_diameter)
        params_layout.addRow("Nozzle Diameter (mm):", self.nozzle_input)
        
        # Print speed
        self.speed_input = QSpinBox()
        self.speed_input.setRange(5, 150)
        self.speed_input.setSingleStep(5)
        self.speed_input.setValue(self.print_speed)
        params_layout.addRow("Print Speed (mm/s):", self.speed_input)
        
        # First layer speed
        self.first_layer_speed_input = QSpinBox()
        self.first_layer_speed_input.setRange(5, 50)
        self.first_layer_speed_input.setSingleStep(5)
        self.first_layer_speed_input.setValue(self.first_layer_speed)
        params_layout.addRow("First Layer Speed (mm/s):", self.first_layer_speed_input)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # OctoPrint option
        if self.octoprint_handler:
            octoprint_group = QGroupBox("OctoPrint Integration")
            octoprint_layout = QVBoxLayout()
            
            self.use_octoprint_checkbox = QCheckBox("Send to OctoPrint after slicing")
            self.use_octoprint_checkbox.setChecked(self.use_octoprint)
            octoprint_layout.addWidget(self.use_octoprint_checkbox)
            
            self.octoprint_setup_button = QPushButton("OctoPrint Setup")
            self.octoprint_setup_button.clicked.connect(self.show_octoprint_setup)
            octoprint_layout.addWidget(self.octoprint_setup_button)
            
            octoprint_group.setLayout(octoprint_layout)
            layout.addWidget(octoprint_group)
        
    def setup_advanced_tab(self):
        """Set up the advanced settings tab"""
        layout = QVBoxLayout(self.advanced_tab)
        
        # Advanced print parameters
        advanced_form = QFormLayout()
        
        # Filament diameter
        self.filament_input = QDoubleSpinBox()
        self.filament_input.setRange(1.0, 3.0)
        self.filament_input.setSingleStep(0.05)
        self.filament_input.setValue(self.filament_diameter)
        advanced_form.addRow("Filament Diameter (mm):", self.filament_input)
        
        # Travel speed
        self.travel_speed_input = QSpinBox()
        self.travel_speed_input.setRange(20, 300)
        self.travel_speed_input.setSingleStep(10)
        self.travel_speed_input.setValue(self.travel_speed)
        advanced_form.addRow("Travel Speed (mm/s):", self.travel_speed_input)
        
        # Temperatures
        temp_layout = QHBoxLayout()
        self.nozzle_temp_input = QSpinBox()
        self.nozzle_temp_input.setRange(150, 300)
        self.nozzle_temp_input.setSingleStep(5)
        self.nozzle_temp_input.setValue(self.nozzle_temperature)
        advanced_form.addRow("Nozzle Temperature (°C):", self.nozzle_temp_input)
        
        self.bed_temp_input = QSpinBox()
        self.bed_temp_input.setRange(0, 120)
        self.bed_temp_input.setSingleStep(5)
        self.bed_temp_input.setValue(self.bed_temperature)
        advanced_form.addRow("Bed Temperature (°C):", self.bed_temp_input)
        
        # Retraction settings
        self.retraction_input = QDoubleSpinBox()
        self.retraction_input.setRange(0, 10)
        self.retraction_input.setSingleStep(0.5)
        self.retraction_input.setValue(self.retraction_distance)
        advanced_form.addRow("Retraction Distance (mm):", self.retraction_input)
        
        self.retraction_speed_input = QSpinBox()
        self.retraction_speed_input.setRange(10, 150)
        self.retraction_speed_input.setSingleStep(5)
        self.retraction_speed_input.setValue(self.retraction_speed)
        advanced_form.addRow("Retraction Speed (mm/s):", self.retraction_speed_input)
        
        # Fan speed
        fan_layout = QHBoxLayout()
        self.fan_input = QSlider(Qt.Horizontal)
        self.fan_input.setRange(0, 100)
        self.fan_input.setValue(self.fan_speed)
        fan_layout.addWidget(self.fan_input)
        self.fan_value_label = QLabel(f"{self.fan_speed}%")
        fan_layout.addWidget(self.fan_value_label)
        self.fan_input.valueChanged.connect(self.update_fan_value_label)
        advanced_form.addRow("Fan Speed (%):", fan_layout)
        
        # Brim/Raft settings
        self.brim_input = QDoubleSpinBox()
        self.brim_input.setRange(0, 20)
        self.brim_input.setSingleStep(1)
        self.brim_input.setValue(self.brim_width)
        advanced_form.addRow("Brim Width (mm):", self.brim_input)
        
        self.raft_input = QSpinBox()
        self.raft_input.setRange(0, 3)
        self.raft_input.setValue(self.raft_layers)
        advanced_form.addRow("Raft Layers:", self.raft_input)
        
        layout.addLayout(advanced_form)
        
    def setup_preview_tab(self):
        """Set up the layer preview tab"""
        layout = QVBoxLayout(self.preview_tab)
        
        # Layer selector slider
        layer_control_layout = QHBoxLayout()
        layer_control_layout.addWidget(QLabel("Layer:"))
        self.layer_selector = QSlider(Qt.Horizontal)
        self.layer_selector.setEnabled(False)
        self.layer_selector.setMinimum(0)
        self.layer_selector.setMaximum(0)
        self.layer_selector.valueChanged.connect(lambda value: self.update_layer_preview(value))
        self.layer_selector.setTracking(True)  # Update while dragging
        layer_control_layout.addWidget(self.layer_selector)
        self.layer_number_label = QLabel("No layers")
        layer_control_layout.addWidget(self.layer_number_label)
        layout.addLayout(layer_control_layout)
        
        # Layer preview area
        self.preview_label = QLabel("No layers to preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(400)
        self.preview_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.preview_label)
        
        # Statistics
        self.stats_label = QLabel("Layer statistics will appear here after slicing")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        
    def update_fan_value_label(self, value):
        """Update the fan speed label with the current value"""
        self.fan_value_label.setText(f"{value}%")
        
    def refresh_model_list(self):
        """Refresh the list of available STL models"""
        self.model_combo.clear()
        
        # Check if stl_handler is available
        if not self.stl_handler or not hasattr(self.stl_handler, 'meshes'):
            self.model_combo.addItem("No STL models loaded")
            # Ensure slice_button exists before accessing it
            if hasattr(self, 'slice_button'):
                self.slice_button.setEnabled(False)
            self.selected_model_index = -1
            return
            
        # Get list of models from stl_handler
        models = self.stl_handler.meshes
        
        if not models:
            self.model_combo.addItem("No STL models loaded")
            # Ensure slice_button exists before accessing it
            if hasattr(self, 'slice_button'):
                self.slice_button.setEnabled(False)
            self.selected_model_index = -1
            return
            
        # Add models to combo box
        for i, model in enumerate(models):
            self.model_combo.addItem(f"{model['name']} ({i})")
            
        # Select the first model by default
        if self.model_combo.count() > 0:
            self.selected_model_index = 0
            # This will trigger on_model_selection_changed
            self.model_combo.setCurrentIndex(0)
            
        # Enable slice button
        # Ensure slice_button exists before accessing it
        if hasattr(self, 'slice_button'):
            self.slice_button.setEnabled(True)
        
    def on_model_selection_changed(self, index):
        """Handle model selection change"""
        print(f"Model selection changed to index: {index}")
        
        if index < 0:
            print("Invalid model index (negative)")
            self.selected_model_index = -1
            return
            
        if not self.stl_handler:
            print("STL handler not available")
            return
            
        if not hasattr(self.stl_handler, 'meshes'):
            print("STL handler has no meshes attribute")
            return
            
        if index >= len(self.stl_handler.meshes):
            print(f"Index {index} out of range (mesh count: {len(self.stl_handler.meshes)})")
            return
        
        # Valid selection, update the index
        self.selected_model_index = index
        model_name = self.stl_handler.meshes[index]['name']
        print(f"Selected model index {index}: {model_name}")
        
        # Reset preview and stats
        self.preview_label.setText("Slice the model to see layer preview")
        self.stats_label.setText("Layer statistics will appear here after slicing")
        self.layer_selector.setEnabled(False)
        
        # Enable slice button if it exists
        if hasattr(self, 'slice_button'):
            self.slice_button.setEnabled(True)
        
    def slice_model(self):
        """Slice the selected model with current settings"""
        # Debug information
        print(f"Slicing model: selected_model_index={self.selected_model_index}")
        
        # Clear any cached bounds from previous slicing
        if hasattr(self, '_global_bounds_cache'):
            delattr(self, '_global_bounds_cache')
            print("Cleared global bounds cache for new slicing")
        
        if self.selected_model_index < 0 or not self.stl_handler:
            print(f"No model selected: index={self.selected_model_index}, stl_handler exists: {self.stl_handler is not None}")
            QMessageBox.warning(self, "No Model Selected", 
                              "Please select an STL model to slice.")
            return
            
        # Check if the stl_handler has meshes
        if not hasattr(self.stl_handler, 'meshes') or not self.stl_handler.meshes:
            print("STL handler has no meshes")
            QMessageBox.warning(self, "No Models Available", 
                              "No STL models are loaded or available for slicing.")
            return
            
        # Check if the selected index is valid
        if self.selected_model_index >= len(self.stl_handler.meshes):
            print(f"Invalid model index: {self.selected_model_index}, mesh count: {len(self.stl_handler.meshes)}")
            QMessageBox.warning(self, "Invalid Model Selection", 
                              "The selected model index is not valid. Please select a valid model.")
            return
            
        # Get the selected mesh
        mesh = self.stl_handler.meshes[self.selected_model_index]
        print(f"Selected mesh: {mesh['name']}")
        
        # Get settings from UI
        self.layer_height = self.layer_height_input.value()
        self.infill_percentage = self.infill_input.value()
        self.wall_count = self.wall_input.value()
        self.nozzle_diameter = self.nozzle_input.value()
        self.print_speed = self.speed_input.value()
        
        # Convert the mesh data to a trimesh object
        try:
            # Create a trimesh from vertices and faces
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh['vertices'],
                faces=mesh['faces']
            )
            
            # Get original bounds for debugging
            original_bounds = trimesh_mesh.bounds
            print(f"Original mesh bounds: {original_bounds}")
            
            # Apply the transform from the stl_handler's mesh_transforms to position the mesh properly
            if hasattr(self.stl_handler, 'mesh_transforms') and len(self.stl_handler.mesh_transforms) > self.selected_model_index:
                transform = self.stl_handler.mesh_transforms[self.selected_model_index]
                position = transform['position']
                orientation = transform['orientation']
                scale = transform['scale']
                
                print(f"Applying mesh transform: position={position}, orientation={orientation}, scale={scale}")
                
                # Apply scale
                scale_matrix = np.array([
                    [scale[0], 0, 0, 0],
                    [0, scale[1], 0, 0],
                    [0, 0, scale[2], 0],
                    [0, 0, 0, 1]
                ])
                
                # Convert quaternion to rotation matrix
                qx, qy, qz, qw = orientation
                xx = qx * qx
                xy = qx * qy
                xz = qx * qz
                xw = qx * qw
                yy = qy * qy
                yz = qy * qz
                yw = qy * qw
                zz = qz * qz
                zw = qz * qw
                
                rotation_matrix = np.array([
                    [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw), 0],
                    [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw), 0],
                    [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy), 0],
                    [0, 0, 0, 1]
                ])
                
                # Create translation matrix
                translation_matrix = np.array([
                    [1, 0, 0, position[0]],
                    [0, 1, 0, position[1]],
                    [0, 0, 1, position[2]],
                    [0, 0, 0, 1]
                ])
                
                # Apply transforms in sequence: scale, rotate, translate
                transform_matrix = translation_matrix @ rotation_matrix @ scale_matrix
                
                # Debug output for transformation matrices
                print(f"Scale matrix:\n{scale_matrix}")
                print(f"Rotation matrix:\n{rotation_matrix}")
                print(f"Translation matrix:\n{translation_matrix}")
                print(f"Final transform matrix:\n{transform_matrix}")
                
                # Apply transform to mesh
                trimesh_mesh.apply_transform(transform_matrix)
                
                # Get new bounds for debugging
                transformed_bounds = trimesh_mesh.bounds
                print(f"Mesh bounds after transform: {transformed_bounds}")
                print(f"Center position after transform: {trimesh_mesh.center_mass}")
            else:
                print("No transform found for the selected mesh. Using original mesh position.")
            
            # Create slicer
            self.slicer = STLSlicer(
                mesh=trimesh_mesh,
                layer_height=self.layer_height,
                infill_percentage=self.infill_percentage,
                wall_count=self.wall_count,
                nozzle_diameter=self.nozzle_diameter
            )
            
            # Connect signals
            self.slicer.slicing_progress.connect(self.slicing_progress.emit)
            self.slicer.slicing_status.connect(self.slicing_status.emit)
            
            # Disable UI during slicing
            # Ensure slice_button exists before accessing it
            if hasattr(self, 'slice_button'):
                self.slice_button.setEnabled(False)
            self.status_label.setText("Slicing in progress...")
            
            # Start slicing in a separate thread
            import threading
            
            self.slicing_thread = threading.Thread(target=self.process_slice)
            self.slicing_thread.daemon = True
            self.slicing_thread.start()
            
        except Exception as e:
            print(f"Error in slice_model: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(self, "Slicing Error", 
                               f"Error slicing the model: {str(e)}")
            # Ensure slice_button exists before accessing it
            if hasattr(self, 'slice_button'):
                self.slice_button.setEnabled(True)
            
    def process_slice(self):
        """Process the slicing operation (runs in a separate thread)"""
        try:
            # Slice the mesh with the current layer height
            layers = self.slicer.slice_mesh(layer_height=self.layer_height)
            # Store the resulting toolpaths
            self.toolpaths = self.slicer.get_combined_layer_paths()
            
            # Use a signal to communicate back to the main thread rather than QMetaObject.invokeMethod
            # Define a new signal if it doesn't exist already
            from PyQt5.QtCore import QObject, pyqtSignal
            
            class SlicingCompleteBridge(QObject):
                success_signal = pyqtSignal(bool, str)
            
            # Only create it once
            if not hasattr(self, 'slicing_bridge'):
                self.slicing_bridge = SlicingCompleteBridge()
                self.slicing_bridge.success_signal.connect(self._on_slicing_complete_from_signal)
            
            # Emit signal with success status
            self.slicing_bridge.success_signal.emit(True, "")
            
        except Exception as e:
            print(f"Slicing error: {str(e)}")
            traceback.print_exc()
            
            # Emit signal with error status
            if hasattr(self, 'slicing_bridge'):
                self.slicing_bridge.success_signal.emit(False, str(e))
            else:
                # Fallback if bridge wasn't created
                print("Slicing failed, but couldn't communicate back to main thread")
                # Try a direct call as a last resort (not thread-safe but better than nothing)
                if hasattr(self, 'slice_button'):
                    self.slice_button.setEnabled(True)
    
    def _on_slicing_complete_from_signal(self, success, error_message=""):
        """Handle completion of slicing operation (called via signal)"""
        print(f"Slicing complete signal received: success={success}")
        # Delegate to the actual handler
        self.on_slicing_complete(success, error_message)
    
    def on_slicing_complete(self, success, error_message=None):
        """Handle completion of slicing operation"""
        if hasattr(self, 'slice_button'):
            self.slice_button.setEnabled(True)
        if not success:
            QMessageBox.critical(self, "Slicing Error", f"Error slicing the model: {error_message}")
            return
        self.status_label.setText("Slicing complete! Creating robot targets...")
        if self.toolpaths and len(self.toolpaths) > 0:
            self.layer_selector.setRange(0, len(self.toolpaths) - 1)
            self.layer_selector.setValue(0)
            self.layer_selector.setEnabled(True)
            if hasattr(self, '_global_bounds_cache'):
                delattr(self, '_global_bounds_cache')
            try:
                total_layers = len(self.toolpaths)
                total_path_length = self.slicer.get_total_path_length()
                printing_time = self.slicer.get_printing_time_estimate(self.print_speed)
                hours = int(printing_time / 3600)
                minutes = int((printing_time % 3600) / 60)
                seconds = int(printing_time % 60)
                stats_text = (
                    f"Total Layers: {total_layers}\n"
                    f"Total Path Length: {total_path_length:.2f} mm\n"
                    f"Estimated Print Time: {hours}h {minutes}m {seconds}s\n"
                    f"Layer Height: {self.layer_height} mm\n"
                    f"Infill: {self.infill_percentage}%\n"
                    f"Walls: {self.wall_count}"
                )
                self.stats_label.setText(stats_text)
            except Exception as e:
                self.stats_label.setText(f"Error calculating statistics: {str(e)}")
            self.update_layer_preview(0)
            if self.parent and hasattr(self.parent, 'robot_viewer'):
                self.parent.robot_viewer.set_print_layers(self.toolpaths, self.selected_model_index)
                self.parent.robot_viewer.show_print_layers = True
                self.parent.robot_viewer.show_stl_models = False
                self.parent.robot_viewer.targets_visible = False
                self.parent.robot_viewer.current_print_layer = -1
                self.parent.robot_viewer.update()
            self.status_label.setText("Creating robot targets...")
            self._auto_create_robot_targets()
            self.print_button.setEnabled(True)
            if hasattr(self, 'octoprint_button'):
                self.octoprint_button.setEnabled(True)
            self.slicing_complete.emit(self.toolpaths)
            self.tabs.setCurrentWidget(self.preview_tab)
        else:
            QMessageBox.warning(self, "Slicing Result", "No layers were generated. The model may be invalid or too small.")
    
    def update_layer_preview(self, layer_index):
        """
        Update the layer preview with the specified layer index.
        
        Args:
            layer_index: Index of the layer to display
        """
        # First check if we have toolpaths
        if not hasattr(self, 'toolpaths') or not self.toolpaths:
            print("No toolpaths available for preview")
            return
            
        # Validate layer index
        if layer_index < 0 or layer_index >= len(self.toolpaths):
            print(f"Invalid layer index: {layer_index}, max: {len(self.toolpaths)-1}")
            return
            
        layer_data = self.toolpaths[layer_index]
        z_height = layer_data['z_height']
        paths = layer_data['paths']
        
        # Debug output
        print(f"Updating preview for layer {layer_index}: Z={z_height:.3f}mm, {len(paths)} paths")
        
        # Additional debug: show first few points of first path to see if they're different
        if paths and len(paths) > 0 and len(paths[0]) > 0:
            first_path_sample = paths[0][:min(3, len(paths[0]))]  # First 3 points
            print(f"Layer {layer_index} first path sample: {first_path_sample}")
        
        # Show layer data structure
        print(f"Layer {layer_index} data keys: {list(layer_data.keys())}")
        if 'paths' in layer_data:
            print(f"Layer {layer_index} has {len(layer_data['paths'])} paths")
            for i, path in enumerate(layer_data['paths'][:3]):  # Show first 3 paths
                print(f"  Path {i}: {len(path)} points, first point: {path[0] if path else 'empty'}")
        
        # Update layer number label
        self.layer_number_label.setText(f"Layer {layer_index + 1}/{len(self.toolpaths)} (Z={z_height:.2f}mm)")
        
        # Update the 3D viewer to highlight this layer
        if self.parent and hasattr(self.parent, 'robot_viewer'):
            self.parent.robot_viewer.set_current_print_layer(layer_index)
            # Make sure layer visualization is enabled
            self.parent.robot_viewer.show_print_layers = True
        
        # Create a pixmap for the preview
        canvas_width = 600
        canvas_height = 400
        pixmap = QPixmap(canvas_width, canvas_height)
        pixmap.fill(Qt.white)
        
        # Draw on the pixmap if we have paths
        if paths:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw a subtle background grid to help visualize positioning
            painter.setPen(QPen(QColor(240, 240, 240), 1))  # Very light gray
            grid_spacing = 50  # pixels
            for x in range(0, canvas_width, grid_spacing):
                painter.drawLine(x, 0, x, canvas_height)
            for y in range(0, canvas_height, grid_spacing):
                painter.drawLine(0, y, canvas_width, y)
            
            # Draw center lines
            painter.setPen(QPen(QColor(220, 220, 220), 1))  # Light gray
            painter.drawLine(canvas_width // 2, 0, canvas_width // 2, canvas_height)  # Vertical center
            painter.drawLine(0, canvas_height // 2, canvas_width, canvas_height // 2)  # Horizontal center
            
            # Calculate global bounds across ALL layers for consistent scaling
            if not hasattr(self, '_global_bounds_cache'):
                print("Calculating global bounds for consistent layer preview scaling...")
                global_minx = float('inf')
                global_miny = float('inf')
                global_maxx = float('-inf')
                global_maxy = float('-inf')
                
                for layer in self.toolpaths:
                    layer_paths = layer.get('paths', [])
                    for path in layer_paths:
                        for point in path:
                            x, y, _ = point
                            global_minx = min(global_minx, x)
                            global_miny = min(global_miny, y)
                            global_maxx = max(global_maxx, x)
                            global_maxy = max(global_maxy, y)
                
                self._global_bounds_cache = {
                    'minx': global_minx,
                    'miny': global_miny,
                    'maxx': global_maxx,
                    'maxy': global_maxy
                }
                print(f"Global bounds: minx={global_minx:.3f}, miny={global_miny:.3f}, maxx={global_maxx:.3f}, maxy={global_maxy:.3f}")
            
            # Use global bounds for consistent scaling
            global_bounds = self._global_bounds_cache
            minx = global_bounds['minx']
            miny = global_bounds['miny']
            maxx = global_bounds['maxx']
            maxy = global_bounds['maxy']
            
            # Find bounds of current layer for debugging
            layer_minx = float('inf')
            layer_miny = float('inf')
            layer_maxx = float('-inf')
            layer_maxy = float('-inf')
            
            total_points = 0
            for path in paths:
                for point in path:
                    x, y, _ = point
                    layer_minx = min(layer_minx, x)
                    layer_miny = min(layer_miny, y)
                    layer_maxx = max(layer_maxx, x)
                    layer_maxy = max(layer_maxy, y)
                    total_points += 1
            
            # Debug bounds for this layer vs global
            print(f"Layer {layer_index} bounds: minx={layer_minx:.3f}, miny={layer_miny:.3f}, maxx={layer_maxx:.3f}, maxy={layer_maxy:.3f}")
            print(f"Global bounds: minx={minx:.3f}, miny={miny:.3f}, maxx={maxx:.3f}, maxy={maxy:.3f}")
            print(f"Layer {layer_index} total points: {total_points}")
            
            # Calculate scaling to fit the preview area with margins using global bounds
            margin = 20  # pixels
            width = maxx - minx
            height = maxy - miny
            
            print(f"Global dimensions: width={width:.3f}, height={height:.3f}")
            
            if width <= 0 or height <= 0:
                # Invalid dimensions
                painter.end()
                self.preview_label.setText("Invalid global dimensions")
                print(f"Invalid global dimensions")
                return
                
            # Calculate scaling to fit while maintaining aspect ratio
            scale_x = (canvas_width - 2 * margin) / width
            scale_y = (canvas_height - 2 * margin) / height
            scale = min(scale_x, scale_y)
            
            print(f"Global scale: {scale:.3f}")
            
            # Calculate centering offset using global bounds
            offset_x = margin + (canvas_width - 2 * margin - width * scale) / 2
            offset_y = margin + (canvas_height - 2 * margin - height * scale) / 2
            
            # Create transform function to map model coordinates to screen coordinates
            def transform(point):
                x, y, _ = point
                screen_x = offset_x + (x - minx) * scale
                screen_y = canvas_height - (offset_y + (y - miny) * scale)  # Flip Y
                return screen_x, screen_y
            
            # Draw the paths with different colors for different path types
            for i, path in enumerate(paths):
                if len(path) < 2:
                    continue
                    
                # Use different colors based on path characteristics
                # Try to differentiate between perimeter, walls, and infill
                path_length = len(path)
                
                # Estimate path type based on length and position
                if i == 0:
                    # First path is likely perimeter - blue
                    painter.setPen(QPen(QColor(0, 0, 255), 2))
                elif path_length > 20:
                    # Long paths are likely perimeters or outer walls - blue/green
                    painter.setPen(QPen(QColor(0, 150, 0), 2))
                elif path_length < 5:
                    # Short paths are likely infill segments - red
                    painter.setPen(QPen(QColor(255, 0, 0), 1))
                else:
                    # Medium paths are likely inner walls - dark green
                    painter.setPen(QPen(QColor(0, 100, 0), 1))
                
                # Draw the path
                for j in range(len(path) - 1):
                    x1, y1 = transform(path[j])
                    x2, y2 = transform(path[j + 1])
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            
            painter.end()
            
            # Set the pixmap to the preview label
            self.preview_label.setPixmap(pixmap)
        else:
            self.preview_label.setText("No paths in this layer")
            print(f"No paths found in layer {layer_index}")
    
    def _auto_create_robot_targets(self):
        """Automatically create robot targets from the sliced toolpaths, without confirmation dialog."""
        if not self.toolpaths:
            print("No toolpaths available for auto target creation")
            return
        print(f"[AUTO] Creating robot targets for {len(self.toolpaths)} layers")
        try:
            z_offset = 0.0
            if hasattr(self.parent, 'ribbon'):
                if hasattr(self.parent.ribbon, 'icons_controls') and 'z_offset_combo' in self.parent.ribbon.icons_controls:
                    try:
                        z_offset_mm = self.parent.ribbon.icons_controls['z_offset_combo'].value()
                        z_offset = z_offset_mm / 1000.0
                        print(f"[AUTO] Using Z offset from ribbon: {z_offset_mm}mm ({z_offset}m)")
                    except Exception as e:
                        print(f"[AUTO] Error getting Z offset value: {e}")
            if hasattr(self.parent, '_generate_print_targets') and callable(self.parent._generate_print_targets):
                print("[AUTO] Calling parent's _generate_print_targets...")
                created_targets_list = self.parent._generate_print_targets(self.toolpaths, slicer_dialog=self)
                if created_targets_list is not None:
                    self.progress_bar.setValue(100)
                    self.status_label.setText(f"Created {len(created_targets_list)} robot targets.")
                    if hasattr(self, 'targets_created'):
                        print("[AUTO] Emitting targets_created signal from dialog")
                        self.targets_created.emit(created_targets_list)
                    else:
                        print("[AUTO] Could not emit targets_created signal from dialog")
                else:
                    self.status_label.setText("Target generation failed.")
                # --- Ensure STL model stays hidden and layer lines stay visible ---
                if self.parent and hasattr(self.parent, 'robot_viewer'):
                    self.parent.robot_viewer.show_stl_models = False
                    self.parent.robot_viewer.show_print_layers = True
                    self.parent.robot_viewer.update()
                # ---
            else:
                print("[AUTO] Error: Parent does not have _generate_print_targets method.")
                QMessageBox.critical(self, "Internal Error", "Cannot generate targets: Parent method missing.")
        except Exception as e:
            print(f"[AUTO] Error in auto_create_robot_targets: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error Creating Targets", f"Error creating robot targets: {str(e)}")
    
    def start_printing(self):
        """Start the 3D printing process using the robot"""
        print("Starting start_printing method...")
        
        if not self.toolpaths:
            print("No toolpaths available")
            QMessageBox.warning(self, "No Toolpaths", 
                              "Please slice a model first to generate toolpaths.")
            return
            
        # Check if robot programmer is available
        if not self.robot_programmer:
            print("Robot programmer not available")
            QMessageBox.warning(self, "Robot Programmer Not Available", 
                              "Cannot start printing: Robot programmer not available.")
            return
            
        # Get the targets from the program
        targets = self.robot_programmer.program.get("targets", [])
        if not targets:
            print("No targets in program")
            QMessageBox.warning(self, "No Targets", 
                              "No targets available for printing.")
            return

        print(f"3D print path detected with {len(targets)} targets")
        
        try:
            # Hide the dialog BEFORE emitting signal
            self.hide()
            
            # Emit signal to start printing
            print("Emitting start_printing_signal")
            self.start_printing_signal.emit(targets)
            
        except Exception as e:
            error_msg = f"Error in start_printing: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Error Starting Print", error_msg)
            self.status_label.setText("Error starting print.")
    
    def get_print_time_estimate(self):
        """
        Get a human-readable estimate of the print time.
        
        Returns:
            str: Formatted print time estimate
        """
        if not self.slicer:
            return "Unknown"
            
        # Get print speed from UI
        print_speed = self.speed_input.value()  # mm/s
        
        # Get time estimate from slicer
        printing_time = self.slicer.get_printing_time_estimate(print_speed)
        
        # Format time
        hours = int(printing_time / 3600)
        minutes = int((printing_time % 3600) / 60)
        seconds = int(printing_time % 60)
        
        return f"{hours}h {minutes}m {seconds}s"
    
    def update_progress(self, progress):
        """Update the progress bar"""
        self.progress_bar.setValue(progress)
        
    def update_status(self, status):
        """Update the status label"""
        self.status_label.setText(status)
        
    def show_octoprint_setup(self):
        """Show the OctoPrint setup dialog"""
        if not self.octoprint_handler:
            return
            
        try:
            from octoprint_handler import OctoPrintSetupDialog
            setup_dialog = OctoPrintSetupDialog(self.octoprint_handler, self)
            setup_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Setup Error", 
                               f"Error setting up OctoPrint: {str(e)}")

    def reject(self):
        """Handle dialog rejection (close button or cancel)"""
        # Ensure program is stopped and flags are reset
        if self.robot_programmer and hasattr(self.robot_programmer, 'program'):
            self.robot_programmer.program['is_running'] = False
            if hasattr(self.robot_programmer, 'stop_program'):
                self.robot_programmer.stop_program()
        
        # Call parent reject
        super().reject()

    def closeEvent(self, event):
        """Handle dialog close event"""
        # Ensure program is stopped and flags are reset
        if self.robot_programmer and hasattr(self.robot_programmer, 'program'):
            self.robot_programmer.program['is_running'] = False
            if hasattr(self.robot_programmer, 'stop_program'):
                self.robot_programmer.stop_program()
        
        # Call parent close event
        super().closeEvent(event)


def start_3d_printing(stl_handler, robot_programmer, parent=None):
    """
    Start the 3D printing process.
    
    Args:
        stl_handler: The STL handler to use for model access
        robot_programmer: The robot programmer for creating targets
        parent: Parent widget for the dialog
        
    Returns:
        bool: True if the dialog was accepted, False otherwise
    """
    # Validate STL handler has models before creating the dialog
    if not stl_handler or not hasattr(stl_handler, 'meshes') or not stl_handler.meshes:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(parent, "No STL Models", 
                          "No STL models are loaded. Please import at least one STL model first.")
        return False
    
    print(f"Creating 3D printing dialog with {len(stl_handler.meshes)} models")
    
    # Create dialog with initial selected model index set to the first model
    dialog = SlicerDialog(stl_handler, robot_programmer, parent)
    
    # Make sure a model is selected if there are models available
    if dialog.model_combo.count() > 0 and dialog.selected_model_index < 0:
        print("Setting initial selected model index to 0")
        dialog.model_combo.setCurrentIndex(0)
        dialog.selected_model_index = 0
    
    # Show the dialog
    result = dialog.exec_()
    return result == QDialog.Accepted 