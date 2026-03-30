import numpy as np

# Apply collections compatibility fixes for Python 3.11+
import collections_compat

import trimesh
from PyQt5.QtCore import QObject, pyqtSignal
import math
import time
import os

class STLSlicer(QObject):
    """
    Slices STL models into layers for 3D printing with a robot arm.
    
    This class handles the process of converting a 3D STL model into a series
    of 2D contours at different Z heights, which can then be converted into
    robot paths for 3D printing.
    """
    
    # Signals for progress updates
    slicing_progress = pyqtSignal(int)     # Progress percentage
    slicing_status = pyqtSignal(str)       # Status message
    
    def __init__(self, mesh=None, layer_height=0.2, infill_percentage=20, 
                 wall_count=2, nozzle_diameter=0.4):
        """
        Initialize the STL slicer with parameters.
        
        Args:
            mesh: The trimesh mesh object to slice
            layer_height: Height of each layer in mm
            infill_percentage: Percentage of infill (0-100)
            wall_count: Number of perimeter walls
            nozzle_diameter: Diameter of the printer nozzle in mm
        """
        super().__init__()
        self.mesh = mesh
        self.layer_height = layer_height
        self.infill_percentage = infill_percentage
        self.wall_count = wall_count
        self.nozzle_diameter = nozzle_diameter
        
        # Calculate derived parameters
        self.infill_spacing = self.calculate_infill_spacing()
        
        # Store generated paths
        self.layers = []  # List of layer data
        
    def calculate_infill_spacing(self):
        """Calculate spacing between infill lines based on infill percentage"""
        # Convert percentage to line spacing
        # 100% infill would be lines spaced at nozzle_diameter
        # 0% infill would be no infill lines
        
        if self.infill_percentage <= 0:
            return 0  # No infill
            
        # Scale from 1x to 5x nozzle diameter based on infill percentage
        # Lower percentage = wider spacing
        max_spacing = 5.0 * self.nozzle_diameter
        min_spacing = 1.0 * self.nozzle_diameter
        
        # Linear interpolation between min and max spacing
        spacing = max_spacing - (self.infill_percentage / 100.0) * (max_spacing - min_spacing)
        return spacing
    
    def slice_mesh(self, mesh=None, layer_height=None):
        """
        Slice the mesh into layers and generate paths for each layer.
        
        Args:
            mesh: Optional mesh to slice (uses self.mesh if None)
            layer_height: Optional layer height (uses self.layer_height if None)
            
        Returns:
            List of layers, each containing paths for perimeters and infill
        """
        if mesh is not None:
            self.mesh = mesh
        
        if layer_height is not None:
            self.layer_height = layer_height
            
        if self.mesh is None:
            self.slicing_status.emit("No mesh provided for slicing")
            return []
            
        # Get mesh bounds to determine slicing range
        bounds = self.mesh.bounds
        min_z = bounds[0][2]
        max_z = bounds[1][2]
        
        # Convert mesh dimensions from meters to millimeters
        min_z_mm = min_z * 1000
        max_z_mm = max_z * 1000
        height_mm = max_z_mm - min_z_mm
        
        # Calculate number of layers using millimeter values
        # self.layer_height is already in millimeters
        num_layers = math.ceil(height_mm / self.layer_height)
        
        print(f"Mesh bounds (meters): min_z={min_z}m, max_z={max_z}m")
        print(f"Mesh height (mm): {height_mm}mm")
        print(f"Layer Height: {self.layer_height}mm")
        print(f"Calculated Number of Layers: {num_layers}")
        
        self.slicing_status.emit(f"Slicing mesh into {num_layers} layers")
        self.layers = []
        
        # Process each layer
        start_time = time.time()
        for i in range(num_layers):
            # Calculate current Z height (in meters, since the mesh is in meters)
            z_height = min_z + (i * self.layer_height / 1000.0)  # Convert layer height back to meters
            print(f"Slicing at Z height: {z_height}m ({z_height * 1000}mm)")
            
            # Update progress
            progress = int(i / num_layers * 100)
            self.slicing_progress.emit(progress)
            self.slicing_status.emit(f"Slicing layer {i+1}/{num_layers} at height {z_height * 1000:.2f}mm")
            
            # Slice mesh at current height
            layer_data = self.slice_at_height(z_height)
            if layer_data:
                print(f"Layer {i+1} data: {layer_data}")
                self.layers.append(layer_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final progress update
        self.slicing_progress.emit(100)
        self.slicing_status.emit(f"Completed slicing {len(self.layers)} layers in {duration:.2f} seconds")
        
        return self.layers
        
    def slice_at_height(self, z):
        """
        Slice the mesh at a specific Z height.
        
        Args:
            z: The Z height to slice at
            
        Returns:
            Dictionary containing the layer data including paths
        """
        # Create a slicing plane at height z
        slicing_plane = trimesh.intersections.plane_lines
        plane_normal = [0, 0, 1]  # Z-axis
        plane_origin = [0, 0, z]
        
        # Get slice as a cross-section
        try:
            sections = self.mesh.section(plane_origin=plane_origin, 
                                         plane_normal=plane_normal)
        except Exception as e:
            print(f"Error slicing at height {z}: {str(e)}")
            return None
            
        if sections is None or len(sections.entities) == 0:
            print(f"No intersection found at height {z}")
            return None
            
        # Get 2D polygons from cross-section, preserving scale
        try:
            # Get the original 3D coordinates for this specific layer
            original_vertices = sections.vertices.copy()
            
            # Convert to planar coordinates while preserving scale
            polygons, transform = sections.to_planar()
            
            # Calculate mesh center and scale for this specific layer
            layer_center = np.mean(original_vertices, axis=0)
            min_bounds = np.min(original_vertices, axis=0)
            max_bounds = np.max(original_vertices, axis=0)
            layer_size = max_bounds - min_bounds
            
            # Store layer-specific scale
            layer_scale = np.array([layer_size[0], layer_size[1]])
            
            # Debug prints
            print(f"Layer {z} scale: {layer_scale}")
            print(f"Layer {z} center: {layer_center}")
            
            # Use layer-specific center coordinates
            layer_center_x = layer_center[0]
            layer_center_y = layer_center[1]
            
        except Exception as e:
            print(f"Error processing layer at height {z}: {str(e)}")
            return None
            
        if polygons is None or len(polygons.entities) == 0:
            print(f"No polygons found at height {z}")
            return None
        
        # Build layer data
        layer_data = {
            'z_height': z,
            'perimeters': [],
            'inner_walls': [],
            'infill': [],
            'travel': []
        }
        
        # Process each polygon from the slice
        for polygon in polygons.polygons_full:
            # First get all perimeter points in 2D planar coordinates
            planar_perimeter_points = [point for point in polygon.exterior.coords]
            
            # Find the 2D center of this specific polygon in planar space
            planar_center_x = sum(p[0] for p in planar_perimeter_points) / len(planar_perimeter_points)
            planar_center_y = sum(p[1] for p in planar_perimeter_points) / len(planar_perimeter_points)
            
            # Convert to a format our system can use (list of points)
            perimeter_points = []
            for point in polygon.exterior.coords:
                # Calculate offset from polygon center in planar space
                dx = point[0] - planar_center_x
                dy = point[1] - planar_center_y
                
                # Apply offsets using the layer-specific center
                x_coord = layer_center_x + dx
                y_coord = layer_center_y + dy
                
                # Add point with correct position relative to this layer
                perimeter_points.append((x_coord, y_coord, z))
                
            # Add perimeter to layer
            if perimeter_points:
                layer_data['perimeters'].append(perimeter_points)
                
                # Generate inner walls (multiple perimeters)
                inner_walls = self.generate_inner_walls(polygon, z, planar_center_x, planar_center_y, layer_center_x, layer_center_y)
                if inner_walls:
                    layer_data['inner_walls'].extend(inner_walls)
                    
                # Generate infill
                if self.infill_percentage > 0:
                    infill_paths = self.generate_infill(polygon, z, planar_center_x, planar_center_y, layer_center_x, layer_center_y)
                    if infill_paths:
                        layer_data['infill'].extend(infill_paths)
        
        # Generate travel paths to connect segments
        all_print_paths = (layer_data['perimeters'] + 
                          layer_data['inner_walls'] + 
                          layer_data['infill'])
                          
        if all_print_paths:
            travel_paths = self.generate_travel_paths(
                layer_data['perimeters'],
                layer_data['inner_walls'],
                layer_data['infill'],
                z
            )
            layer_data['travel'] = travel_paths
            
        return layer_data
    
    def generate_inner_walls(self, polygon, z, planar_center_x, planar_center_y, layer_center_x, layer_center_y):
        """
        Generate inner wall paths for a polygon.
        
        Args:
            polygon: The polygon to generate inner walls for
            z: The Z height of the layer
            planar_center_x: X center of polygon in planar space
            planar_center_y: Y center of polygon in planar space
            layer_center_x: X center of the current layer
            layer_center_y: Y center of the current layer
            
        Returns:
            List of inner wall paths (each path is a list of points)
        """
        inner_walls = []
        current_polygon = polygon
        
        # Generate inner walls based on wall_count
        for i in range(self.wall_count - 1):
            # Inset the polygon by nozzle diameter
            try:
                inset = current_polygon.buffer(-self.nozzle_diameter)
            except Exception as e:
                # Print debugging info and skip if there's an error
                print(f"Error generating inner wall {i+1}: {str(e)}")
                break
                
            if inset is None or inset.is_empty:
                # No more inner walls possible
                break
                
            # Handle potential MultiPolygon result
            if hasattr(inset, 'geoms'):
                # Multiple polygons from the inset operation
                for inner_poly in inset.geoms:
                    inner_path = []
                    for point in inner_poly.exterior.coords:
                        # Calculate offset from polygon center in planar space
                        dx = point[0] - planar_center_x
                        dy = point[1] - planar_center_y
                        
                        # Apply offsets using the layer-specific center
                        x_coord = layer_center_x + dx
                        y_coord = layer_center_y + dy
                        
                        inner_path.append((x_coord, y_coord, z))
                    if len(inner_path) > 2:  # Ensure valid path
                        inner_walls.append(inner_path)
            else:
                # Single polygon from inset
                inner_path = []
                for point in inset.exterior.coords:
                    # Calculate offset from polygon center in planar space
                    dx = point[0] - planar_center_x
                    dy = point[1] - planar_center_y
                    
                    # Apply offsets using the layer-specific center
                    x_coord = layer_center_x + dx
                    y_coord = layer_center_y + dy
                    
                    inner_path.append((x_coord, y_coord, z))
                if len(inner_path) > 2:  # Ensure valid path
                    inner_walls.append(inner_path)
                
            # Update current polygon for next inset
            current_polygon = inset
        
        return inner_walls
    
    def generate_infill(self, polygon, z, planar_center_x, planar_center_y, layer_center_x, layer_center_y):
        """
        Generate infill paths for a polygon.
        
        Args:
            polygon: The polygon to generate infill for
            z: The Z height of the layer
            planar_center_x: X center of polygon in planar space
            planar_center_y: Y center of polygon in planar space
            layer_center_x: X center of the current layer
            layer_center_y: Y center of the current layer
            
        Returns:
            List of infill paths (each path is a list of points)
        """
        if self.infill_percentage <= 0 or self.infill_spacing <= 0:
            return []
            
        infill_paths = []
        
        # Get bounds of the polygon
        minx, miny, maxx, maxy = polygon.bounds
        
        # Alternate infill direction based on layer height to create stronger parts
        layer_index = int(z / self.layer_height)
        is_odd_layer = layer_index % 2 == 1
        
        if is_odd_layer:
            # Generate horizontal lines
            y = miny
            while y <= maxy:
                # Create a horizontal line at this Y coordinate
                line_start = (minx - 1, y)  # Extend slightly beyond bounds
                line_end = (maxx + 1, y)    # Extend slightly beyond bounds
                
                # Intersect line with polygon to get segments
                try:
                    from shapely.geometry import LineString
                    line = LineString([line_start, line_end])
                    intersection = polygon.intersection(line)
                    
                    # Process the intersection result
                    if hasattr(intersection, 'geoms'):
                        # Multiple line segments from intersection
                        for segment in intersection.geoms:
                            segment_path = []
                            for coord in segment.coords:
                                # Calculate offset from polygon center in planar space
                                dx = coord[0] - planar_center_x
                                dy = coord[1] - planar_center_y
                                
                                # Apply offsets using the layer-specific center
                                x_coord = layer_center_x + dx
                                y_coord = layer_center_y + dy
                                
                                segment_path.append((x_coord, y_coord, z))
                            if len(segment_path) >= 2:
                                infill_paths.append(segment_path)
                    elif not intersection.is_empty:
                        # Single line segment
                        segment_path = []
                        for coord in intersection.coords:
                            # Calculate offset from polygon center in planar space
                            dx = coord[0] - planar_center_x
                            dy = coord[1] - planar_center_y
                            
                            # Apply offsets using the layer-specific center
                            x_coord = layer_center_x + dx
                            y_coord = layer_center_y + dy
                            
                            segment_path.append((x_coord, y_coord, z))
                        if len(segment_path) >= 2:
                            infill_paths.append(segment_path)
                except Exception as e:
                    # Skip on error and continue
                    print(f"Error generating horizontal infill: {str(e)}")
                    
                # Move to next Y position
                y += self.infill_spacing
        else:
            # Generate vertical lines
            x = minx
            while x <= maxx:
                # Create a vertical line at this X coordinate
                line_start = (x, miny - 1)  # Extend slightly beyond bounds
                line_end = (x, maxy + 1)    # Extend slightly beyond bounds
                
                # Intersect line with polygon to get segments
                try:
                    from shapely.geometry import LineString
                    line = LineString([line_start, line_end])
                    intersection = polygon.intersection(line)
                    
                    # Process the intersection result
                    if hasattr(intersection, 'geoms'):
                        # Multiple line segments from intersection
                        for segment in intersection.geoms:
                            segment_path = []
                            for coord in segment.coords:
                                # Calculate offset from polygon center in planar space
                                dx = coord[0] - planar_center_x
                                dy = coord[1] - planar_center_y
                                
                                # Apply offsets using the layer-specific center
                                x_coord = layer_center_x + dx
                                y_coord = layer_center_y + dy
                                
                                segment_path.append((x_coord, y_coord, z))
                            if len(segment_path) >= 2:
                                infill_paths.append(segment_path)
                    elif not intersection.is_empty:
                        # Single line segment
                        segment_path = []
                        for coord in intersection.coords:
                            # Calculate offset from polygon center in planar space
                            dx = coord[0] - planar_center_x
                            dy = coord[1] - planar_center_y
                            
                            # Apply offsets using the layer-specific center
                            x_coord = layer_center_x + dx
                            y_coord = layer_center_y + dy
                            
                            segment_path.append((x_coord, y_coord, z))
                        if len(segment_path) >= 2:
                            infill_paths.append(segment_path)
                except Exception as e:
                    # Skip on error and continue
                    print(f"Error generating vertical infill: {str(e)}")
                    
                # Move to next X position
                x += self.infill_spacing
        
        return infill_paths
    
    def generate_travel_paths(self, perimeters, inner_walls, infill_paths, z):
        """
        Generate travel paths between print segments.
        
        Args:
            perimeters: List of perimeter paths
            inner_walls: List of inner wall paths
            infill_paths: List of infill paths
            z: The Z height of the layer
            
        Returns:
            List of travel paths connecting print segments
        """
        travel_paths = []
        
        # Create a combined sequence of all paths to print
        print_sequence = []
        
        # Add perimeters first (most important for quality)
        for perimeter in perimeters:
            print_sequence.append(perimeter)
            
        # Add inner walls next
        for wall in inner_walls:
            print_sequence.append(wall)
            
        # Add infill last
        for infill in infill_paths:
            print_sequence.append(infill)
            
        # Generate travel paths between segments
        if len(print_sequence) > 1:
            for i in range(len(print_sequence) - 1):
                current_path = print_sequence[i]
                next_path = print_sequence[i + 1]
                
                # Get end point of current path and start point of next path
                end_point = current_path[-1]
                start_point = next_path[0]
                
                # Create travel path (just two points)
                travel_path = [end_point, start_point]
                travel_paths.append(travel_path)
        
        return travel_paths

    def get_combined_layer_paths(self):
        """
        Get all paths combined and sorted by layer.
        
        Returns:
            List of layers, each containing a list of paths (perimeters, walls, infill)
        """
        combined_paths = []
        
        # For each layer, combine all printing paths in the right order
        for layer in self.layers:
            layer_paths = []
            
            # Add perimeters first
            for perimeter in layer['perimeters']:
                layer_paths.append(perimeter)
                
            # Add inner walls next
            for wall in layer['inner_walls']:
                layer_paths.append(wall)
                
            # Add infill last
            for infill in layer['infill']:
                layer_paths.append(infill)
                
            combined_paths.append({
                'z_height': layer['z_height'],
                'paths': layer_paths
            })
            
        return combined_paths

    def get_total_path_length(self):
        """
        Calculate the total path length of all printing paths.
        
        Returns:
            Total path length in mm
        """
        total_length = 0
        
        for layer in self.layers:
            # Add up lengths of all paths in this layer
            for path_type in ['perimeters', 'inner_walls', 'infill']:
                for path in layer[path_type]:
                    for i in range(len(path) - 1):
                        p1 = path[i]
                        p2 = path[i + 1]
                        # Calculate distance between consecutive points
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        dz = p2[2] - p1[2]
                        segment_length = math.sqrt(dx*dx + dy*dy + dz*dz)
                        total_length += segment_length
        
        return total_length

    def get_printing_time_estimate(self, print_speed_mm_per_s):
        """
        Estimate the printing time based on path length and print speed.
        
        Args:
            print_speed_mm_per_s: Print speed in mm per second
            
        Returns:
            Estimated printing time in seconds
        """
        if print_speed_mm_per_s <= 0:
            return 0
            
        total_length = self.get_total_path_length()
        time_seconds = total_length / print_speed_mm_per_s
        
        # Add some time for travel moves and acceleration
        travel_factor = 1.2  # 20% extra time for travel and acceleration
        return time_seconds * travel_factor 