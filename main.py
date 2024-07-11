from dataclasses import dataclass
from typing import Tuple, List, Dict, Set

import math
import mathutils
import numpy as np
from mathutils import Vector
import random
import bpy

@dataclass
class Layout:
    location: Tuple[float, float, float]
    min: Tuple[float, float, float]
    max: Tuple[float, float, float]
    orientation: Tuple[float, float, float] # Euler angles (pitch, yaw, roll)
    imported_object: bpy.types.Object # Object

def scale_group(objects: List[bpy.types.Object], scale_factor: float) -> None:
    """
    Scale a group of objects by a given factor.

    Args:
        objects (List[bpy.types.Object]): List of Blender objects to scale.
        scale_factor (float): The scale factor to apply.

    Example:
    scale_group([object1, object2], 1.5)
    """
    for obj in objects:
        obj.scale = (obj.scale.x * scale_factor,
                    obj.scale.y * scale_factor,
                    obj.scale.z * scale_factor)
        obj.matrix_world = obj.matrix_world * scale_factor

def find_highest_vertex_point(objs: List[bpy.types.Object]) -> Dict[str, float]:
    """
    Find the highest vertex point among a list of objects.

    Args:
        objs (List[bpy.types.Object]): List of Blender objects to evaluate.

    Returns:
        Dict[str, float]: The highest x, y, and z coordinates.

    Example:
        highest_point = find_highest_vertex_point([object1, object2])
    """
    bpy.context.view_layer.update()
    highest_points = {'x': -float('inf'), 'y': -float('inf'), 'z': -float('inf')}
                                                                          
    for obj in objs:
        # Apply the object’s current transformation to its vertices
        obj_matrix_world = obj.matrix_world

        if obj.type == 'MESH':
            # Update mesh to the latest data
            obj.data.update()
            for vertex in obj.data.vertices:

                # Transform Local vertex Coordinates to World Coordinates
                world_vertex = obj_matrix_world @ vertex.co

                highest_points['x'] = max(highest_points['x'], world_vertex.x)
                highest_points['y'] = max(highest_points['y'], world_vertex.y)
                highest_points['z'] = max(highest_points['z'], world_vertex.z)
        return highest_points
    
def find_lowest_vertex_point(objs: List[bpy.types.Object]) -> Dict[str, float]:
    """
    Find the lowest vertex point among a list of objects.

    Args:
        objs (List[bpy.types.Object]): List of Blender objects to evaluate.

    Returns:
        Dict[str, float]: The lowest x, y, and z coordinates.

    Example:
        lowest_point = find_lowest_vertex_point([object1, object2])
    """
    bpy.context.view_layer.update()
    lowest_points = {'x': float('inf'), 'y': float('inf'), 'z': float('inf')}
    for obj in objs:
        # Apply the object’s current transformation to its vertices
        obj_matrix_world = obj.matrix_world
        if obj.type == 'MESH':
            # Update mesh to the latest data
            obj.data.update()
            for vertex in obj.data.vertices:
                world_vertex = obj_matrix_world @ vertex.co
                lowest_points['x'] = min(lowest_points['x'], world_vertex.x)
                lowest_points['y'] = min(lowest_points['y'], world_vertex.y)
                lowest_points['z'] = min(lowest_points['z'], world_vertex.z)

                return lowest_points

def rotate_objects_z_axis(objects: List[bpy.types.Object], angle_degrees: float) -> None:
    """
    Rotate a group of objects around the Z-axis by a given angle.

    Args:
        objects (List[bpy.types.Object]): List of objects to rotate.
        angle_degrees (float): The angle in degrees to rotate.

    Example:
        rotate_objects_z_axis([object1, object2], 45)
    """
    bpy.context.view_layer.update()
    angle_radians = math.radians(angle_degrees) # Convert angle to radians
    rotation_matrix = mathutils.Matrix.Rotation(angle_radians, 4, 'Y')
    lowest_point = find_lowest_vertex_point(objects)
    highest_points = find_highest_vertex_point(objects)
    center_point = {'x': (lowest_point['x'] + highest_points['x']) / 2,
                    'y': (lowest_point['y'] + highest_points['y']) / 2,
                    'z': 0}
    for obj in objects:
        if obj.type == 'MESH':
            obj.data.update()
            obj.matrix_world = obj.matrix_world @ rotation_matrix

    lowest_point = find_lowest_vertex_point(objects)
    highest_points = find_highest_vertex_point(objects)
    center_point['x'] -= (lowest_point['x'] + highest_points['x']) / 2
    center_point['y'] -= (lowest_point['y'] + highest_points['y']) / 2
    shift(objects, center_point)

def shift(objects: List[bpy.types.Object], shift_loc: Dict[str, float]) -> None:
    """
    Shift a group of objects with shift_loc.

    Args:
        objects (List[bpy.types.Object]): List of objects to rotate.
        shift_loc (float): The shift vector.

    Example:
        rotate_objects_z_axis([object1, object2], (5,3,1))
    """
    for obj in objects:
    # Shift object so the lowest point is at (0,0,0)
        obj.location.x += shift_loc['x']
        obj.location.y += shift_loc['y']
        obj.location.z += shift_loc['z']
    bpy.context.view_layer.update()

def calculate_shortest_distance(vertices1: Set[Tuple[float, float, float]], vertices2: Set[Tuple[float, float, float]]) -> float:
    """
    Calculate the shortest distance between two sets of vertices.

    Args:
        vertices1 (Set[Tuple[float, float, float]]): First set of vertices.
        vertices2 (Set[Tuple[float, float, float]]): Second set of vertices.

    Returns:
        float: Shortest distance over the Z-axis.
    """
    min_distance = float('inf')
    for v1_tuple in vertices1:
        v1 = Vector(v1_tuple)
        for v2_tuple in vertices2:
            v2 = Vector(v2_tuple)
            distance = (v1 - v2).length
            min_distance = min(min_distance, distance)
    return min_distance

def check_vertex_overlap(vertices1: Set[Vector], vertices2: Set[Vector], threshold: float = 0.01) -> float:
    """
    Check if there is any overlap between two sets of vertices within a threshold.

    Args:
        vertices1 (Set[Vector]): First set of vertices.
        vertices2 (Set[Vector]): Second set of vertices.
        threshold (float): Distance threshold to consider as an overlap.

    Returns:
        bool: True if there is an overlap, False otherwise.
    """
    for v1_tuple in vertices1:
        v1 = Vector(v1_tuple)
        for v2_tuple in vertices2:
            v2 = Vector(v2_tuple)
            if (v1 - v2).length <= threshold:
                return 1.0
    return 0.0

def evaluate_constraints(assets, constraints):
    """Evaluate all constraints and return the overall score."""
    total_score = 0
    for constraint_func, param in constraints:
        func = f"{constraint_func}("
        for param_key in param:
            if param_key == "object1" or param_key == "object2":
                func = func + param_key + "=" + f'assets["{param[param_key]}"]' + ", "
            elif param_key == "objects" or param_key == "assets":
                func = func + param_key + "=" + "["
                for obj_name in param[param_key]:
                    func += f'assets["{obj_name}"]' + ", "
                func += "]"
            else:
                func += param_key + "=" + param[param_key]
        func += ")"
        print(func)
        score = exec(func)
        print(score)
        
        total_score += score # Summing scores 
    return total_score

def adjust_positions(assets, adjustment_step=1):
    """Randomly adjust the positions of assets."""
    for asset in assets.items():
        # Randomly adjust position within a small range to explore the space
        name, object = asset 
        object.location = (
            object.location[0] + random.uniform(-adjustment_step, adjustment_step),
            object.location[1] + random.uniform(-adjustment_step, adjustment_step),
            object.location[2] # Z position kept constant for simplicity
            )
    return assets

def constraint_solving(assets, constraints, max_iterations=100):
    """Find an optimal layout of assets to maximize the score defined by constraints."""
    best_score = evaluate_constraints(assets, constraints)
    best_layout = assets # Assuming a copy method exists

    for _ in range(max_iterations):
        assets = adjust_positions(assets)
        current_score = evaluate_constraints(assets, constraints)
        print(f"current_score: {current_score}")
        if current_score > best_score:
            best_score = current_score
            best_layout = assets
        else:
            # Revert to best layout if no improvement
            assets = best_layout
    return best_layout, best_score

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)

def orientation_similarity(orientation1: Tuple[float, float, float], orientation2: Tuple[float, float, float]) -> float:
    """Calculate the similarity between two orientations, represented as Euler angles."""
    # Convert Euler angles to vectors for simplicity in comparison
    vector1 = np.array(orientation1)
    vector2 = np.array(orientation2)
    # Calculate the cosine similarity between the two orientation vectors
    cos_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cos_similarity

def parallelism_score(assets: List[Layout]) -> float:
    """
    Evaluates and returns a score indicating the degree of parallelism in a list of assets' layouts, considering both position and ori

    Args:
        assets (List[Layout]): A list of asset layouts.

    Returns:
    float: A score between 0 and 1 indicating the parallelism of the assets.
    """
    if len(assets) < 2:
        return 1.0 # Single asset or no asset is arbitrarily considered perfectly parallel
    
    # Positional parallelism
    vectors = [calculate_vector(assets[i].location, assets[i+1].location) for i in range(len(assets)-1)]
    normalized_vectors = [normalize_vector(v) for v in vectors]
    dot_products_position = [np.dot(normalized_vectors[i], normalized_vectors[i+1]) for i in range(len(normalized_vectors)-1)]

    # Rotational similarity
    orientation_similarities = [orientation_similarity(assets[i].orientation, assets[i+1].orientation) for i in range(len(assets)-1)]
    print(f"orientation_similarities: {orientation_similarities}")

    # Combine scores
    position_score = np.mean([0.5 * (dot + 1) for dot in dot_products_position])
    orientation_score = np.mean([(similarity + 1) / 2 for similarity in orientation_similarities])

    # Average the position and orientation scores for the final score
    final_score = (position_score + orientation_score) / 2
    return final_score

def calculate_distance(location1: Tuple[float, float, float], location2: Tuple[float, float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(location1) - np.array(location2))

def proximity_score(object1: Layout, object2: Layout, min_distance: float = 1.0, max_distance: float = 5.0) -> float:
    """
    Calculates a proximity score indicating how close two objects are, with 1 being very close and 0 being far apart.

    Args:
        object1 (Layout): The first object's layout.
        object2 (Layout): The second object's layout.
        min_distance (float): The distance below which objects are considered to be at optimal closeness. Scores 1.
        max_distance (float): The distance beyond which objects are considered too far apart. Scores 0.

    Returns:
        float: A score between 0 and 1 indicating the proximity of the two objects.
    """
    distance = calculate_distance(object1.location, object2.location)
    print(f"distance: {distance}")
    if distance <= min_distance:
        return 1.0
    elif distance >= max_distance:
        return 0.0
    else:
        # Linearly interpolate the score based on the distance
        return 1 - (distance - min_distance) / (max_distance - min_distance)

def euler_to_forward_vector(orientation: Tuple[float, float, float]) -> np.ndarray:
    """Convert Euler angles to a forward direction vector."""
    # Converting Euler angles to a forward direction vector 
    # involves transforming the rotational information encoded in Euler angles 
    # into a vector that points in the direction the object is facing. 

    pitch, yaw, _ = orientation
    # Assuming the angles are in radians
    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)
    return np.array([x, y, z])

def calculate_vector(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> np.ndarray:
    """Calculate the directional vector from point a to b."""
    return np.array(b) - np.array(a)

def direction_score(object1: Layout, object2: Layout) -> float:
    """
    Calculates a score indicating how directly object1 is targeting object2.

    Args:
        object1 (Layout): The first object's layout, assumed to be the one doing the targeting.
        object2 (Layout): The second object's layout, assumed to be the target.

    Returns:
        float: A score between 0 and 1 indicating the directionality of object1 towards object2.
    """
    forward_vector = euler_to_forward_vector(object1.orientation)
    target_vector = calculate_vector(object1.location, object2.location)

    # Normalize vectors to ensure the dot product calculation is based only on direction
    forward_vector_normalized = normalize_vector(forward_vector)
    target_vector_normalized = normalize_vector(target_vector)

    # Calculate the cosine of the angle between the two vectors
    cos_angle = np.dot(forward_vector_normalized, target_vector_normalized)

    # Map the cosine range [-1, 1] to a score range [0, 1]
    score = (cos_angle + 1) / 2
    return score

def alignment_score(assets: List[Layout], axis: str) -> float:
    """
    Calculates an alignment score for a list of assets along a specified axis.
    Args:
    assets (List[Layout]): A list of asset layouts to be evaluated for alignment.
    axis (str): The axis along which to evaluate alignment ('x', 'y', or 'z').
    Returns:
    float: A score between 0 and 1 indicating the degree of alignment along the specified axis.
    """
    if not assets or axis not in ['x', 'y', 'z']:
        return 0.0 # Return a score of 0 for invalid input

    # Axis index mapping to the location tuple
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]

    # Extract the relevant coordinate for each asset based on the chosen axis
    coordinates = [asset.location[axis_index] for asset in assets]
    # Calculate the variance of these coordinates
    variance = np.var(coordinates)
    # Inverse the variance to calculate the score, assuming a lower variance indicates better alignment
    # Normalize the score to be between 0 and 1, considering a reasonable threshold for "perfect" alignment
    threshold_variance = 1.0 # Define a threshold variance for "perfect" alignment
    score = 1 / (1 + variance / threshold_variance)
    # Clamp the score between 0 and 1
    score = max(0, min(score, 1))
    return score

def symmetry_score(assets: List[Layout], axis: str) -> float:
    """
    Calculates a symmetry score for a list of assets along a specified axis.

    Args:
    assets (List[Layout]): A list of asset layouts to be evaluated for symmetry.
    axis (str): The axis along which to evaluate symmetry ('x', 'y', or 'z').

    Returns:
    float: A score between 0 and 1 indicating the degree of symmetry along the specified axis.
    """
    if not assets or axis not in ['x', 'y', 'z']:
        return 0.0 # Return a score of 0 for invalid input
    # Axis index mapping to the location tuple
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]

    # Find the median coordinate along the specified axis to define the symmetry axis
    coordinates = [asset.location[axis_index] for asset in assets]
    symmetry_axis = np.median(coordinates)

    # Calculate the deviation from symmetry for each asset
    deviations = []
    for asset in assets:
        # Find the mirrored coordinate across the symmetry axis
        mirrored_coordinate = 2 * symmetry_axis - asset.location[axis_index]
        # Find the closest asset to this mirrored coordinate
        closest_distance = min(abs(mirrored_coordinate - other.location[axis_index]) for other in assets)
        deviations.append(closest_distance)

    # Calculate the average deviation from perfect symmetry
    avg_deviation = np.mean(deviations)

    # Convert the average deviation to a score, assuming smaller deviations indicate better symmetry
    # The scoring formula can be adjusted based on the specific requirements for symmetry in the application
    max_deviation = 10.0 # Define a maximum deviation for which the score would be 0
    score = max(0, 1 - avg_deviation / max_deviation)

    return score

def perpendicularity_score(object1: Layout, object2: Layout) -> float:
    """
    Calculates a score indicating how perpendicular two objects are, based on their forward direction vectors.
    
    Args:
        object1 (Layout): The first object's layout, including its orientation as Euler angles.
        object2 (Layout): The second object's layout, including its orientation as Euler angles.
    
    Returns:
    float: A score between 0 and 1 indicating the degree of perpendicularity.
    """
    vector1 = euler_to_forward_vector(object1.orientation)
    vector2 = euler_to_forward_vector(object2.orientation)
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    score = 1 - np.abs(cos_angle)
    return score

def calculate_volume(layout: Layout) -> float:
    """Calculate the volume of an object based on its layout dimensions."""
    length = abs(layout.max[0] - layout.min[0])
    width = abs(layout.max[1] - layout.min[1])
    height = abs(layout.max[2] - layout.min[2])
    return length * width * height

def evaluate_hierarchy(assets: List[Layout], expected_order: List[str]) -> float:
    """
    Evaluates how well a list of objects follows a specified hierarchical order based on size.

    Args:
        assets (List[Layout]): A list of asset layouts to be evaluated.
        expected_order (List[str]): A list of identifiers (names) for the assets, specifying the expected order of sizes.

    Returns:
        float: A metric indicating how well the actual sizes of the objects match the expected hierarchical order.
    """
    # Map identifiers to volumes
    id_to_volume = {asset_id: calculate_volume(asset) for asset_id, asset in zip(expected_order, assets)}

    # Calculate the actual order based on sizes
    actual_order = sorted(id_to_volume.keys(), key=lambda x: id_to_volume[x], reverse=True)

    # Evaluate the match between the expected and actual orders
    correct_positions = sum(1 for actual, expected in zip(actual_order, expected_order) if actual == expected)
    total_positions = len(expected_order)

    # Calculate the match percentage as a measure of hierarchy adherence
    match_percentage = correct_positions / total_positions
    return match_percentage

def calculate_angle_from_center(center: Tuple[float, float, float], object_location: Tuple[float, float, float]) -> float:
    """Calculate the angle of an object relative to a central point."""
    vector = np.array(object_location) - np.array(center)
    angle = np.arctan2(vector[1], vector[0])
    return angle

def rotation_uniformity_score(objects: List[Layout], center: Tuple[float, float, float]) -> float:
    """
    Calculates how uniformly objects are distributed around a central point in terms of rotation.
    
    Args:
        objects (List[Layout]): A list of object layouts to be evaluated.
        center (Tuple[float, float, float]): The central point around which objects are rotating.

    Returns:
        float: A score between 0 and 1 indicating the uniformity of object distribution around the center.
    """
    angles = [calculate_angle_from_center(center, obj.location) for obj in objects]
    angles = np.sort(np.mod(angles, 2*np.pi)) # Normalize angles to [0, 2\pi] and sort

    # Calculate differences between consecutive angles, including wrap-around difference
    angle_diffs = np.diff(np.append(angles, angles[0] + 2*np.pi))

    # Evaluate uniformity as the variance of these differences
    variance = np.var(angle_diffs)
    uniformity_score = 1 / (1 + variance) # Inverse variance, higher score for lower variance
    
    return uniformity_score

def get_all_vertices(objects):
    # Assuming obj_dict[moving_set_name] and obj_dict[target_set_name] are lists of bpy.types.Object
    all_vertices = set()
    for obj in objects:
        if obj.type == 'MESH':
            mesh = obj.data
            for vertex in mesh.vertices:
                all_vertices.add(tuple(vertex.co))
    return all_vertices

def put_ontop(obj_dict, moving_set_name, target_set_name, threshold, step):
    """
    Adjust objects in moving_set_name until the shortest distance to target_set_name is below the threshold.

    Args:
        obj_dict (dict): Dictionary of object sets.
        moving_set_name (str): The key for the set of objects to move.
        target_set_name (str): The key for the set of objects to calculate distance to.
        threshold (float): The distance threshold.
        step (float): The step by which to move objects in the Z direction.
    """
    while True:
        vertices_set1 = get_all_vertices(obj_dict[moving_set_name])
        vertices_set2 = get_all_vertices(obj_dict[target_set_name])
        shortest_distance = calculate_shortest_distance(vertices_set1, vertices_set2)
        print(shortest_distance)
        if shortest_distance < threshold:
            break

    for obj in obj_dict[moving_set_name]:
        obj.location.z -= max(step, shortest_distance)
    
    bpy.context.view_layer.update()

def repeat_object(original: Layout, direction: Tuple[float, float, float], repetitions: int, distance: float) -> List[Layout]:
    """
    Creates a series of duplicated objects based on the original, repeating them in a specified direction at a set distance.

    Args:
        original (Layout): The original object to be repeated.
        direction (Tuple[float, float, float]): The direction vector along which to repeat the object.
        repetitions (int): The number of times the object should be repeated.
        distance (float): The distance between each object.

    Returns:
        List[Layout]: A list of Layout objects representing the original and its duplicates.
    """
    repeated_objects = [original] # Include the original object in the output list
    for i in range(1, repetitions):
    # Calculate the new location for each repeated object
        new_location = (
            original.location[0] + direction[0] * distance * i,
            original.location[1] + direction[1] * distance * i,
            original.location[2] + direction[2] * distance * i
        )
    # Create a new Layout instance for each repetition
        new_object = Layout(
            location=new_location,
            min=original.min,
            max=original.max,
            orientation=original.orientation
        )
    repeated_objects.append(new_object)
    return repeated_objects

def add_camera(location: Tuple[float, float, float], target_point: Tuple[float, float, float], lens: float = 35) -> bpy.types.Object:
    """
    Add a camera to the Blender scene.

    Args:
        location (Vector): The location to place the camera.
        target_point (Vector): The point the camera should be aimed at.
        lens (float, optional): The lens size. Defaults to 35.

    Returns:
        bpy.types.Object: The created camera object.
    Example:
        camera = add_camera((10, 10, 10), (0, 0, 0))
    """
    # Create a new camera data object
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.lens = lens # Set the lens property

    # Create a new camera object and link it to the scene
    cam_object = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_object)

    # Set the camera location
    cam_object.location = location

    # Calculate the direction vector from the camera to the target point
    direction = Vector(target_point) - Vector(location)
    # Orient the camera to look at the target point
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_object.rotation_euler = rot_quat.to_euler()
    
    # Set the created camera as the active camera in the scene
    bpy.context.scene.camera = cam_object
    
    return cam_object

def import_object(object_path: str, location: Tuple[float, float, float], orientation: Tuple[float, float, float]) -> bpy.types.Object:
    """
    Imports an .fbx object into Blender and positions/orients it according to the given parameters.

    Parameters:
    object_path (str): The path of the object to import (.fbx file) 
    location (Tuple[float, float, float]): The location (x, y, z) where the object should be placed.
    orientation (Tuple[float, float, float]): The orientation (pitch, yaw, roll in radians) of the object.

    Returns:
    bpy.types.Object: The imported object in Blender.
    """

    if (object_path.split('.')[1] == 'fbx'):
        # Import .fbx file
        fbx_filepath = object_path  
        bpy.ops.import_scene.fbx(filepath=fbx_filepath)
    elif (object_path.split('.')[1] == 'obj'):
        # Import .obj file
        obj_filepath = object_path
        bpy.ops.wm.obj_import(filepath=obj_filepath)

    # Get a reference to the imported object
    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if imported_objects:
        imported_object = imported_objects[0]
        
        # Set location
        imported_object.location = location

        # Set orientation (convert Euler angles to radians)
        orientation_rad = tuple(angle for angle in orientation)
        imported_object.rotation_euler = orientation_rad

        # Example of calculating min and max bounds
        min_bound = (
            imported_object.location[0] - imported_object.dimensions[0] / 2,
            imported_object.location[1] - imported_object.dimensions[1] / 2,
            imported_object.location[2] - imported_object.dimensions[2] / 2
        )
        max_bound = (
            imported_object.location[0] + imported_object.dimensions[0] / 2,
            imported_object.location[1] + imported_object.dimensions[1] / 2,
            imported_object.location[2] + imported_object.dimensions[2] / 2
        )

        # Create and return Layout dataclass instance
        layout = Layout(location=imported_object.location,
                        min=min_bound,
                        max=max_bound,
                        orientation=orientation_rad,
                        imported_object=imported_object)
        
        return layout

    else:
        return None

class Movement():
    def __init__(self, frame_start, frame_end, speed, trajectory, object):
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.speed = speed
        self.trajectory = trajectory 
        self.object = object

    def move(self, current_frame):
        if (current_frame >= self.frame_start) and (current_frame <= self.frame_end):    
            trajectory, param = self.trajectory

            # Current location
            x = self.object.imported_object.location[0]
            y = self.object.imported_object.location[1]
            z = self.object.imported_object.location[2]

            # The total number of step
            number_step = self.frame_end - self.frame_start + 1

            # Step
            step = current_frame - self.frame_start + 1 

            # Set the new location
            self.object.imported_object.location = trajectory(x=x, 
                                                              y=y, 
                                                              z=z, 
                                                              step=step,
                                                              number_step=number_step, 
                                                              speed=self.speed,
                                                              **param)

            # Insert keyframe for the new location
            self.object.imported_object.keyframe_insert(data_path="location", frame=current_frame)

def circle(location: Tuple[float, float, float], step: float, number_step: float, speed: float, radius: float):
    angle = step/number_step * 2 * math.pi
    x = location[0] + speed * radius * math.cos(angle)
    y = location[1] + speed * radius * math.sin(angle)
    z = location[2]
    return (x, y, z)

def straight(location: Tuple[float, float, float], step: float, number_step: float, speed: float, orientation: Tuple[float, float, float]):
    x = location[0] + speed * (number_step - step) * orientation[0] * 0.01
    y = location[1] + speed * (number_step - step) * orientation[1] * 0.01
    z = location[2] + speed * (number_step - step) * orientation[2] * 0.01
    return (x, y, z)

if __name__ == '__main__':
    input = 'a girl play with her dog in the garden'

    # Phase 1: Xác định các vật thể có xuất hiện trong video
    context = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets). 
After listing the assets, structure them in this format:
object_list = [
  {{"name": x1, "description": y1}},
  {{"name": x2, "description": y2}},
  {{"name": x3, "description": y3}},
  ...
]
Describe each asset with a concise name (x) and a detailed visual description (y). 
Avoid using normal text; format your response strictly as specified above.

Natural language description: {input}
"""

    output_phase_1 = """
object_list = [
    {"name": "girl", "description": "A young girl with long hair, wearing a summer dress, smiling and looking happy."},
    {"name": "dog", "description": "A medium-sized dog with a fluffy coat, wagging its tail, and looking energetic."},
    {"name": "grass", "description": "A patch of green grass, well-trimmed and lush."},
    {"name": "flower1", "description": "A bright red rose in full bloom."},
    {"name": "flower2", "description": "A cluster of yellow daisies, vibrant and lively."},
    {"name": "tree1", "description": "A tall oak tree with a thick trunk and a wide canopy providing shade."},
    {"name": "tree2", "description": "A smaller cherry blossom tree with pink flowers blooming."}
]
"""
    exec(output_phase_1)

    # Phase 3: Khởi tạo môi trường sự kiện
    
    # Phase 3.1: Phân loại các vật thể vừa xác định thành 2 trường: 
        # Base Evironments objects
        # Main character and Creatures objects
    context = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural descriptions.
Your job is to classify the objects from the objects list below and natural descriptions into two groups: one group for objects used to create the base environment, and another group for objects that are the main characters and creatures in the animation.

Objects list:
{output_phase_1}

Natural language description: {input}

After listing the assets, structure them in this format:
env_objs = [name_obj1, name_obj2, ...]
main_objs = [name_obj4, name_obj5, ...]

Avoid using normal text; format your response strictly as specified above.
"""

    output_phase_3_1 = """
env_objs = ["grass", "flower1", "flower2", "tree1", "tree2"]
main_objs = ["girl", "dog"]
    """
    exec(output_phase_3_1)

    # Phase 3.2: Khởi tạo môi trường
    # Phase 3.2.1: Sinh ra mô tả chung về layout
    context = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to create a concrete plan to put them into the scene from the objects list below and natural descriptions.
Please think step by step, and give me a multi-step plan to put assets into the
scene.

Objects list:
{output_phase_1}

Natural language description: {input}

For each step, structure your output as:

layout_plan_i = {{
"title": title_i,
"asset_list": [asset_name_1, asset_name_2],
"description": desc_i
}}

where title_i is the high-level name for this step, and desc is detailed visual text description of what it shall look like after layout. 
Avoid using normal text; format your response strictly as specified above.
"""

    output_phase_3_2_1 ="""
layout_plan_1 = {
    "title": "Set Up Grass",
    "asset_list": ["grass"],
    "description": "Cover the ground with a well-trimmed, lush patch of green grass to create the base environment."
}

layout_plan_2 = {
    "title": "Place Trees",
    "asset_list": ["tree1", "tree2"],
    "description": "Position the tall oak tree to one side of the scene to provide shade, and place the smaller cherry blossom tree with pink flowers blooming on the other side to add color and depth."
}

layout_plan_3 = {
    "title": "Add Flowers",
    "asset_list": ["flower1", "flower2"],
    "description": "Scatter the bright red rose in full bloom and the cluster of vibrant yellow daisies around the garden to add a touch of vibrancy and color."
}

layout_plan_4 = {
    "title": "Position Girl",
    "asset_list": ["girl"],
    "description": "Place the young girl with long hair, wearing a summer dress, in the middle of the garden, smiling and looking happy."
}

layout_plan_5 = {
    "title": "Place Dog",
    "asset_list": ["dog"],
    "description": "Position the medium-sized dog with a fluffy coat next to the girl, wagging its tail energetically, to show that the girl is playing with her dog in the garden."
}     
"""   
    exec(output_phase_3_2_1)

    # Phase 3.2.2: Sinh ra vị trí location khởi tạo của các vật thể và các ràng buộc giữa các vật thể dựa vào lay_out_plan
    context = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list and the layout plan.
Please think step by step.

Objects list:
{output_phase_1}

Natural language description: {input}

Constraints: 
proximity_score(object1: Layout, object2: Layout): A constraint enforcing the closeness of two objects, e.g., a chair near a table.
direction_score(object1: Layout, object2: Layout): The angle of one object is targeting at the other.
alignment_score(assets: List[Layout], axis: str): Ensuring objects align along a common axis (x, y, z), e.g., paintings aligned vertically on a wall.
symmetry_score(assets: List[Layout], axis: str): Mirroring objects along an axis (x, y, z), e.g., symmetrical placement of lamps on either side of a bed.
parallelism_score(assets: List[Layout]): Objects parallel to each other, suggesting direction, e.g., parallel rows of seats in a theater.
perpendicularity_score(object1: Layout, object2: Layout): Objects intersecting at a right angle, e.g., a bookshelf perpendicular to a desk.
rotation_uniformity_score(objects: List[Layout], center: Tuple[float, float, float]): a list of objects rotate a cirtain point, e.g., rotating chairs around a meeting table.
repeat_object(original: Layout, direction: Tuple[float, float, float], repetitions: int, distance: float): Repeating patterns for rhythm or emphasis, e.g., a sequence of street lights.
scale_group(objects: List[Layout], scale_factor: float): Adjusting object sizes for depth or focus, e.g., smaller background trees to create depth perception.

Layout plan:
{output_phase_3_2_1}   

The answer should include 2 lists, initial_position and constraints, where initial_positions is a dictionary with keys as object names and values as their initial positions, and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.

After determining initial_position and constraints, structure them in this format:
initial_position = {{key: value, ...}}
constraints = [(constraint1, ("param1": "object1", ...)), ...]

Avoid using normal text; format your response strictly as specified above.
"""    

    output_phase_3_2_2 = """
initial_position = {
    "grass": (0, 0, 0),
    "tree1": (-5, 0, 0),
    "tree2": (5, 0, 0),
    "flower1": (-2, 0, 2),
    "flower2": (2, 0, 2),
    "girl": (0, 0, 0),
    "dog": (1, 0, 0)
}

constraints = [
    ("proximity_score", {"object1": "girl", "object2": "dog"}),
    ("direction_score", {"object1": "girl", "object2": "dog"}),
    ("alignment_score", {"assets": ["tree1", "tree2"], "axis": "y"}),
    ("symmetry_score", {"assets": ["tree1", "tree2"], "axis": "x"}),
    ("rotation_uniformity_score", {"objects": ["tree1", "tree2"], "center": (0, 0, 0)}),
    ("scale_group", {"objects": ["tree1", "tree2"], "scale_factor": 1.0})
]
"""    
    exec(output_phase_3_2_2)

# Phase 2: Import các vật thể từ trong database 3D object
    assets = {}
    for object in object_list:
        assets[object["name"]] = None
        import_obj = f'assets["{object["name"]}"] = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/{object["name"]}.obj", location={initial_position[object["name"]]}, orientation=(0, 0, 0))'
        exec(import_obj)        

# Phase 3.2.3: Thực hiện khởi tạo môi trường
    best_layout, best_score = constraint_solving(assets=assets, constraints=constraints)

    for asset in best_layout.items():
        name, object = asset
        print(object.location)
        # Set location
        object.imported_object.location = object.location

        # Set orientation (convert Euler angles to radians)
        orientation_rad = tuple(angle for angle in object.orientation)
        object.imported_object.rotation_euler = orientation_rad

    bpy.context.view_layer.update()

# Phase 3.2.4: Sinh ra kịch bản chuyển động của các đối tượng chính
    context = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to script the animation sequences for objects based on natural language descriptions, the list of objects and their initial positions, and motion trajectory functions below.
Please think step by step.

Natural language description: a girl play with her dog in the garden

Objects and their constraints, initial position:
{initial_position}

Motion trajectory functions:
    circle(radius: float)
        Moves in a circular trajectory around the origin.
        Parameters:
        radius: Radius of the circle.


    straight(orientation: Tuple[float, float, float]):
        Moves in a straight line trajectory.
        Parameters:
        orientation: Tuple indicating the direction of movement (x, y, z).
    
Your answer should be formatted as a dictionary with two main keys: total_frames and motions, where total_frames represents the total number of frames in the video, formatted as an integer, and motions is a list of motions that will occur in the video, where each element contains fields including start_frame, end_frame, speed, and motion trajectory function        
After determining your answer, structure them in this format:
trajectory = {{
    "total_frames": total_frame,
    "motions": [
        {{"frame_start": frame_start, "frame_end": frame_end, "trajectory": (trajectory, {{"param1": value1, ...}}), "speed": speed, "object": object}}, 
        ...
            ]
}}
Avoid using normal text; format your response strictly as specified above.
"""

    output_phase_3_2_4 = """
trajectory = {
    "total_frames": 200,
    "motions": [
        {"frame_start": 0, "frame_end": 100, "trajectory": ("circle", {"radius": 2}), "speed": 1, "object": "girl"},
        {"frame_start": 0, "frame_end": 100, "trajectory": ("circle", {"radius": 2}), "speed": 1.2, "object": "dog"},
        {"frame_start": 100, "frame_end": 200, "trajectory": ("straight", {"orientation": (1, 0, 0)}), "speed": 1, "object": "girl"},
        {"frame_start": 100, "frame_end": 200, "trajectory": ("straight", {"orientation": (1, 0, 0)}), "speed": 1.2, "object": "dog"}
    ]
}
"""
    exec(output_phase_3_2_4)

    # Create movement to blender
    num_frames = trajectory["total_frames"]
    mov_list = []
    for i, motion in enumerate(trajectory["motions"]):
        func = f'mov{i} = Movement(frame_start={motion["frame_start"]}, frame_end={motion["frame_end"]}, trajectory={motion["trajectory"]}, speed={motion["speed"]}, object={assets[motion["object"]]})'
        exec(func)
        mov_list.append(f"mov{i}")

    for frame in range(num_frames):
        for mov in mov_list:
            func = f"{mov}.move(current_frame={frame})"
            exec(func)

    bpy.context.scene.frame_end = num_frames 
    
        
