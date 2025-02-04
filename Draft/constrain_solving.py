import random
import numpy as np

def calculate_distance(location1, location2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(location1) - np.array(location2))

def searching_asset(assets, base_assets):
    list_index = []
    for i, asset in enumerate(base_assets):
        if (asset["name"] in assets):
            list_index.append(i)
    return list_index

def proximity_score(score, base_assets, object1, object2, min_distance=0.5, max_distance= 20.0):
    assets = [object1, object2]
    list_index = searching_asset(assets=assets, base_assets=base_assets)

    distance = calculate_distance(base_assets[list_index[0]]["position"], base_assets[list_index[0]]["position"])
    print(f"    distance: {distance}")
    if distance <= min_distance:
        score[0] =  1.0
    elif distance >= max_distance:
        score[0] = 0.0
    else:
        # Linearly interpolate the score based on the distance
        score[0] = 1 - (distance - min_distance) / (max_distance - min_distance)
    return score[0]

def euler_to_forward_vector(orientation):
    pitch, yaw, _ = orientation
    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)
    return np.array([x, y, z])

def direction_score(score, base_assets, object1, object2):
    """
    Calculates a score indicating how directly object1 is targeting object2.

    Args:
        object1 (Layout): The first object's layout, assumed to be the one doing the targeting.
        object2 (Layout): The second object's layout, assumed to be the target.

    Returns:
        float: A score between 0 and 1 indicating the directionality of object1 towards object2.
    """

    assets = [object1, object2]
    list_index = searching_asset(assets=assets, base_assets=base_assets)

    forward_vector = euler_to_forward_vector(base_assets[list_index[0]]["orientation"])
    target_vector = calculate_vector(base_assets[list_index[0]]["orientation"], base_assets[list_index[1]]["orientation"])

    # Normalize vectors to ensure the dot product calculation is based only on direction
    forward_vector_normalized = normalize_vector(forward_vector)
    target_vector_normalized = normalize_vector(target_vector)

    # Calculate the cosine of the angle between the two vectors
    cos_angle = np.dot(forward_vector_normalized, target_vector_normalized)

    # Map the cosine range [-1, 1] to a score range [0, 1]
    score[0] = (cos_angle + 1) / 2
    return score[0]

def alignment_score(score, base_assets, assets, axis):
    """
    Calculates an alignment score for a list of assets along a specified axis.
    Args:
    assets (List[Layout]): A list of asset layouts to be evaluated for alignment.
    axis (str): The axis along which to evaluate alignment ('x', 'y', or 'z').
    Returns:
    float: A score between 0 and 1 indicating the degree of alignment along the specified axis.
    """

    list_index = searching_asset(assets=assets, base_assets=base_assets)

    if not assets or axis not in ['x', 'y', 'z']:
        score[0] = 0
        return score[0] # Return a score of 0 for invalid input

    # Axis index mapping to the location tuple
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]

    # Extract the relevant coordinate for each asset based on the chosen axis
    coordinates = [base_assets[index]["position"][axis_index] for index in list_index]
    # Calculate the variance of these coordinates
    variance = np.var(coordinates)
    # Inverse the variance to calculate the score, assuming a lower variance indicates better alignment
    # Normalize the score to be between 0 and 1, considering a reasonable threshold for "perfect" alignment
    threshold_variance = 1.0 # Define a threshold variance for "perfect" alignment
    score[0] = 1 / (1 + variance / threshold_variance)
    # Clamp the score between 0 and 1
    score[0] = max(0, min(score[0], 1))
    return score[0]

def symmetry_score(score, base_assets, assets, axis):
    """
    Calculates a symmetry score for a list of assets along a specified axis.

    Args:
    assets (List[Layout]): A list of asset layouts to be evaluated for symmetry.
    axis (str): The axis along which to evaluate symmetry ('x', 'y', or 'z').

    Returns:
    float: A score between 0 and 1 indicating the degree of symmetry along the specified axis.
    """

    list_index = searching_asset(assets=assets, base_assets=base_assets)

    if not assets or axis not in ['x', 'y', 'z']:
        score[0] = 0
        return score[0] # Return a score of 0 for invalid input
    
    # Axis index mapping to the location tuple
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]

    # Find the median coordinate along the specified axis to define the symmetry axis
    coordinates = [base_assets[i]["position"] for i in list_index]
    symmetry_axis = np.median(coordinates)

    # Calculate the deviation from symmetry for each asset
    deviations = []
    for index in list_index:
        # Find the mirrored coordinate across the symmetry axis
        mirrored_coordinate = 2 * symmetry_axis - base_assets[index]["position"][axis_index]
        # Find the closest asset to this mirrored coordinate

        temp_list = list_index.copy
        temp_list.remove(index)

        closest_distance = min(abs(mirrored_coordinate - base_assets[others_index]["position"][axis_index]) for others_index in temp_list)
        deviations.append(closest_distance)

    # Calculate the average deviation from perfect symmetry
    avg_deviation = np.mean(deviations)

    # Convert the average deviation to a score, assuming smaller deviations indicate better symmetry
    # The scoring formula can be adjusted based on the specific requirements for symmetry in the application
    max_deviation = 10.0 # Define a maximum deviation for which the score would be 0
    score[0] = max(0, 1 - avg_deviation / max_deviation)

def calculate_vector(a, b):
    """Calculate the directional vector from point a to b."""
    return np.array(b) - np.array(a)

def normalize_vector(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)

def orientation_similarity(orientation1, orientation2):
    """Calculate the similarity between two orientations, represented as Euler angles."""
    # Convert Euler angles to vectors for simplicity in comparison
    vector1 = np.array(orientation1)
    vector2 = np.array(orientation2)
    # Calculate the cosine similarity between the two orientation vectors
    cos_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cos_similarity

def parallelism_score(score, base_assets, assets):
    """
    Evaluates and returns a score indicating the degree of parallelism in a list of assets' layouts, considering both position and ori

    Args:
        assets (List[Layout]): A list of asset layouts.

    Returns:
    float: A score between 0 and 1 indicating the parallelism of the assets.
    """

    list_index = searching_asset(assets=assets, base_assets=base_assets)

    if len(assets) < 2:
        score[0] = 1.0
        return score[0] # Single asset or no asset is arbitrarily considered perfectly parallel
    
    # Positional parallelism
    vectors = [calculate_vector(base_assets[list_index[i]]["position"], base_assets[list_index[i+1]]["position"]) for i in range(len(list_index)-1)]
    normalized_vectors = [normalize_vector(v) for v in vectors]
    dot_products_position = [np.dot(normalized_vectors[i], normalized_vectors[i+1]) for i in range(len(normalized_vectors)-1)]

    # Rotational similarity
    orientation_similarities = [orientation_similarity(base_assets[list_index[i]]["orientation"], base_assets[list_index[i+1]]["orientation"]) for i in range(len(list_index)-1)]

    # Combine scores
    position_score = np.mean([0.5 * (dot + 1) for dot in dot_products_position])
    orientation_score = np.mean([(similarity + 1) / 2 for similarity in orientation_similarities])

    # Average the position and orientation scores for the final score
    if (len(dot_products_position) != 0):
        score[0] = (position_score + orientation_score) / 2
    else:
        score[0] = (0 + orientation_score) / 2

    return score[0]

def perpendicularity_score(score, base_assets, object1, object2):
    """
    Calculates a score indicating how perpendicular two objects are, based on their forward direction vectors.
    
    Args:
        object1 (Layout): The first object's layout, including its orientation as Euler angles.
        object2 (Layout): The second object's layout, including its orientation as Euler angles.
    
    Returns:
    float: A score between 0 and 1 indicating the degree of perpendicularity.
    """

    assets = [object1, object2]
    list_index = searching_asset(assets=assets, base_assets=base_assets)

    vector1 = euler_to_forward_vector(base_assets[list_index[0]]["orientation"])
    vector2 = euler_to_forward_vector(base_assets[list_index[1]]["orientation"])
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    score[0] = 1 - np.abs(cos_angle)
    return score[0]

def calculate_angle_from_center(center, object_location) -> float:
    """Calculate the angle of an object relative to a central point."""
    vector = np.array(object_location) - np.array(center)
    angle = np.arctan2(vector[1], vector[0])
    return angle

def rotation_uniformity_score(score, base_assets, objects, center) -> float:
    """
    Calculates how uniformly objects are distributed around a central point in terms of rotation.
    
    Args:
        objects (List[Layout]): A list of object layouts to be evaluated.
        center (Tuple[float, float, float]): The central point around which objects are rotating.

    Returns:
        float: A score between 0 and 1 indicating the uniformity of object distribution around the center.
    """

    list_index = searching_asset(assets=objects, base_assets=base_assets)

    angles = [calculate_angle_from_center(center, base_assets[index]["position"]) for index in list_index]
    angles = np.sort(np.mod(angles, 2*np.pi)) # Normalize angles to [0, 2\pi] and sort

    # Calculate differences between consecutive angles, including wrap-around difference
    angle_diffs = np.diff(np.append(angles, angles[0] + 2*np.pi))

    # Evaluate uniformity as the variance of these differences
    variance = np.var(angle_diffs)
    score[0] = 1 / (1 + variance) # Inverse variance, higher score for lower variance
    
    return score[0]

def adjust_positions_orientations(assets, adjustment_step=0.01):
    """Randomly adjust the positions of assets."""
    for asset in assets:
        # Randomly adjust position within a small range to explore the space
        asset["position"][0] += random.uniform(-adjustment_step, adjustment_step) # Modify X position  
        asset["position"][1] += random.uniform(-adjustment_step, adjustment_step) # Modify Y position  
        # Z position kept constant for simplicity

        # Only orientation of Z axis
        asset["orientation"][3] += random.uniform(-adjustment_step, adjustment_step) # Modify X position  
        
    return assets

def evaluate_constraints(assets, constraints):
    """Evaluate all constraints and return the overall score."""
    total_score = 0
    score = [0]

    for constraint_func, param in constraints:
        func = f"{constraint_func}(score=score, base_assets=assets, param**)"
        print(func)
        exec(func)        
        total_score += score[0] # Summing scores
        print(f"-> score = {score[0]}; total score = {total_score}\n")

    return total_score

def constraint_solving(assets, constraints, max_iterations=100000):
    """Find an optimal layout of assets to maximize the score defined by constraints."""
    best_score = evaluate_constraints(assets, constraints)
    best_layout = assets.copy() 

    for _ in range(max_iterations):
        assets = adjust_positions_orientations(assets)
        current_score = evaluate_constraints(assets, constraints)
        print(f"current_score: {current_score}")
        if current_score > best_score:
            best_score = current_score
            best_layout = assets.copy()
        else:
            # Revert to best layout if no improvement
            assets = best_layout
    return best_layout, best_score
