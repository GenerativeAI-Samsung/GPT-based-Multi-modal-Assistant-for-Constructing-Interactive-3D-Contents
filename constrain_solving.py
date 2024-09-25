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

# def parallelism_score(score, base_assets, assets):
#     """
#     Evaluates and returns a score indicating the degree of parallelism in a list of assets' layouts, considering both position and ori

#     Args:
#         assets (List[Layout]): A list of asset layouts.

#     Returns:
#     float: A score between 0 and 1 indicating the parallelism of the assets.
#     """
#     if len(assets) < 2:
#         score[0] = 1.0
#         return # Single asset or no asset is arbitrarily considered perfectly parallel
    
#     # Positional parallelism
#     vectors = [calculate_vector(base_assets[assets[i]], base_assets[assets[i+1]]) for i in range(len(assets)-1)]
#     normalized_vectors = [normalize_vector(v) for v in vectors]
#     dot_products_position = [np.dot(normalized_vectors[i], normalized_vectors[i+1]) for i in range(len(normalized_vectors)-1)]

#     # Combine scores
#     position_score = np.mean([0.5 * (dot + 1) for dot in dot_products_position])

#     print(f"position_score: {position_score}")

#     # Average the position and orientation scores for the final score
#     if (len(dot_products_position) != 0):
#         score[0] = position_score
#     else:
#         score[0] = 0 

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
