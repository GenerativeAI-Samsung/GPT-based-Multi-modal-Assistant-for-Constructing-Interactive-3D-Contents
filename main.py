from animate import *

if __name__ == '__main__':
    input = 'a girl play with her dog in the garden'

    # Phase 1: Xác định các vật thể có xuất hiện trong video
    context = """
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets). 
After listing the assets, structure them in this format:
[
  {"name": x1, "description": y1},
  {"name": x2, "description": y2},
  {"name": x3, "description": y3},
  ...
]
Describe each asset with a concise name (x) and a detailed visual description (y). 
Avoid using normal text; format your response strictly as specified above.

Natural language description: a girl play with her dog in the garden
"""

    output_phase_1 = """
[
    {"name": "girl", "description": "A young girl with long hair, wearing a summer dress, smiling and looking happy."},
    {"name": "dog", "description": "A medium-sized dog with a fluffy coat, wagging its tail, and looking energetic."},
    {"name": "grass", "description": "A patch of green grass, well-trimmed and lush."},
    {"name": "flower1", "description": "A bright red rose in full bloom."},
    {"name": "flower2", "description": "A cluster of yellow daisies, vibrant and lively."},
    {"name": "tree1", "description": "A tall oak tree with a thick trunk and a wide canopy providing shade."},
    {"name": "tree2", "description": "A smaller cherry blossom tree with pink flowers blooming."}
]
"""

    # Phase 2: Import các vật thể từ trong database 3D object
    girl = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/girl.obj", location=(0, 0, 0), orientation=(0, 0, 0))
    dog = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/dog.obj", location=(0, 0, 0), orientation=(0, 0, 0))
    grass = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/grass.obj", location=(0, 0, 0), orientation=(0, 0, 0))
    flower1 = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/flower1.obj", location=(0, 0, 0), orientation=(0, 0, 0))
    flower2 = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/flower2.obj", location=(0, 0, 0), orientation=(0, 0, 0))
    tree1 = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/tree1.obj", location=(0, 0, 0), orientation=(0, 0, 0))
    tree2 = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/tree2.obj", location=(0, 0, 0), orientation=(0, 0, 0))

    # Phase 3: Khởi tạo môi trường sự kiện
    
    # Phase 3.1: Phân loại các vật thể vừa xác định thành 2 trường: 
        # Base Evironments objects
        # Main character and Creatures objects
    context = """
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural descriptions.
Your job is to classify the objects from the objects list below and natural descriptions into two groups: one group for objects used to create the base environment, and another group for objects that are the main characters and creatures in the animation.

Objects list:
[
    {"name": "girl", "description": "A young girl with long hair, wearing a summer dress, smiling and looking happy."},
    {"name": "dog", "description": "A medium-sized dog with a fluffy coat, wagging its tail, and looking energetic."},
    {"name": "grass", "description": "A patch of green grass, well-trimmed and lush."},
    {"name": "flower1", "description": "A bright red rose in full bloom."},
    {"name": "flower2", "description": "A cluster of yellow daisies, vibrant and lively."},
    {"name": "tree1", "description": "A tall oak tree with a thick trunk and a wide canopy providing shade."},
    {"name": "tree2", "description": "A smaller cherry blossom tree with pink flowers blooming."}
]

Natural language description: a girl play with her dog in the garden

After listing the assets, structure them in this format:
env_objs = [name_obj1, name_obj2, ...]
main_objs = [name_obj4, name_obj5, ...]

Avoid using normal text; format your response strictly as specified above.
"""

    output_phase_3_1 = """
    env_objs = ["grass", "flower1", "flower2", "tree1", "tree2"]
    main_objs = ["girl", "dog"]
    """

    # Phase 3.2: Khởi tạo môi trường
    # Phase 3.2.1: Sinh ra mô tả chung về layout
    context = """
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to create a concrete plan to put them into the scene from the objects list below and natural descriptions.
Please think step by step, and give me a multi-step plan to put assets into the
scene.

Objects list:
[
    {"name": "girl", "description": "A young girl with long hair, wearing a summer dress, smiling and looking happy."},
    {"name": "dog", "description": "A medium-sized dog with a fluffy coat, wagging its tail, and looking energetic."},
    {"name": "grass", "description": "A patch of green grass, well-trimmed and lush."},
    {"name": "flower1", "description": "A bright red rose in full bloom."},
    {"name": "flower2", "description": "A cluster of yellow daisies, vibrant and lively."},
    {"name": "tree1", "description": "A tall oak tree with a thick trunk and a wide canopy providing shade."},
    {"name": "tree2", "description": "A smaller cherry blossom tree with pink flowers blooming."}
]

Natural language description: a girl play with her dog in the garden

For each step, structure your output as:

layout_plan_i = {
"title": title_i,
"asset_list": [asset_name_1, asset_name_2],
"description": desc_i
}

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

    # Phase 3.2.2: Sinh ra vị trí location khởi tạo của các vật thể và các ràng buộc giữa các vật thể dựa vào lay_out_plan
    context = """
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list and the layout plan.
Please think step by step.

Objects list:
[
    {"name": "girl", "description": "A young girl with long hair, wearing a summer dress, smiling and looking happy."},
    {"name": "dog", "description": "A medium-sized dog with a fluffy coat, wagging its tail, and looking energetic."},
    {"name": "grass", "description": "A patch of green grass, well-trimmed and lush."},
    {"name": "flower1", "description": "A bright red rose in full bloom."},
    {"name": "flower2", "description": "A cluster of yellow daisies, vibrant and lively."},
    {"name": "tree1", "description": "A tall oak tree with a thick trunk and a wide canopy providing shade."},
    {"name": "tree2", "description": "A smaller cherry blossom tree with pink flowers blooming."}
]

Natural language description: a girl play with her dog in the garden

Constraints: 
proximity(object1: Layout, object2: Layout): A constraint enforcing the closeness of two objects, e.g., a chair near a table.
direction(object1: Layout, object2: Layout): The angle of one object is targeting at the other.
alignment(assets: List[Layout], axis: str): Ensuring objects align along a common axis (x, y, z), e.g., paintings aligned vertically on a wall.
symmetry(assets: List[Layout], axis: str): Mirroring objects along an axis (x, y, z), e.g., symmetrical placement of lamps on either side of a bed.
overlap(object1: Layout, object2: Layout): One object partially covering another, creating depth, e.g., a rug under a coffee table.
parallelism(assets: List[Layout]): Objects parallel to each other, suggesting direction, e.g., parallel rows of seats in a theater.
perpendicularity(object1: Layout, object2: Layout): Objects intersecting at a right angle, e.g., a bookshelf perpendicular to a desk.
rotation(objects: List[Layout], center: Tuple[float, float, float]): a list of objects rotate a cirtain point, e.g., rotating chairs around a meeting table.
repetition(original: Layout, direction: Tuple[float, float, float], repetitions: int, distance: float): Repeating patterns for rhythm or emphasis, e.g., a sequence of street lights.
scaling(objects: List[Layout], scale_factor: float): Adjusting object sizes for depth or focus, e.g., smaller background trees to create depth perception.

Layout plan:
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

The answer should include 2 lists, initial_position and constraints, where initial_positions is a dictionary with keys as object names and values as their initial positions, and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.

After determining initial_position and constraints, structure them in this format:
initial_position = {key: value, ...}
constraints = [constraint1(param1=object1, ...), ...]

Avoid using normal text; format your response strictly as specified above.
"""    

    output_phase_3_2_2 = """
initial_position = {
    "grass": (0, 0, 0),
    "tree1": (-5, 0, 0),
    "tree2": (5, 0, 0),
    "flower1": (-3, 0, 3),
    "flower2": (3, 0, 3),
    "girl": (0, 0, 0),
    "dog": (1, 0, 0)
}

constraints = [
    proximity(object1="girl", object2="dog"),
    direction(object1="girl", object2="dog"),
    alignment(assets=["tree1", "tree2"], axis="y"),
    symmetry(assets=["tree1", "tree2"], axis="x"),
    overlap(object1="girl", object2="tree1"),
    overlap(object1="girl", object2="tree2"),
    rotation(objects=["tree1", "tree2"], center=(0, 0, 0)),
    scaling(objects=["tree1", "tree2"], scale_factor=1.0)
]
"""    

# Phase 3.2.3: Thực hiện khởi tạo môi trường

# Phase 3.2.4: Sinh ra kịch bản chuyển động của các đối tượng chính
context = """
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to script the animation sequences for objects based on natural language descriptions, the list of objects and their initial positions, and motion trajectory functions below.
Please think step by step.

Natural language description: a girl play with her dog in the garden

Objects and their initial position:
[
    {"object_name": "girl", "initial_position": (0, 0, 0)},
    {"object_name": "dog", "initial_position": (1, 0, 0)}
]

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
{
    "total_frames": total_frame,
    "motions": [
        {"frame_start": frame_start, "frame_end": frame_end, "trajectory": (trajectory, {"param1": value1, ...}), "speed": speed, "object": object}, 
        ...
            ]
}
Avoid using normal text; format your response strictly as specified above.
"""
output_phase_3_2_4 = """
{
    "total_frames": 100,
    "motions": [
        {"frame_start": 0, "frame_end": 99, "trajectory": ("circle", {"radius": 2.0}), "speed": 0.1, "object": "girl"},
        {"frame_start": 0, "frame_end": 99, "trajectory": ("straight", {"orientation": (1.0, 0.0, 0.0)}), "speed": 0.1, "object": "dog"}
    ]
}
"""