if __name__ == '__main__':
    # step 1: -> Base on the demand, Language Model decide which step should be change to response.
    prompt = """
Your task is to determine which steps in the following 3D scene construction process need to be adjusted to meet the user's requirements:

Step 1: Identify the objects that will appear in the 3D scene.
-> object_list = [
  {"name": "garden_grassy_field", "description": "A large, lush, green grassy field serving as the base of the garden."},
  {"name": "playground_slide", "description": "A brightly colored children's slide made of plastic, with a smooth surface and gentle slope."},
  {"name": "playground_swing", "description": "A single swing with a wooden seat and metal chains, positioned in the garden area."},
  {"name": "little_girl", "description": "A young girl, around 7-10 years old, wearing a bright yellow sweater, a colorful dress, and high heels. She has an energetic and playful appearance."},
  {"name": "medium_dog", "description": "A medium-sized dog with a friendly demeanor, either standing or playfully interacting with the girl."}
]

Step 2: Classify the types of the identified objects.
-> env_objs = [
  "garden_grassy_field",
  "playground_slide",
  "playground_swing"
]

main_objs = [
  "little_girl",
  "medium_dog"
]

Step 3: Generate a general description of the layout.
-> layout_plan_1 = {
  "title": "Set Up Garden Base",
  "asset_list": ["garden_grassy_field"],
  "description": "Place the large, lush, green grassy field as the base of the garden. Ensure it covers the entire scene to provide a natural, grassy environment."
}

layout_plan_2 = {
  "title": "Add Playground Equipment",
  "asset_list": ["playground_slide", "playground_swing"],
  "description": "Position the brightly colored children's slide and the single swing within the garden area. Place the slide to one side, with the swing slightly in the background, creating a playful and engaging playground atmosphere."
}

layout_plan_3 = {
  "title": "Place Main Characters",
  "asset_list": ["little_girl", "medium_dog"],
  "description": "Place the young girl, dressed in a bright yellow sweater, colorful dress, and high heels, interacting with the medium-sized dog. Position the girl near the playground equipment, playing with the dog to show a dynamic and joyful interaction."
}

Step 4: Generate the initial location coordinates of the objects and the constraints between them.
-> initial_position = {
  "garden_grassy_field": (0, 0, 0),
  "playground_slide": (5, 0, 0),
  "playground_swing": (8, 0, 0),
  "little_girl": (6, 1, 0),
  "medium_dog": (6, -1, 0)
}

constraints = [
  ("proximity_score", {"object1": "playground_slide", "object2": "playground_swing"}),
  ("proximity_score", {"object1": "little_girl", "object2": "medium_dog"}),
  ("direction_score", {"object1": "little_girl", "object2": "medium_dog"}),
  ("alignment_score", {"assets": ["playground_slide", "playground_swing"], "axis": "x"}),
  ("perpendicularity_score", {"object1": "playground_slide", "object2": "playground_swing"}),
  ("parallelism_score", {"assets": ["playground_slide", "playground_swing"]}),
  ("rotation_uniformity_score", {"objects": ["playground_slide", "playground_swing"], "center": (0, 0, 0)})
]

Step 5: Generate the motion script of the main objects.
-> trajectory = {
    "total_frames": 240,
    "motions": [
        {"frame_start": 1, "frame_end": 60, "trajectory": [(6, 1, 0), (7, 1, 0), (8, 1, 0)], "object": "little_girl"},
        {"frame_start": 1, "frame_end": 60, "trajectory": [(6, -1, 0), (7, -1, 0), (8, -1, 0)], "object": "medium_dog"},
        {"frame_start": 61, "frame_end": 120, "trajectory": [(8, 1, 0), (9, 2, 0), (10, 3, 0)], "object": "little_girl"},
        {"frame_start": 61, "frame_end": 120, "trajectory": [(8, -1, 0), (9, -2, 0), (10, -3, 0)], "object": "medium_dog"},
        {"frame_start": 121, "frame_end": 180, "trajectory": [(10, 3, 0), (11, 3, 0), (12, 3, 0)], "object": "little_girl"},
        {"frame_start": 121, "frame_end": 180, "trajectory": [(10, -3, 0), (11, -3, 0), (12, -3, 0)], "object": "medium_dog"},
        {"frame_start": 181, "frame_end": 240, "trajectory": [(12, 3, 0), (12, 2, 0), (12, 1, 0)], "object": "little_girl"},
        {"frame_start": 181, "frame_end": 240, "trajectory": [(12, -3, 0), (12, -2, 0), (12, -1, 0)], "object": "medium_dog"}
    ]
}

User's requirements: I want the girl and her dog running around the playground_swing

After determining your answer, structure them in this format:
change_step = [step1, step2, ...]

Avoid using normal text; format your response strictly as specified above
"""

    respone = """
change_step = [5]
"""

    exec(respone)
    
    prompt = """
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your mission is to script the animation sequences for objects based on natural language descriptions, the list of objects and their initial positions

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

Natural language description: 
To create a 3D scene for the text "A girl plays with her dog in the garden," we can use the following details:

    The Setting:
        The scene takes place in a garden, which includes a grassy field.
        Consider adding elements to make it look like a children's playground.

    The Girl:
        She is young, possibly a little girl or a pre-teen.
        She is wearing a yellow sweater, a dress, and a pair of high heels.
        Her clothing is mostly bright and colorful.

    The Dog:
        The dog is medium-sized.
        The dog can be portrayed standing or interacting with the girl.

    Interaction:
        The girl is playing with the dog, so they should be interacting with each other in a playful manner.

Based on these details, you can imagine a vibrant and joyful scene. 
The girl, dressed in her colorful outfit, is happily playing with her medium-sized dog in a lush, green garden. 
The garden might include elements of a playground to enhance the playful atmosphere.

Objects and their constraints, initial position:
initial_position = {
  "garden_grassy_field": (0, 0, 0),
  "playground_slide": (5, 0, 0),
  "playground_swing": (8, 0, 0),
  "little_girl": (6, 1, 0),
  "medium_dog": (6, -1, 0)
}

your previous answer: 
trajectory = {
    "total_frames": 240,
    "motions": [
        {"frame_start": 1, "frame_end": 60, "trajectory": [(6, 1, 0), (7, 1, 0), (8, 1, 0)], "object": "little_girl"},
        {"frame_start": 1, "frame_end": 60, "trajectory": [(6, -1, 0), (7, -1, 0), (8, -1, 0)], "object": "medium_dog"},
        {"frame_start": 61, "frame_end": 120, "trajectory": [(8, 1, 0), (9, 2, 0), (10, 3, 0)], "object": "little_girl"},
        {"frame_start": 61, "frame_end": 120, "trajectory": [(8, -1, 0), (9, -2, 0), (10, -3, 0)], "object": "medium_dog"},
        {"frame_start": 121, "frame_end": 180, "trajectory": [(10, 3, 0), (11, 3, 0), (12, 3, 0)], "object": "little_girl"},
        {"frame_start": 121, "frame_end": 180, "trajectory": [(10, -3, 0), (11, -3, 0), (12, -3, 0)], "object": "medium_dog"},
        {"frame_start": 181, "frame_end": 240, "trajectory": [(12, 3, 0), (12, 2, 0), (12, 1, 0)], "object": "little_girl"},
        {"frame_start": 181, "frame_end": 240, "trajectory": [(12, -3, 0), (12, -2, 0), (12, -1, 0)], "object": "medium_dog"}
    ]
}

Your answer should be formatted as a dictionary with two main keys: total_frames and motions, where total_frames represents the total number of frames in the video, formatted as an integer, and motions is a list of motions that will occur in the video, where each element contains fields including start_frame, end_frame, and list the coordinates of the points through which the path will pass to later perform interpolation to create a trajectory."        

After determining your answer, structure them in this format:
trajectory = {{
    "total_frames": total_frame,
    "motions": [
        {{"frame_start": frame_start, "frame_end": frame_end, "trajectory": [cordinate1, cordinate2, ...], "object": object}}, 
        ...
            ]
}}
Avoid using normal text; format your response strictly as specified above.
"""
