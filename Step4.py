import torch
from utils import interact_with_lm, split_answer_from_respone

def step4(tokenizer, model, user_request, step1_respone, step3_respone):
    step4_answer_format = """
initial_position = {key: value, ...}
constraints = [(constraint1, ("param1": "object1", ...)), ...]
    
The answer should include 2 lists, initial_position and constraints, where initial_positions is a dictionary with keys as object names and values as their initial positions, and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.
"""
    step4_prompt = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list and the layout plan.
Please think step by step.

You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list and the layout plan.
Please think step by step.

Objects list:
{step1_respone}

Natural language description: {user_request}

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
{step3_respone}

After determining initial_position and constraints, structure them in this format:
{step4_answer_format}

Avoid using normal text; format your response strictly as specified above.
Respone: 
"""
    
    step4_response = interact_with_lm(tokenizer=tokenizer, model=model, prompt=step4_prompt)
    step4_response = split_answer_from_respone(respone=step4_response)
    
    return step4_answer_format, step4_prompt, step4_response
