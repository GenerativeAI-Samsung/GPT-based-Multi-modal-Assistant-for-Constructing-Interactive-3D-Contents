import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import TextStreamer
from peft import PeftModel

import asyncio
import random
import g4f

from PIL import Image

list_object_avaible = [
	{"name": "Cat", "action": ["jump", "run", "sit", "walk"]},
	{"name": "Gull", "action": ["fly"]},
	{"name": "Duck", "action": ["swim", "jump", "run", "walk"]},
    {"name": "Human", "action": ["swim", "jump", "run", "walk", "stand"]},
    {"name": "Wolf", "action": ["swim", "jump", "run", "walk", "stand"]},
    {"name": "Deer", "action": ["jump", "run", "walk", "stand"]},
]

list_object_avaible_name =[obj["name"] for obj in list_object_avaible]

list_trajectory = [
        "zicZac(initialPosition: Euclidean coordinates (x, y, z), finalPosition: Euclidean coordinates (x, y, z))",
    	"standStill()",
    	"runStraight(initialPosition: Euclidean coordinates (x, y, z), finalPosition: Euclidean coordinates (x, y, z))",   
    	"elip(initialPosition: Euclidean coordinates (x, y, z), center: Euclidean coordinates, majorAxisLength: float, minorAxisLength: float, rotationAngle: Euler angles (roll, pitch, yaw))", 
    	"jump(initialPosition: Euclidean coordinates (x, y, z), apexPosition: Euclidean coordinates (x, y, z), finalPosition: Euclidean coordinates (x, y, z))"
]

list_object_environement = [
    {"name": "Tree"},
	{"name": "Flower"},
	{"name": "Rock"},
	{"name": "Bush"},
    {"name": "River"},
]

async def test_generate(prompt):
    async def process_api_request(request, index):
        while True:
            try:
                await asyncio.sleep(random.randint(10, 20))
                print(f"Started API request of index: {index}.")
                response = await g4f.ChatCompletion.create_async(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": request}],
                )
                if len(response) == 0:
                    continue
                print(response[0])
                print(f"Completed API request of index: {index}")
                return response
            except Exception as e:
                print(f"Request of index {index} - Error: {str(e)}")
                await asyncio.sleep(10)    
    tasks = []
    for index, request in enumerate(prompt):
        tasks.append(process_api_request(request, index))
    return await asyncio.gather(*tasks, return_exceptions=True)

class TestUserInteractModel():
    def __init__(self):
        pass

    def generate(self, batch):
        respone = asyncio.run(test_generate(batch))
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = asyncio.run(test_generate(batch))
        return respone

class TestScenePlanningModel():
    def __init__(self):
        pass
    
    def step1_preprocess_data(self, batch):
        processed_batch = []

        step1_answer_format = """
object_list = [
{"name": obj1},
{"name": obj2}
...
]

Each asset is described with a concise name (x), but only include the specific objects mentioned. Avoid including general scene elements (e.g., sky, ground, trajectories). 
If an object appears multiple times in a scene, you can differentiate each instance by naming them sequentially, like "Cat1," "Cat2," "Cat3," and so on.    
"""

        for sample in batch:
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your task is to identify and list the main assets that are explicitly mentioned and are essential objects in the description from the list of object available below.

Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

List of object available:
{list_object_avaible_name}

Natural language description: "{sample}"    
    
After listing the assets, structure them in this format:
{step1_answer_format}

Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step1_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch
    
    def step1_preprocess_data_version_modify(self, batch, feedback, previous_answers):
        processed_batch = []

        step1_answer_format = """
object_list = [
{"name": obj1},
{"name": obj2}
...
]

Each asset is described with a concise name (x), but only include the specific objects mentioned. Avoid including general scene elements (e.g., sky, ground, trajectories). 
If an object appears multiple times in a scene, you can differentiate each instance by naming them sequentially, like "Cat1," "Cat2," "Cat3," and so on.        
"""

        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your task is to identify and list the main assets that are explicitly mentioned and are essential objects in the description from the list of object available below.

Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

List of object available:
{list_object_avaible_name}

Natural language description: "{sample}"    

User Feedback: {fb}

Your previous answer: 
{previous_aws}

After listing the assets, structure them in this format:
{step1_answer_format}
Some information might conflict. Howerver, you should always priority what user said in User Feedback.
Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step1_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch
    
    def step1_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('object_list = [' in temp1):
                temp2 = temp1.split('object_list = [')[1]
                temp3 = temp2.split(']')[0]
                temp = 'object_list = [' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("object_list = []")
                print(f"respone: object_list = []")
        return cropped_respone_batch

    def step1_generate(self, batch, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step1_preprocess_data(batch=batch)
        elif (mode == "modify"):
            processed_batch = self.step1_preprocess_data_version_modify(batch=batch, feedback=feedback, previous_answers=previous_answers)
        
        respone = asyncio.run(test_generate(processed_batch))
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = asyncio.run(test_generate(processed_batch))
        
        # Crop output from response
        respone = self.step1_crop_respone(respone)
        
        return respone

    def step2_preprocess_data(self, batch, objects_list):
        processed_batch = []

        step2_answer_format = """
init_pos_ori = [
    {"name": obj1, "pos": (x1, y1, z1), "ori": (roll1, pitch1, yaw1)},
    {"name": obj2, "pos": (x2, y2, z2), "ori": (roll2, pitch2, yaw2)},
    ...
]

Ensure the coordinates and orientations reflect the described actions and positions of objects.
"""

        for sample in batch:
            processed_sample = f"""
You are an assistant for developing Blender scripts to create scenes based on natural descriptions.

Your task is to analyze the description and suggest initial positions and orientations for the objects listed.
Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

Objects list:
{objects_list}

Natural language description: {sample}

Requirements:
Provide a list called init_pos_ori, consisting of dictionaries for each object with:
- Object name
- Initial position as Euclidean coordinates (x, y, z)
- Initial orientation as Euler angles (roll, pitch, yaw)

After determining your answer, structure them in this format:
{step2_answer_format}
Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step2_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch
    
    def step2_preprocess_data_version_modify(self, batch, objects_list, feedback, previous_answers):
        processed_batch = []

        step2_answer_format = """
init_pos_ori = [
    {"name": obj1, "pos": (x1, y1, z1), "ori": (roll1, pitch1, yaw1)},
    {"name": obj2, "pos": (x2, y2, z2), "ori": (roll2, pitch2, yaw2)},
    ...
]

Ensure the coordinates and orientations reflect the described actions and positions of objects.
"""
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing Blender scripts to create scenes based on natural descriptions.

Your task is to analyze the description and suggest initial positions and orientations for the objects listed.
Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

Objects list:
{objects_list}

Natural language description: {sample}

User Feedback: {fb}

Your previous answer: 
{previous_aws}

After listing the assets, structure them in this format:
{step2_answer_format}

Some information might conflict. Howerver, you should always priority what user said in User Feedback.
Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step2_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step2_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('init_pos_ori = [' in temp1):
                temp2 = temp1.split('init_pos_ori = [')[1]
                temp3 = temp2.split(']')[0]
                temp = 'init_pos_ori = [' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("init_pos_ori = []")
                print(f"respone: init_pos_ori = []")
        return cropped_respone_batch

    def step2_generate(self, batch, objects_list, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step2_preprocess_data(batch, objects_list)
        elif (mode == "modify"):
            processed_batch = self.step2_preprocess_data_version_modify(batch, objects_list,feedback, previous_answers)

        respone = asyncio.run(test_generate(processed_batch))
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = asyncio.run(test_generate(processed_batch))
        
        # Crop output from response
        respone = self.step2_crop_respone(respone)
        
        return respone

    def step3_preprocess_data(self, batch, objects_list, init_pos_ori):
        processed_batch = []

        step3_answer_format = """
trajectory = {
        "total_frames": total_frame,
        "motions": [
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "trajectory": (trajectory_name, {"param1": param1, "param2": param2, ...})
                "object": object,
                "object_action": action,
            },
            ...
        ]
    }
        - total_frames: Total duration of the animation in frames (an integer).
        - motions: A list of movements occurring in the animation, where each movement includes:
            - frame_start: The frame at which the motion begins.
            - frame_end: The frame at which the motion ends.
            - trajectory: A tuple containing the name of the trajectory function and its parameters defining the path the object will follow.
            - action: The specific action performed by the object during the motion.
"""

        for sample in batch:
            processed_sample = f"""
You are tasked with developing Blender scripts to create animation scenes based on natural language descriptions. Your goal is to script the animation sequences for the objects specified in the object list, using the provided natural language description and the initial positions and orientaions of object and the actions they could do.
The trajectory of each object should be taken from the list of trajectory functions provided below.
Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

Instructions:

    - Natural Language Description:
        {sample}

    - Object List:
        {objects_list}

    - the initial positions and orientaions of object:
        {init_pos_ori}

    - Trajectory Functions:
        {list_trajectory}

    - Action:
        {list_object_avaible}

Natural language description: {sample}

After processing the information, present your answer in the following format:
{step3_answer_format}

Note: Avoid using normal text in your response; strictly adhere to the specified format.    
"""
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step3_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        return processed_batch
    
    def step3_preprocess_data_version_modify(self, batch, objects_list, init_pos_ori, feedback, previous_answers):
        processed_batch = []

        step3_answer_format = """
trajectory = {
        "total_frames": total_frame,
        "motions": [
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "trajectory": (trajectory_name, {"param1": param1, "param2": param2, ...})
                "object": object,
                "object_action": action,
            },
            ...
        ]
    }
        - total_frames: Total duration of the animation in frames (an integer).
        - motions: A list of movements occurring in the animation, where each movement includes:
            - frame_start: The frame at which the motion begins.
            - frame_end: The frame at which the motion ends.
            - trajectory: A tuple containing the name of the trajectory function and its parameters defining the path the object will follow.
            - action: The specific action performed by the object during the motion.
"""

        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are tasked with developing Blender scripts to create animation scenes based on natural language descriptions. Your goal is to script the animation sequences for the objects specified in the object list, using the provided natural language description and the initial positions and orientaions of object and the actions they could do.
The trajectory of each object should be taken from the list of trajectory functions provided below.
Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

Instructions:

    - Natural Language Description:
        {sample}

    - Object List:
        {objects_list}

    - the initial positions and orientaions of object:
        {init_pos_ori}

    - Trajectory Functions:
        {list_trajectory}

    - Action:
        {list_object_avaible}

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.        

User Feedback: {fb}

Your previous answer: 
{previous_aws}        

Natural language description: {sample}

After processing the information, present your answer in the following format:
{step3_answer_format}

Some information might conflict. Howerver, you should always priority what user said in User Feedback.
Note: Avoid using normal text in your response; strictly adhere to the specified format.    
"""
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step3_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step3_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('trajectory =' in temp1):
                temp2 = temp1.split('trajectory =', 1)[1]
                temp3 = temp2.rsplit('}', 1)[0]
                temp = 'trajectory =' + temp3 + '}'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("trajectory = {}")
                print("respone: trajectory = {}")
        return cropped_respone_batch

    def step3_generate(self, batch, objects_list, init_pos_ori, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step3_preprocess_data(batch, objects_list, init_pos_ori)
        elif (mode == "modify"):
            processed_batch = self.step3_preprocess_data_version_modify(batch, objects_list, init_pos_ori, feedback, previous_answers)

        respone = asyncio.run(test_generate(processed_batch))
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = asyncio.run(test_generate(processed_batch))
        
        # Crop output from response
        respone = self.step3_crop_respone(respone)
        return respone

    def step4_preprocess_data(self, batch, main_object):
        processed_batch = []

        step4_answer_format = """
object_list = [
    {"name": obj1},
    {"name": obj2},
    ...
]  
"""
        
        for sample in batch:
            processed_sample = f"""
You are an assistant for developing Blender scripts to create scenes for various animation projects from natural language descriptions.

Your task is to identify and list assets to construct the environment where the scene takes place from the list of assets available below, based on the given description.
Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

List of object available:
    {list_object_environement}
    
    
Natural language description: {sample}

After identifying the assets needed to build the environment (excluding the main character objects), structure them in this format:
{step4_answer_format}

Main character objects to exclude:
    {main_object}

After processing the information, present your answer in the following format:
{step4_answer_format}
Include only specific environmental objects essential for making the scene functional. Avoid adding general scene elements (e.g., 'sky,' 'ground,' or 'trajectories').
If an object appears multiple times in a scene, you can differentiate each instance by naming them sequentially, like "Tree1," "Tree2," "Flower" and so on.
Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step4_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step4_preprocess_data_version_modify(self, batch, main_object, feedback, previous_answers):
        processed_batch = []

        step4_answer_format = """
object_list = [
    {"name": obj1},
    {"name": obj2},
    ...
]    
"""
        
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing Blender scripts to create scenes for various animation projects from natural language descriptions.

Your task is to identify and list assets to construct the environment where the scene takes place from the list of assets available below, based on the given description.
Your response should strictly adhere to the user's requirements and any previous answer provided (if applicable).

List of object available:
    {list_object_environement}

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.        

User Feedback: {fb}

Your previous answer: 
{previous_aws}            

Natural language description: {sample}

After identifying the assets needed to build the environment (excluding the main character objects), structure them in this format:
{step4_answer_format}

Main character objects to exclude:
    {main_object}

After processing the information, present your answer in the following format:
{step4_answer_format}    

Include only specific environmental objects essential for making the scene functional. Avoid adding general scene elements (e.g., 'sky,' 'ground,' or 'trajectories').
If an object appears multiple times in a scene, you can differentiate each instance by naming them sequentially, like "Tree1," "Tree2," "Flower" and so on.
Some information might conflict. Howerver, you should always priority what user said in user Feedback.
Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step4_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step4_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('object_list =' in temp1):
                temp2 = temp1.split('object_list =', 1)[1]
                temp3 = temp2.rsplit(']', 1)[0]
                temp = 'object_list =' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("object_list = {}")
                print("respone: object_list = {}")
        return cropped_respone_batch

    def step4_generate(self, batch, main_object, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step4_preprocess_data(batch, main_object)
        elif (mode == "modify"):
            processed_batch = self.step4_preprocess_data_version_modify(batch, main_object, feedback, previous_answers)
        
        respone = asyncio.run(test_generate(processed_batch))
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = asyncio.run(test_generate(processed_batch))
        
        # Crop output from response
        respone = self.step4_crop_respone(respone)
        return respone
    
    def step5_preprocess_data(self, batch, object_list):
        processed_batch = []

        step5_answer_format = """
initial_position = {objectName1: (x1, y1, z1), objectName2: (x2, y2, z2), ...}
constraints = [(nameOfConstraint1, ("param1": "object1", ...)), ...]
    
The answer should include 2 lists, initial_position and constraints, where initial_positions is a dictionary with keys as object names and values as their initial positions, and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.
"""
        
        for sample in batch:
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list to create a suitable enviroment for scene described in the natural descriptions happen.
Please think step by step.

Objects list:
{object_list}

Natural language description: {sample}

Constraints: 
proximity_score(object1: Layout, object2: Layout): A constraint enforcing the closeness of two objects, e.g., a chair near a table.
direction_score(object1: Layout, object2: Layout): The angle of one object is targeting at the other.
alignment_score(assets: List[Layout], axis: str): Ensuring objects align along a common axis (x, y, z), e.g., paintings aligned vertically on a wall.
symmetry_score(assets: List[Layout], axis: str): Mirroring objects along an axis (x, y, z), e.g., symmetrical placement of lamps on either side of a bed.
parallelism_score(assets: List[Layout]): Objects parallel to each other, suggesting direction, e.g., parallel rows of seats in a theater.
perpendicularity_score(object1: Layout, object2: Layout): Objects intersecting at a right angle, e.g., a bookshelf perpendicular to a desk.
rotation_uniformity_score(objects: List[Layout], center: Tuple[float, float, float]): a list of objects rotate a cirtain point, e.g., rotating chairs around a meeting table.

After determining initial_position and constraints, structure them in this format:
{step5_answer_format}

Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step5_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        return processed_batch
    
    def step5_preprocess_data_version_modify(self, batch, object_list, feedback, previous_answers):
        processed_batch = []

        step5_answer_format = """
initial_position = {objectName1: (x1, y1, z1), objectName2: (x2, y2, z2), ...}
constraints = [(nameOfConstraint1, ("param1": "object1", ...)), ...]
    
The answer should include 2 lists, initial_position and constraints, where initial_positions is a dictionary with keys as object names and values as their initial positions, and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.
"""
        
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list to create a suitable enviroment for scene described in the natural descriptions happen.
Please think step by step.

Objects list:
{object_list}

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

User Feedback: {fb}

Your previous answer: 
{previous_aws}

Natural language description: {sample}

Constraints: 
proximity_score(object1: Layout, object2: Layout): A constraint enforcing the closeness of two objects, e.g., a chair near a table.
direction_score(object1: Layout, object2: Layout): The angle of one object is targeting at the other.
alignment_score(assets: List[Layout], axis: str): Ensuring objects align along a common axis (x, y, z), e.g., paintings aligned vertically on a wall.
symmetry_score(assets: List[Layout], axis: str): Mirroring objects along an axis (x, y, z), e.g., symmetrical placement of lamps on either side of a bed.
parallelism_score(assets: List[Layout]): Objects parallel to each other, suggesting direction, e.g., parallel rows of seats in a theater.
perpendicularity_score(object1: Layout, object2: Layout): Objects intersecting at a right angle, e.g., a bookshelf perpendicular to a desk.
rotation_uniformity_score(objects: List[Layout], center: Tuple[float, float, float]): a list of objects rotate a cirtain point, e.g., rotating chairs around a meeting table.

After determining initial_position and constraints, structure them in this format:
{step5_answer_format}

Some information might conflict. Howerver, you should always priority what user said in user Feedback.
Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step5_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step5_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('initial_position' in temp1):
                temp2 = temp1.split('initial_position', 1)[1]
                temp3 = temp2.rsplit(']', 1)[0]
                temp = 'initial_position' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("initial_position = []\nconstraints=[]")
                print("respone: initial_position = []\nconstraints=[]")
        return cropped_respone_batch
    
    def step5_generate(self, batch, object_list, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step5_preprocess_data(batch=batch,object_list=object_list)
        elif (mode == "modify"):
            processed_batch = self.step5_preprocess_data_version_modify(batch=batch, 
                                                                object_list=object_list,
                                                                feedback=feedback,
                                                                previous_answers=previous_answers)
        
        respone = asyncio.run(test_generate(processed_batch))
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = asyncio.run(test_generate(processed_batch))

        # Crop output from response
        respone = self.step5_crop_respone(respone)
        return respone
    
    def modify_preprocess_data(self, 
                              batch,
                              step1_respone, 
                              step2_respone, 
                              step3_respone,
                              step4_respone,
                              step5_respone):
        processed_batch = []

        modify_answer_format = """
change_step = [step1, step2, ...]
"""
        
        for sample in batch:
            processed_sample = f"""
Your task is to determine which steps in the following 3D scene construction process need to be adjusted to meet the user's requirements:

Step 1: Identify main objects in the scene
{step1_respone}

Step 2: Identify main objects's location and orientation
{step2_respone}

Step 3: Generate motion and trajectory for main objects 
{step3_respone}

Step 4: Identify environment objects
{step4_respone}

Step 5: Generate the initial location coordinates of the environment objects and the constraints between them.
{step5_respone}

User's requirements: {sample}

After determining your answer, structure them in this format:
{modify_answer_format}

Avoid using normal text; format your response strictly as specified above
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {modify_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def modify_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('change_step' in temp1):
                temp2 = temp1.split('change_step', 1)[1]
                temp3 = temp2.rsplit(']', 1)[0]
                temp = 'change_step' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("change_step = []")
                print("respone: change_step = []")
        return cropped_respone_batch

    def modified_user_request(self, 
                              batch, 
                              step1_respone, 
                              step2_respone, 
                              step3_respone,
                              step4_respone,
                              step5_respone):
        # Prompt for input
        processed_batch = self.modify_preprocess_data(batch=batch, 
                                                      step1_respone=step1_respone, 
                                                     step2_respone=step2_respone,
                                                     step3_respone=step3_respone,
                                                     step4_respone=step4_respone,
                                                     step5_respone=step5_respone)

        respone = asyncio.run(test_generate(processed_batch))
        while ("Unusual activity" in str(respone[0])) or ("Request ended with status code 404" in str(respone[0])):
            respone = asyncio.run(test_generate(processed_batch))

        # Crop output from response
        respone = self.modify_crop_respone(respone)
        
        return respone

class VisionLangugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", 
                                               torch_dtype=torch.float16, 
                                               trust_remote_code=True).to("cuda")
        self.processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", 
                                                       trust_remote_code=True)
    
    def process(self, image_path=None, query=None):
        raw_image = Image.open(image_path)

        prompt = f"""<|im_start|>user
                    <image>
                    {query}<|im_end|>
                    <|im_start|>assistant
                """
        
        with torch.inference_mode():
            inputs = self.processor(prompt, 
                                    [raw_image], 
                                    self.model, 
                                    max_crops=100, 
                                    num_tokens=728)
            
        streamer = TextStreamer(self.processor.tokenizer)
        with torch.inference_mode():
            output = self.model.generate(**inputs, 
                                         max_new_tokens=200, 
                                         do_sample=True, 
                                         use_cache=False, 
                                         top_p=0.9, 
                                         temperature=1.2, 
                                         eos_token_id=self.processor.tokenizer.eos_token_id, 
                                         streamer=streamer)
        return self.processor.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", "")

class UserInteractModel(nn.Module):
    def __init__(self, MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                                       model_max_length=1536)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    def generate(self, batch):
        # Tokenize the input prompt
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True)

        # Move tensors to the appropriate device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generating
        outputs = self.model.generate(**inputs, max_length=1024, output_hidden_states=True, return_dict_in_generate=True)

        # Decode the generated tokens back to text
        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        return respone

class ScenePlanningModel(nn.Module):
    def __init__(self, MODEL_ID, adapter_layers):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                                       model_max_length=1536,
                                                       padding_side="right",
                                                       use_fast=False)
        self.base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        self.smart_tokenizer_and_embedding()

        self.step1_layer = PeftModel.from_pretrained(self.base_model,
                                                    adapter_layers[0],
                                                    is_trainable=False)

        self.step2_layer = PeftModel.from_pretrained(self.base_model,
                                                    adapter_layers[1],
                                                    is_trainable=False)

        self.step3_layer = PeftModel.from_pretrained(self.base_model,
                                                    adapter_layers[2],
                                                    is_trainable=False)

        self.step4_layer = PeftModel.from_pretrained(self.base_model,
                                                    adapter_layers[3],
                                                    is_trainable=False)
        
        self.step5_layer = PeftModel.from_pretrained(self.base_model,
                                                    adapter_layers[4],
                                                    is_trainable=False)

    def add_special_tokens(self):
        default_pad_token = "[PAD]"
        default_eos_token = "</s>"
        default_bos_token = "<s>"
        default_unk_token = "<unk>"

        # Adding special token to tokenizer
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = default_pad_token
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = default_eos_token
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = default_bos_token
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = default_unk_token
    
    
    def smart_tokenizer_and_embedding(self):
        num_new_tokens = self.add_special_tokens()
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.base_model.get_input_embeddings().weight.data
            output_embeddings = self.base_model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def step1_preprocess_data(self, batch):
        processed_batch = []

        step1_answer_format = """
object_list = [
{"name": x1, "description": y1},
{"name": x2, "description": y2},
{"name": x3, "description": y3},
...
]
Each asset is described with a concise name (x) and a detailed visual description (y).
Asset should be about the environment, main characters, animals, sounds, lighting, camera angles and layout
    """
        for sample in batch:
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets). 

Natural language description: "{sample}"    
    
After listing the assets, structure them in this format:
{step1_answer_format}

Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step1_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch
    
    def step1_preprocess_data_version_modify(self, batch, feedback, previous_answers):
        processed_batch = []

        step1_answer_format = """
object_list = [
{"name": x1, "description": y1},
{"name": x2, "description": y2},
{"name": x3, "description": y3},
...
]
Each asset is described with a concise name (x) and a detailed visual description (y).
Asset should be about the environment, main characters, animals, sounds, lighting, camera angles and layout
    """
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets). 

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

Natural language description: "{sample}"    

User Feedback: {fb}

Your previous answer: 
{previous_aws}

After listing the assets, structure them in this format:
{step1_answer_format}

Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step1_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch
    

    def step1_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('object_list = [' in temp1):
                temp2 = temp1.split('object_list = [')[1]
                temp3 = temp2.split(']')[0]
                temp = 'object_list = [' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("object_list = []")
                print(f"respone: object_list = []")
        return cropped_respone_batch

    def step1_generate(self, batch, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step1_preprocess_data(batch=batch)
        elif (mode == "modify"):
            processed_batch = self.step1_preprocess_data_version_modify(batch=batch, feedback=feedback, previous_answers=previous_answers)

        # Tokenize the input prompt
        inputs = self.tokenizer(processed_batch, return_tensors="pt", padding=True)

        # Move tensors to the appropriate device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generating
        outputs = self.step1_layer.generate(**inputs, max_length=1536, output_hidden_states=True, return_dict_in_generate=True)

        # Decode the generated tokens back to text
        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        
        # Crop output from response
        respone = self.step1_crop_respone(respone)
        
        return respone

    def step2_preprocess_data(self, batch, objects_list):
        processed_batch = []

        step2_answer_format = """
object_classified_list = [{"name": "base_environment", "objects": (obj1, obj2, ...)},
                        {"name": "main_characters_and_creatures", "objects": (obj8, obj9, ...)},
                        {"name": "illumination", "objects": (obj15, obj16, ...)},
                        {"name": "audio", "objects": (obj23, obj24, obj25)}
                        {"name": "camera_view", "objects": (obj21, obj22, ...)}]
"""
        for sample in batch:
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural descriptions.
Your job is to classify the objects from the objects list below and natural descriptions into four groups: 
1. Base environment: Objects that form the background, scenery, or surroundings.
2. Main characters and creatures: The primary characters and creatures featured in the animation.
3. Illumination: Objects or elements responsible for providing or adjusting light in the scene.
4. Audio: Objects or systems that generate or manipulate sound.
5. Camera view: Objects or elements involved in camera positioning, movement, or focus.
6. Layout Discription: elements that discripting about the layout

Objects list:
{objects_list}

Natural language description: {sample}

After listing the assets, structure them in this format:
{step2_answer_format}

Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step2_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step2_preprocess_data_version_modify(self, batch, objects_list, feedback, previous_answers):
        processed_batch = []

        step2_answer_format = """
object_classified_list = [{"name": "base_environment", "objects": (obj1, obj2, ...)},
                        {"name": "main_characters_and_creatures", "objects": (obj8, obj9, ...)},
                        {"name": "illumination", "objects": (obj15, obj16, ...)},
                        {"name": "audio", "objects": (obj23, obj24, obj25)}
                        {"name": "camera_view", "objects": (obj21, obj22, ...)}]
"""
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural descriptions.
Your job is to classify the objects from the objects list below and natural descriptions into four groups: 
1. Base environment: Objects that form the background, scenery, or surroundings.
2. Main characters and creatures: The primary characters and creatures featured in the animation.
3. Illumination: Objects or elements responsible for providing or adjusting light in the scene.
4. Audio: Objects or systems that generate or manipulate sound.
5. Camera view: Objects or elements involved in camera positioning, movement, or focus.

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

Objects list:
{objects_list}

Natural language description: {sample}

User Feedback: {fb}

Your previous answer: 
{previous_aws}

After listing the assets, structure them in this format:
{step2_answer_format}

Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step2_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step2_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('object_classified_list = [' in temp1):
                temp2 = temp1.split('object_classified_list = [')[1]
                temp3 = temp2.split(']')[0]
                temp = 'object_classified_list = [' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("object_classified_list = []")
                print(f"respone: object_classified_list = []")
        return cropped_respone_batch

    def step2_generate(self, batch, objects_list, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step2_preprocess_data(batch, objects_list)
        elif (mode == "modify"):
            processed_batch = self.step2_preprocess_data_version_modify(batch, objects_list,feedback, previous_answers)

        # Tokenize the input prompt
        inputs = self.tokenizer(processed_batch, return_tensors="pt", padding=True)

        # Move tensors to the appropriate device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generating
        outputs = self.step2_layer.generate(**inputs, max_length=1536, output_hidden_states=True, return_dict_in_generate=True)

        # Decode the generated tokens back to text
        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        
        # Crop output from response
        respone = self.step2_crop_respone(respone)
        
        return respone

    def step3_preprocess_data(self, batch, objects_list, object_classified_list):
        processed_batch = []

        step3_answer_format = """
For each step, structure your output as:
    layout_plan_i = {
            "title": title_i,
            "asset_list": [{"name": asset_name_1, "discription": asset_name_1_discription}, {"name": asset_name_2, "discription": asset_name_2_discription}, ...],
            "description": desc_i
    }

where title_i is the high-level name for this step, and desc is detailed visual text description of what it shall look like after layout. 
    """
        for sample in batch:
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to create a concrete plan to put them into the scene from the objects list below and natural descriptions.
Each layout_plan should include camera design, sound design and lighting
Please think step by step, and give me a multi-step plan to put assets into the scene.


Objects list:
{objects_list}

object list after classified:
{object_classified_list}

Natural language description: {sample}

After listing the assets, structure them in this format:
{step3_answer_format}

Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step3_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step3_preprocess_data_version_modify(self, batch, objects_list, object_classified_list, feedback, previous_answers):
        processed_batch = []

        step3_answer_format = """
For each step, structure your output as:
    layout_plan_i = {
            "title": title_i,
            "asset_list": [{"name": asset_name_1, "discription": asset_name_1_discription}, {"name": asset_name_2, "discription": asset_name_2_discription}, ...],
            "description": desc_i
    }

where title_i is the high-level name for this step, and desc is detailed visual text description of what it shall look like after layout. 
    """
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to create a concrete plan to put them into the scene from the objects list below and natural descriptions.
Please think step by step, and give me a multi-step plan to put assets into the scene.
Each layout_plan should include camera design, sound design and lighting

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

Objects list:
{objects_list}

object list after classified:
{object_classified_list}

Natural language description: {sample}

User Feedback: {fb}

Your previous answer: 
{previous_aws}

After listing the assets, structure them in this format:
{step3_answer_format}

Avoid using normal text; format your response strictly as specified above.
    """
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step3_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step3_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('layout_plan' in temp1):
                temp2 = temp1.split('layout_plan', 1)[1]
                temp3 = temp2.rsplit('}', 1)[0]
                temp = 'layout_plan' + temp3 + '}'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("layout_plan_1 = {}")
                print("respone: layout_plan_1 = {}")
        return cropped_respone_batch

    def step3_generate(self, batch, objects_list, object_classified_list, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step3_preprocess_data(batch, objects_list, object_classified_list)
        elif (mode == "modify"):
            processed_batch = self.step3_preprocess_data_version_modify(batch, objects_list, object_classified_list, feedback, previous_answers)

        # Tokenize the input prompt
        inputs = self.tokenizer(processed_batch, return_tensors="pt", padding=True)

        # Move tensors to the appropriate device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generating
        outputs = self.step3_layer.generate(**inputs, max_length=1536, output_hidden_states=True, return_dict_in_generate=True)

        # Decode the generated tokens back to text
        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        
        # Crop output from response
        respone = self.step3_crop_respone(respone)
        
        return respone

    def step4_preprocess_data(self, batch, base_environment, main_characters_and_creatures, layout_plan):
        processed_batch = []

        step4_answer_format = """
initial_position_and_orientation = [{"name": obj1, "position": (x1, y1, z1), "orientation": (roll1, pitch1, yaw1)},
                                    {"name": obj2, "position": (x2, y2, z2), "orientation": (roll2, pitch2, yaw2)},
                                    ...]
constraints = [(constraint1, {"param1": "object1", ...}), (constraint2, {"param2": "object2", ...}), ...]
    """
        
        for sample in batch:
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list and the layout plan.
Please think step by step.

Objects list:
{base_environment + main_characters_and_creatures}

Natural language description: {sample}

Constraints: 
proximity_score(object1: Layout, object2: Layout): A constraint enforcing the closeness of two objects, e.g., a chair near a table.
direction_score(object1: Layout, object2: Layout): The angle of one object is targeting at the other.
alignment_score(assets: List[Layout], axis: str): Ensuring objects align along a common axis (x, y, z), e.g., paintings aligned vertically on a wall.
symmetry_score(assets: List[Layout], axis: str): Mirroring objects along an axis (x, y, z), e.g., symmetrical placement of lamps on either side of a bed.
parallelism_score(assets: List[Layout]): Objects parallel to each other, suggesting direction, e.g., parallel rows of seats in a theater.
perpendicularity_score(object1: Layout, object2: Layout): Objects intersecting at a right angle, e.g., a bookshelf perpendicular to a desk.
rotation_uniformity_score(objects: List[Layout], center: Tuple[float, float, float]): a list of objects rotate a cirtain point, e.g., rotating chairs around a meeting table.

Layout plan:
{layout_plan}   

The answer should include 2 lists, initial_position_and_orientation and constraints, where initial_position_and_orientation is a list of dictionary with keys are object names, their initial positions (Euclidean coordinates) and their initial orientation (Euler angles), and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.

After determining initial_position_and_orientation and constraints, structure them in this format:
{step4_answer_format}

Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step4_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step4_preprocess_data_version_modify(self, batch, base_environment, main_characters_and_creatures, layout_plan, feedback, previous_answers):
        processed_batch = []

        step4_answer_format = """
initial_position_and_orientation = [{"name": obj1, "position": (x1, y1, z1), "orientation": (roll1, pitch1, yaw1)},
                                    {"name": obj2, "position": (x2, y2, z2), "orientation": (roll2, pitch2, yaw2)},
                                    ...]
constraints = [(constraint1, {"param1": "object1", ...}), (constraint2, {"param2": "object2", ...}), ...]
    """
        
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to suggest the initial position of objects and their constraints based on the objects list, the natural descriptions, the constraint list and the layout plan.
Please think step by step.

User has recently provided some feedback on your previous asnswer. Your task this time is to adjust the response to meet the user's feedback.

Objects list:
{base_environment + main_characters_and_creatures}

Natural language description: {sample}

Constraints: 
proximity_score(object1: Layout, object2: Layout): A constraint enforcing the closeness of two objects, e.g., a chair near a table.
direction_score(object1: Layout, object2: Layout): The angle of one object is targeting at the other.
alignment_score(assets: List[Layout], axis: str): Ensuring objects align along a common axis (x, y, z), e.g., paintings aligned vertically on a wall.
symmetry_score(assets: List[Layout], axis: str): Mirroring objects along an axis (x, y, z), e.g., symmetrical placement of lamps on either side of a bed.
parallelism_score(assets: List[Layout]): Objects parallel to each other, suggesting direction, e.g., parallel rows of seats in a theater.
perpendicularity_score(object1: Layout, object2: Layout): Objects intersecting at a right angle, e.g., a bookshelf perpendicular to a desk.
rotation_uniformity_score(objects: List[Layout], center: Tuple[float, float, float]): a list of objects rotate a cirtain point, e.g., rotating chairs around a meeting table.

Layout plan:
{layout_plan}   

User Feedback: {fb}

Your previous answer: 
{previous_aws}

The answer should include 2 lists, initial_position_and_orientation and constraints, where initial_position_and_orientation is a list of dictionary with keys are object names, their initial positions (Euclidean coordinates) and their initial orientation (Euler angles), and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.

After determining initial_position_and_orientation and constraints, structure them in this format:
{step4_answer_format}

Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step4_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step4_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('initial_position =' in temp1):
                temp2 = temp1.split('initial_position =', 1)[1]
                temp3 = temp2.rsplit(']', 1)[0]
                temp = 'initial_position =' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("initial_position = {}\nconstraints = []")
                print("respone: initial_position = {}\nconstraints = []")
        return cropped_respone_batch

    def step4_generate(self, batch, base_environment, main_characters_and_creatures, layout_plan, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step4_preprocess_data(batch, base_environment, main_characters_and_creatures, layout_plan)
        elif (mode == "modify"):
            processed_batch = self.step4_preprocess_data_version_modify(batch, base_environment, main_characters_and_creatures, layout_plan, feedback, previous_answers)
        
        # Tokenize the input prompt
        inputs = self.tokenizer(processed_batch, return_tensors="pt", padding=True)

        # Move tensors to the appropriate device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generating
        outputs = self.step4_layer.generate(**inputs, max_length=1536, output_hidden_states=True, return_dict_in_generate=True)

        # Decode the generated tokens back to text
        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        
        # Crop output from response
        respone = self.step4_crop_respone(respone)

        return respone
    
    def step5_preprocess_data(self, batch, main_characters_and_creatures, layout_plan, list_of_object, object_initial_position):
        processed_batch = []

        step5_answer_format = """
trajectory = {
    "total_frames": total_frame,
    "motions": [
        {
            "frame_start": frame_start,
            "frame_end": frame_end,
            "position_trajectory": [(x1, y1, z1), (x2, y2, z2), ...],  
            "orientation_trajectory": [(roll1, pitch1, yaw1), (roll2, pitch2, yaw2), ...],               
            "object": object,
            "object_action": action,
            "sound": sound
        },
        ...
    ]
}
Where total_frames represents the total duration of the video in frames, given as an integer. The motions field is a list of movements that occur in the video, where each motion is defined by the following elements:
- frame_start: The frame at which the motion begins.
- frame_end: The frame at which the motion ends.
- position_trajectory: A list of tuples representing Euclidean coordinates (x, y, z) that define the path the object will follow in 3D space. These points will later be used for interpolation to create a smooth trajectory.
- orientaion_trajectory: A list of tuples representing the object's orientation in Euclidean space, typically described by Euler angles (roll, pitch, yaw) for rotation, allowing smooth transitions and realistic rotation behavior.
- object: The name of the object being animated.
- action: The specific action the object performs during this motion.
- sound: The sound associated with the object during this motion, or None if no sound is involved.
"""
        
        for sample in batch:
            processed_sample = f"""
You are responsible for developing multiple Blender scripts to create animation scenes based on natural language descriptions. Your task is to script the animation sequences for the objects listed in main_characters_and_creatures, using the provided natural language descriptions, the scene layout plan, the list of objects, and their initial positions.    
please think step by step

main_characters_and_creatures list:
{main_characters_and_creatures}

Natural language description: {sample}

Scene layout plan:
{layout_plan}

List of objects:
{list_of_object}

Objects initial position:
{object_initial_position}

After determining your answer, structure them in this format:
{step5_answer_format}

Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step5_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step5_preprocess_data_version_modify(self, batch, main_characters_and_creatures, layout_plan, list_of_object, object_initial_position, feedback, previous_answers):
        processed_batch = []

        step5_answer_format = """
trajectory = {
    "total_frames": total_frame,
    "motions": [
        {
            "frame_start": frame_start,
            "frame_end": frame_end,
            "position_trajectory": [(x1, y1, z1), (x2, y2, z2), ...],  
            "orientation_trajectory": [(roll1, pitch1, yaw1), (roll2, pitch2, yaw2), ...],               
            "object": object,
            "object_action": action,
            "sound": sound
        },
        ...
    ]
}
Where total_frames represents the total duration of the video in frames, given as an integer. The motions field is a list of movements that occur in the video, where each motion is defined by the following elements:
- frame_start: The frame at which the motion begins.
- frame_end: The frame at which the motion ends.
- position_trajectory: A list of tuples representing Euclidean coordinates (x, y, z) that define the path the object will follow in 3D space. These points will later be used for interpolation to create a smooth trajectory.
- orientaion_trajectory: A list of tuples representing the object's orientation in Euclidean space, typically described by Euler angles (roll, pitch, yaw) for rotation, allowing smooth transitions and realistic rotation behavior.
- object: The name of the object being animated.
- action: The specific action the object performs during this motion.
- sound: The sound associated with the object during this motion, or None if no sound is involved.
"""
        
        for sample, fb, previous_aws in (batch, feedback, previous_answers):
            processed_sample = f"""
You are responsible for developing multiple Blender scripts to create animation scenes based on natural language descriptions. Your task is to script the animation sequences for the objects listed in main_characters_and_creatures, using the provided natural language descriptions, the scene layout plan, the list of objects, and their initial positions.    
please think step by step

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

main_characters_and_creatures list:
{main_characters_and_creatures}

Natural language description: {sample}

Scene layout plan:
{layout_plan}

List of objects:
{list_of_object}

Objects initial position:
{object_initial_position}

User Feedback: {fb}

Your previous answer: 
{previous_aws}

After determining your answer, structure them in this format:
{step5_answer_format}

Avoid using normal text; format your response strictly as specified above.
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step5_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def step5_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('trajectory' in temp1):
                temp2 = temp1.split('trajectory', 1)[1]
                temp3 = temp2.rsplit('}', 1)[0]
                temp = 'trajectory' + temp3 + '}'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("trajectory = []")
                print("respone: trajectory = []")
        return cropped_respone_batch
    
    def step5_generate(self, batch, main_characters_and_creatures, layout_plan, list_of_object, object_initial_position, mode, feedback=None, previous_answers=None):
        # Prompt for input
        if (mode == "new"):
            processed_batch = self.step5_preprocess_data(batch=batch, main_characters_and_creatures=main_characters_and_creatures, 
                                                     layout_plan=layout_plan,
                                                     list_of_object=list_of_object,
                                                     object_initial_position=object_initial_position)
        elif (mode == "modify"):
            processed_batch = self.step5_preprocess_data_version_modify(batch=batch, main_characters_and_creatures=main_characters_and_creatures, 
                                                     layout_plan=layout_plan,
                                                     list_of_object=list_of_object,
                                                     object_initial_position=object_initial_position,
                                                     feedback=feedback,
                                                     previous_answers=previous_answers)
            
        # Tokenize the input prompt
        inputs = self.tokenizer(processed_batch, return_tensors="pt", padding=True)

        # Move tensors to the appropriate device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generating
        outputs = self.step5_layer.generate(**inputs, max_length=1536, output_hidden_states=True, return_dict_in_generate=True)

        # Decode the generated tokens back to text
        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        
        # Crop output from response
        respone = self.step5_crop_respone(respone)

        return respone
    
    def modify_preprocess_data(self, 
                              batch,
                              step1_respone, 
                              step2_respone, 
                              step3_respone,
                              step4_respone,
                              step5_respone):
        processed_batch = []

        modify_answer_format = """
change_step = [step1, step2, ...]
"""
        
        for sample in batch:
            processed_sample = f"""
Your task is to determine which steps in the following 3D scene construction process need to be adjusted to meet the user's requirements:

Step 1: Identify the objects that will appear in the 3D scene.
{step1_respone}

Step 2: Classify the types of the identified objects.
{step2_respone}

Step 3: Generate a general description of the layout.
{step3_respone}

Step 4: Generate the initial location coordinates of the objects and the constraints between them.
{step4_respone}

Step 5: Generate the motion script of the main objects.
{step5_respone}

User's requirements: {sample}

After determining your answer, structure them in this format:
{modify_answer_format}

Avoid using normal text; format your response strictly as specified above
"""    
            processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {modify_answer_format}
    ------------------------------------------------------------------------
    """
            processed_sample += "\nRespone:"
            processed_batch.append(processed_sample)
        
        return processed_batch

    def modify_crop_respone(self, batch):
        cropped_respone_batch = []
        for respone in batch:
            temp1 = respone.split('\nRespone:', 1)[1]
            if ('change_step' in temp1):
                temp2 = temp1.split('change_step', 1)[1]
                temp3 = temp2.rsplit(']', 1)[0]
                temp = 'change_step' + temp3 + ']'
                print(f"respone: {temp}")
                cropped_respone_batch.append(temp)
            else:
                cropped_respone_batch.append("change_step = []")
                print("respone: change_step = []")
        return cropped_respone_batch


    def modified_user_request(self, 
                              batch, 
                              step1_respone, 
                              step2_respone, 
                              step3_respone,
                              step4_respone,
                              step5_respone):
        # Prompt for input
        processed_batch = self.modify_preprocess_data(batch=batch, 
                                                      step1_respone=step1_respone, 
                                                     step2_respone=step2_respone,
                                                     step3_respone=step3_respone,
                                                     step4_respone=step4_respone,
                                                     step5_respone=step5_respone)

        # Tokenize the input prompt
        inputs = self.tokenizer(processed_batch, return_tensors="pt", padding=True)

        # Move tensors to the appropriate device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generating
        outputs = self.base_model.generate(**inputs, max_length=1536, output_hidden_states=True, return_dict_in_generate=True)

        # Decode the generated tokens back to text
        respone = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        
        # Crop output from response
        respone = self.modify_crop_respone(respone)
        
        return respone        