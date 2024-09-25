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
        respone = test_generate(batch)
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

    def step1_generate(self, batch):
        # Prompt for input
        processed_batch = self.step1_preprocess_data(batch)

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

    def step2_generate(self, batch, objects_list):
        # Prompt for input
        processed_batch = self.step2_preprocess_data(batch, objects_list)

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
            "asset_list": [asset_name_1, asset_name_2],
            "description": desc_i
    }

where title_i is the high-level name for this step, and desc is detailed visual text description of what it shall look like after layout. 
    """
        for sample in batch:
            processed_sample = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to create a concrete plan to put them into the scene from the objects list below and natural descriptions.
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

    def step3_generate(self, batch, objects_list, object_classified_list):
        # Prompt for input
        processed_batch = self.step3_preprocess_data(batch, objects_list, object_classified_list)

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
initial_position_and_orientation = [{"name": obj1, "position": obj1_position, "orientation": obj1_orientation},
                                    {"name": obj2, "position": obj2_position, "orientation": obj2_orientation},
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

The answer should include 2 lists, initial_position_and_orientation and constraints, where initial_position_and_orientation is a list of dictionary with keys are object names, their initial positions and their initial orientation, and constraints is a list containing constraints between objects, each containing constraint functions taken from the above list of constraints and parameters being objects taken from the above list of objects.

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

    def step4_generate(self, batch, base_environment, main_characters_and_creatures, layout_plan):
        # Prompt for input
        processed_batch = self.step4_preprocess_data(batch, base_environment, main_characters_and_creatures, layout_plan)

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
        {"frame_start": frame_start, "frame_end": frame_end, "trajectory": [cordinate1, cordinate2, ...], "object": object, "object_action": action, "sound": sound}, 
        ...
            ]
}
Where total_frames represents the total duration of the video in frames, given as an integer. The motions field is a list of movements that occur in the video, where each motion is defined by the following elements:
- frame_start: The frame at which the motion begins.
- frame_end: The frame at which the motion ends.
- trajectory: A list of coordinates that define the path the object will follow. These points will later be used for interpolation to create a smooth trajectory.
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
    
    def step5_generate(self, batch, main_characters_and_creatures, layout_plan, list_of_object, object_initial_position):
        # Prompt for input
        processed_batch = self.step5_preprocess_data(batch, main_characters_and_creatures=main_characters_and_creatures, 
                                                     layout_plan=layout_plan,
                                                     list_of_object=list_of_object,
                                                     object_initial_position=object_initial_position)

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