import torch
from utils import interact_with_lm, split_answer_from_respone

def step2(tokenizer, model, user_request, step1_respone):
    step2_answer_format = """
env_objs = [name_obj1, name_obj2, ...]
main_objs = [name_obj4, name_obj5, ...]

where env_objs contains objects used to create the base environment and main_objs objects contains the main characters and creatures in the animation.
"""
    step2_prompt = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural descriptions.
Your job is to classify the objects from the objects list below and natural descriptions into two groups: one group for objects used to create the base environment, and another group for objects that are the main characters and creatures in the animation.

Objects list:
{step1_respone}

Natural language description: {user_request}

After listing the assets, structure them in this format:
{step2_answer_format}

Avoid using normal text; format your response strictly as specified above.
Respone: 
"""
    
    step2_response = interact_with_lm(tokenizer=tokenizer, model=model, prompt=step2_prompt)
    step2_response = split_answer_from_respone(respone=step2_response)
    
    return step2_answer_format, step2_prompt, step2_response
