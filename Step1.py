import torch

from utils import interact_with_lm, split_answer_from_respone

def step1(tokenizer, model, user_request):

    step1_prompts = []

    step1_answer_format = """
object_list = [
  {"name": x1, "description": y1},
  {"name": x2, "description": y2},
  {"name": x3, "description": y3},
  ...
]
Each asset is described with a concise name (x) and a detailed visual description (y).
"""
    for request in user_request:
        step1_prompt = f"""
    You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
    Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets). 
    After listing the assets, structure them in this format:
    {step1_answer_format}

    Avoid using normal text; format your response strictly as specified above.

    Natural language description: {request} 
    """
        step1_prompt += "\nRespone:" 
        step1_prompts.append(step1_prompt)
    
    step1_response, step1_last_hidden_state = interact_with_lm(tokenizer=tokenizer, model=model, prompt=step1_prompts, setting="peft_model")
    
    step1_response = split_answer_from_respone(respone=step1_response)

    return step1_answer_format, step1_prompts, step1_response, step1_last_hidden_state
