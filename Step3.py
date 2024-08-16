import torch
from utils import interact_with_lm, split_answer_from_respone

def step3(tokenizer, model, user_request, step1_respone):
    step3_answer_format = """
layout_plan_i = {
"title": title_i,
"asset_list": [asset_name_1, asset_name_2],
"description": desc_i
}

where title_i is the high-level name for this step, and desc is detailed visual text description of what it shall look like after layout. 
"""
    step3_prompt = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to create a concrete plan to put them into the scene from the objects list below and natural descriptions.
Please think step by step, and give me a multi-step plan to put assets into the
scene.

Objects list:
{step1_respone}

Natural language description: {user_request}

For each step, structure your output as:
{step3_answer_format}

Avoid using normal text; format your response strictly as specified above.
"""
    step3_prompt += "\nRespone:"
    step3_response, step3_all_hidden_state = interact_with_lm(tokenizer=tokenizer, model=model, prompt=step3_prompt, setting="peft_model")
    step3_response = split_answer_from_respone(respone=step3_response)
    
    return step3_answer_format, step3_prompt, step3_response, step3_all_hidden_state
