import torch

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
        # Tokenize the input prompt
    inputs = tokenizer(step3_prompt, return_tensors="pt")

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=4096)

    # Decode the generated tokens back to text
    step3_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return step3_answer_format, step3_prompt, step3_response
