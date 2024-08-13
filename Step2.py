import torch

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
"""
    # Tokenize the input prompt
    inputs = tokenizer(step2_prompt, return_tensors="pt")

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=4096)

    # Decode the generated tokens back to text
    step2_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return step2_answer_format, step2_prompt, step2_response
