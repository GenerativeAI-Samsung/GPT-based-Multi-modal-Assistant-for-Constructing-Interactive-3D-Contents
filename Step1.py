import torch

def step1(tokenizer, model, user_request):
    step1_answer_format = """
object_list = [
  {"name": x1, "description": y1},
  {"name": x2, "description": y2},
  {"name": x3, "description": y3},
  ...
]
Each asset is described with a concise name (x) and a detailed visual description (y).
"""
    step1_prompt = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets). 
After listing the assets, structure them in this format:
{step1_answer_format}

Avoid using normal text; format your response strictly as specified above.

Natural language description: {user_request}
"""
    # Tokenize the input prompt
    inputs = tokenizer(step1_prompt, return_tensors="pt")

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=4096)

    # Decode the generated tokens back to text
    step1_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return step1_answer_format, step1_prompt, step1_response 
