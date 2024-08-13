import torch

def step5(tokenizer, model, user_request, step3_response, initial_position):
    step5_answer_format = """
trajectory = {
    "total_frames": total_frame,
    "motions": [
        {"frame_start": frame_start, "frame_end": frame_end, "trajectory": [cordinate1, cordinate2, ...], "object": object}, 
        ...
            ]
}
where total_frames represents the total number of frames in the video, formatted as an integer, and motions is a list of motions that will occur in the video, where each element contains fields including start_frame, end_frame, and list the coordinates of the points through which the path will pass to later perform interpolation to create a trajectory.
"""
    step5_prompt = f"""
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your job is to script the animation sequences for objects based on natural language descriptions, the layouts, the list of objects and their initial positions.
Please think step by step.

Natural language description: {user_request}

Objects and their initial position:
{initial_position}

Layout plan:
{step3_response}

After determining your answer, structure them in this format:
{step5_answer_format}

Avoid using normal text; format your response strictly as specified above.
"""
        # Tokenize the input prompt
    inputs = tokenizer(step5_prompt, return_tensors="pt")

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=4096)

    # Decode the generated tokens back to text
    step5_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return step5_answer_format, step5_prompt, step5_response
