import torch
from utils import interact_with_lm, split_answer_from_respone

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
Respone: 
"""
    step5_prompt += "\nRespone:"
    step5_response, step5_all_hidden_state = interact_with_lm(tokenizer=tokenizer, model=model, prompt=step5_prompt, setting="peft_model")
    step5_response = split_answer_from_respone(respone=step5_response)
    
    return step5_answer_format, step5_prompt, step5_response, step5_all_hidden_state
