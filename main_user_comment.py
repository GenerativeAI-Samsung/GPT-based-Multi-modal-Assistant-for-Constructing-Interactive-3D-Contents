import json
from LanguageModel import ScenePlanningModel, TestScenePlanningModel

if __name__ == '__main__':
  
  # Open previous answer
  with open('step1_respone.json', 'r') as openfile:
    step1_respone = json.load(openfile)

  with open('step2_respone.json', 'r') as openfile:
    step2_respone = json.load(openfile)

  with open('step3_respone.json', 'r') as openfile:
    step3_respone = json.load(openfile)

  with open('step4_respone.json', 'r') as openfile:
    step4_respone = json.load(openfile)

  with open('step5_respone.json', 'r') as openfile:
    step5_respone = json.load(openfile)
  
  # Loading Model
  print("\n------------------------------------------------------")
  print("Loading Llama3 8B with adapter...")
  # MODEL_ID = "LoftQ/Meta-Llama-3-8B-4bit-64rank"
  # adapter_layers = []
  # scene_plan_model = ScenePlanningModel(MODEL_ID=MODEL_ID, adapter_layers=adapter_layers)
  scene_plan_model = TestScenePlanningModel()
  print("Done!")
  print("------------------------------------------------------")

  no_sastisfy = True
  while no_sastisfy:
    modify_command = input("Input your modified: ")

    respone = scene_plan_model.modified_user_request(batch=[modify_command],
                                            step1_respone=step1_respone,
                                            step2_respone=step2_respone,
                                            step3_respone=step3_respone,
                                            step4_respone=step4_respone,
                                            step5_respone=step5_respone)


    # Viết ra file .json và yêu cầu người dùng check lại
    json_object = json.dumps(respone, indent=4)
    with open("modify_step.json", "w") as outfile:
        outfile.write(json_object)
    print("You should check respone in modify_step.json to make sure that response reliable and executable!")
    control = None
    while (control != 'continue'):
        control = input('Press "continue" if done: ')
    
    # Đọc lại file .json để đến bước tiếp theo
    with open('modify_step.json', 'r') as openfile:
        respone = json.load(openfile)
    print("------------------------------------------------------")

    exec(respone[0])

    modify_step = min(change_step)

    if (modify_step == 1):
      pass
    if (modify_step == 2):
      pass
    if (modify_step == 3):
      pass
    if (modify_step == 4):
      pass
    if(modify_step == 5):
      pass

    control = input("Is that oke? ('yes' if yes, else 'no'): ")
    if (control == 'yes'):
      no_sastisfy = False


  prompt = """
You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
Your mission is to script the animation sequences for objects based on natural language descriptions, the list of objects and their initial positions

User has recently provided some feedback on your previous answer. Your task this time is to adjust the response to meet the user's feedback.

Natural language description: 
To create a 3D scene for the text "A girl plays with her dog in the garden," we can use the following details:

    The Setting:
        The scene takes place in a garden, which includes a grassy field.
        Consider adding elements to make it look like a children's playground.

    The Girl:
        She is young, possibly a little girl or a pre-teen.
        She is wearing a yellow sweater, a dress, and a pair of high heels.
        Her clothing is mostly bright and colorful.

    The Dog:
        The dog is medium-sized.
        The dog can be portrayed standing or interacting with the girl.

    Interaction:
        The girl is playing with the dog, so they should be interacting with each other in a playful manner.

Based on these details, you can imagine a vibrant and joyful scene. 
The girl, dressed in her colorful outfit, is happily playing with her medium-sized dog in a lush, green garden. 
The garden might include elements of a playground to enhance the playful atmosphere.

Objects and their constraints, initial position:
initial_position = {
  "garden_grassy_field": (0, 0, 0),
  "playground_slide": (5, 0, 0),
  "playground_swing": (8, 0, 0),
  "little_girl": (6, 1, 0),
  "medium_dog": (6, -1, 0)
}

your previous answer: 
trajectory = {
    "total_frames": 240,
    "motions": [
        {"frame_start": 1, "frame_end": 60, "trajectory": [(6, 1, 0), (7, 1, 0), (8, 1, 0)], "object": "little_girl"},
        {"frame_start": 1, "frame_end": 60, "trajectory": [(6, -1, 0), (7, -1, 0), (8, -1, 0)], "object": "medium_dog"},
        {"frame_start": 61, "frame_end": 120, "trajectory": [(8, 1, 0), (9, 2, 0), (10, 3, 0)], "object": "little_girl"},
        {"frame_start": 61, "frame_end": 120, "trajectory": [(8, -1, 0), (9, -2, 0), (10, -3, 0)], "object": "medium_dog"},
        {"frame_start": 121, "frame_end": 180, "trajectory": [(10, 3, 0), (11, 3, 0), (12, 3, 0)], "object": "little_girl"},
        {"frame_start": 121, "frame_end": 180, "trajectory": [(10, -3, 0), (11, -3, 0), (12, -3, 0)], "object": "medium_dog"},
        {"frame_start": 181, "frame_end": 240, "trajectory": [(12, 3, 0), (12, 2, 0), (12, 1, 0)], "object": "little_girl"},
        {"frame_start": 181, "frame_end": 240, "trajectory": [(12, -3, 0), (12, -2, 0), (12, -1, 0)], "object": "medium_dog"}
    ]
}

Your answer should be formatted as a dictionary with two main keys: total_frames and motions, where total_frames represents the total number of frames in the video, formatted as an integer, and motions is a list of motions that will occur in the video, where each element contains fields including start_frame, end_frame, and list the coordinates of the points through which the path will pass to later perform interpolation to create a trajectory."        

After determining your answer, structure them in this format:
trajectory = {{
    "total_frames": total_frame,
    "motions": [
        {{"frame_start": frame_start, "frame_end": frame_end, "trajectory": [cordinate1, cordinate2, ...], "object": object}}, 
        ...
            ]
}}
Avoid using normal text; format your response strictly as specified above.
"""
